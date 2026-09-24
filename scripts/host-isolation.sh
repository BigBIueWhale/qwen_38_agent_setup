# shellcheck shell=bash
#
# The host isolation rule: what the Docker daemon must report about container
# isolation before this deployment runs on the host.
#
# This file is byte-identical in agent_service and in the vLLM backend
# repository, Qwen_best_model_ever. Each repository pins its own copy in its
# own input manifest (config/build-inputs.sha256 and
# config/deployment-inputs.sha256), and neither reads the other's, so the two
# state the same rule exactly when the two recorded SHA-256 values are equal.
# A change is made to both copies alike, or to neither.
#
# The daemon reports isolation as `docker info` SecurityOptions: a JSON array
# of strings, each a comma-separated list of key=value attributes, one of which
# is `name`. The rule reads the report into options, names and attributes and
# asserts properties of that structure. It never compares report text, so a
# daemon that words a guarantee differently passes, and one that weakens a
# guarantee is refused whatever else in its report matches.
#
#   apparmor  must be reported: AppArmor confines containers. Where a profile
#             is reported it must be `default`, the builtin template every
#             container's docker-default profile is generated from. Docker
#             Engine 29.8.1 reports it as name=apparmor,profile=default; 29.7.2
#             reports no profile, because the builtin template is the only
#             one it can use.
#   seccomp   must be reported with profile=builtin: the daemon's builtin
#             filter confines every container. `unconfined` disables it, and any
#             other value substitutes another profile for it.
#   cgroupns  must be reported, with no attributes: every container gets a
#             private cgroup namespace by default.
#
# Every other daemon-wide mode a daemon can report is judged by one principle:
# a mode that can only add a restriction the deployment's containers already
# impose is acceptable; a mode that changes the model the stack runs under is
# not.
#
#   no-new-privileges  accepted, with no attributes. It is hardening: it sets
#                      the flag on every container by default, every container
#                      already sets it itself and is verified with it, and it
#                      cannot weaken any property above.
#   userns             refused: it changes the uid and gid mapping that every
#                      socket and file-ownership contract is stated in.
#   rootless           refused: it replaces the rootful system daemon, and with
#                      it the control socket and the mount and network model
#                      the stack is built on.
#   selinux            refused: it changes the LSM in force, from the AppArmor
#                      the containers are verified under to a policy none of
#                      the deployment's mounts is labelled for.
#
# An option name or attribute the rule does not know, a name reported twice,
# and a report that is not a JSON array of plain strings cannot be interpreted.
# Each one is a refusal: nothing unknown ever passes.

# Formats one refusal: what failed, what the rule requires, what the host
# reported, and the next action that repairs it.
host_isolation_refusal() {
  printf '%s\n  Required: %s\n  Reported: %s\n  Next:     %s\n' "$1" "$2" "$3" "$4"
}

# Reads REPORT, the output of `docker info --format '{{json .SecurityOptions}}'`.
# When the host satisfies the rule, prints nothing and returns 0. Otherwise
# prints the report followed by every refusal, and returns 1.
host_isolation_refusals() {
  if (($# != 1)); then
    printf 'host_isolation_refusals takes exactly one report; it received %d arguments.\n' "$#"
    return 1
  fi
  local report="$1" body option pair key name duplicated
  local plain='"[^"\[:cntrl:]]*"'
  local grammar='^[a-z][a-z0-9-]*=[^,=]+(,[a-z][a-z0-9-]*=[^,=]+)*$'
  local unknown_required='every option and attribute Docker reports is one this rule interprets; nothing unknown passes'
  local unknown_next='find out from the Docker Engine documentation which daemon setting produces it, then remove that setting and restart Docker, or extend scripts/host-isolation.sh to interpret it, identically in agent_service and Qwen_best_model_ever'
  local excluded_required='absent: a daemon-wide mode that changes the model the stack runs under is refused; only one that adds a restriction the containers already impose is accepted'
  local apparmor_required='AppArmor enabled with the default profile: name=apparmor, with profile=default wherever a profile is reported'
  local apparmor_next='remove the "apparmor-profile" setting from /etc/docker/daemon.json (or --apparmor-profile from the dockerd command line) and restart Docker'
  local seccomp_required="seccomp in force with the daemon's builtin profile: name=seccomp,profile=builtin"
  local seccomp_next='remove the "seccomp-profile" setting from /etc/docker/daemon.json (or --seccomp-profile from the dockerd command line) and restart Docker, so that it reports profile=builtin'
  local cgroupns_required='a private cgroup namespace for every container by default: name=cgroupns'
  local -a options=() refusals=()
  local -A attributes=() reported=() profiles=()

  if [[ "${report}" == null || "${report}" == '[]' ]]; then
    :
  elif [[ "${report}" =~ ^\[${plain}(,${plain})*\]$ ]]; then
    body="${report:2:${#report}-4}"
    while [[ "${body}" == *'","'* ]]; do
      options+=("${body%%'","'*}")
      body="${body#*'","'}"
    done
    options+=("${body}")
  else
    printf 'Docker reports these security options: %s\n' "${report:-<nothing>}"
    host_isolation_refusal \
      'The report cannot be interpreted: it is not a JSON array of plain strings.' \
      "${unknown_required}" "${report:-<nothing>}" \
      "confirm that docker info --format '{{json .SecurityOptions}}' prints a JSON array of strings; a Docker that reports another shape needs scripts/host-isolation.sh extended to read it, identically in agent_service and Qwen_best_model_ever"
    return 1
  fi

  for option in "${options[@]}"; do
    if [[ ! "${option}" =~ ${grammar} ]]; then
      refusals+=("$(host_isolation_refusal \
        'An option cannot be interpreted: it is not a list of key=value attributes.' \
        "${unknown_required}" "${option:-<empty string>}" "${unknown_next}")")
      continue
    fi
    attributes=()
    duplicated=false
    body="${option},"
    while [[ -n "${body}" ]]; do
      pair="${body%%,*}"
      body="${body#*,}"
      key="${pair%%=*}"
      if [[ -n "${attributes[${key}]+set}" ]]; then
        duplicated=true
      fi
      attributes["${key}"]="${pair#*=}"
    done
    if [[ "${duplicated}" == true || -z "${attributes[name]+set}" ]]; then
      refusals+=("$(host_isolation_refusal \
        'An option cannot be interpreted: it names no option, or repeats an attribute.' \
        "${unknown_required}" "${option}" "${unknown_next}")")
      continue
    fi
    name="${attributes[name]}"
    unset 'attributes[name]'
    case "${name}" in
      apparmor | seccomp | cgroupns | userns | rootless | selinux | no-new-privileges) ;;
      *)
        refusals+=("$(host_isolation_refusal \
          "Docker reports a security option this rule does not know: ${name}." \
          "${unknown_required}" "${option}" "${unknown_next}")")
        continue
        ;;
    esac
    if [[ -n "${reported[${name}]+set}" ]]; then
      refusals+=("$(host_isolation_refusal \
        "Docker reports ${name} more than once, so which report applies cannot be told." \
        "${unknown_required}" "${reported[${name}]} and ${option}" "${unknown_next}")")
      continue
    fi
    reported["${name}"]="${option}"

    case "${name}" in
      apparmor | seccomp)
        for key in "${!attributes[@]}"; do
          if [[ "${key}" != profile ]]; then
            refusals+=("$(host_isolation_refusal \
              "Docker reports ${name} with an attribute this rule does not know: ${key}." \
              "${unknown_required}" "${option}" "${unknown_next}")")
          fi
        done
        if [[ -n "${attributes[profile]+set}" ]]; then
          profiles["${name}"]="${attributes[profile]}"
        fi
        ;;
      cgroupns | no-new-privileges)
        for key in "${!attributes[@]}"; do
          refusals+=("$(host_isolation_refusal \
            "Docker reports ${name} with an attribute this rule does not know: ${key}." \
            "${unknown_required}" "${option}" "${unknown_next}")")
        done
        ;;
      userns)
        refusals+=("$(host_isolation_refusal \
          "The daemon remaps container users and groups (userns-remap), which changes the uid and gid mapping every socket and file-ownership contract of the deployment is stated in." \
          "${excluded_required}" "${option}" \
          'remove the "userns-remap" setting from /etc/docker/daemon.json (or --userns-remap from the dockerd command line) and restart Docker')")
        ;;
      rootless)
        refusals+=("$(host_isolation_refusal \
          "The Docker daemon runs rootless, which replaces the rootful system daemon the stack is built on: its control socket, and its mount and network model." \
          "${excluded_required}" "${option}" \
          'point the docker CLI at the system daemon: unset DOCKER_HOST and run docker context use default')")
        ;;
      selinux)
        refusals+=("$(host_isolation_refusal \
          "The daemon labels containers for SELinux (selinux-enabled), which changes the LSM in force from the AppArmor the containers are verified under to a policy none of the deployment's mounts is labelled for." \
          "${excluded_required}" "${option}" \
          'remove the "selinux-enabled" setting from /etc/docker/daemon.json (or --selinux-enabled from the dockerd command line) and restart Docker')")
        ;;
    esac
  done

  if [[ -z "${reported[apparmor]+set}" ]]; then
    refusals+=("$(host_isolation_refusal \
      'AppArmor cannot be confirmed enabled: the report holds no readable apparmor option.' \
      "${apparmor_required}" 'no readable apparmor option' \
      'enable AppArmor in the host kernel and load its profiles (on Ubuntu: sudo systemctl enable --now apparmor), then restart Docker (sudo systemctl restart docker)')")
  elif [[ -n "${profiles[apparmor]+set}" ]]; then
    case "${profiles[apparmor]}" in
      default) ;;
      unconfined)
        refusals+=("$(host_isolation_refusal \
          "AppArmor does not confine containers: the daemon's default profile is unconfined." \
          "${apparmor_required}" "${reported[apparmor]}" "${apparmor_next}")")
        ;;
      *)
        refusals+=("$(host_isolation_refusal \
          "AppArmor confines containers with a substituted profile template (${profiles[apparmor]}), not the default one." \
          "${apparmor_required}" "${reported[apparmor]}" "${apparmor_next}")")
        ;;
    esac
  fi

  if [[ -z "${reported[seccomp]+set}" ]]; then
    refusals+=("$(host_isolation_refusal \
      'seccomp cannot be confirmed in force: the report holds no readable seccomp option.' \
      "${seccomp_required}" 'no readable seccomp option' \
      'run Docker on a kernel with seccomp filtering (CONFIG_SECCOMP_FILTER=y) and a Docker Engine built with seccomp support, then restart Docker')")
  elif [[ -z "${profiles[seccomp]+set}" ]]; then
    refusals+=("$(host_isolation_refusal \
      'seccomp is reported without a profile, so the builtin profile cannot be confirmed.' \
      "${seccomp_required}" "${reported[seccomp]}" "${seccomp_next}")")
  else
    case "${profiles[seccomp]}" in
      builtin) ;;
      unconfined)
        refusals+=("$(host_isolation_refusal \
          "seccomp is disabled: the daemon's default profile is unconfined." \
          "${seccomp_required}" "${reported[seccomp]}" "${seccomp_next}")")
        ;;
      *)
        refusals+=("$(host_isolation_refusal \
          "seccomp applies a substituted profile (${profiles[seccomp]}), not the daemon's builtin one." \
          "${seccomp_required}" "${reported[seccomp]}" "${seccomp_next}")")
        ;;
    esac
  fi

  if [[ -z "${reported[cgroupns]+set}" ]]; then
    refusals+=("$(host_isolation_refusal \
      'A private cgroup namespace cannot be confirmed: the report holds no readable cgroupns option.' \
      "${cgroupns_required}" 'no readable cgroupns option' \
      'set "default-cgroupns-mode": "private" in /etc/docker/daemon.json (or --default-cgroupns-mode=private on the dockerd command line) on a cgroup v2 host and restart Docker')")
  fi

  if ((${#refusals[@]} == 0)); then
    return 0
  fi
  printf 'Docker reports these security options: %s\n' "${report}"
  printf '%s\n' "${refusals[@]}"
  return 1
}

# The rule's proof by example: every case is a report, the verdict the rule
# must reach on it, and for a refusal a phrase its refusal must contain. Each
# repository feeds every case through its own host check, so both prove the
# same verdicts on the path their operator commands run. Fields are separated
# by the ASCII unit separator, which no report contains.
host_isolation_cases() {
  local -a cases=(
    # Docker 29.7.2 / containerd 2.3.4.
    accept '["name=apparmor","name=seccomp,profile=builtin","name=cgroupns"]' ''
    # Docker 29.8.1 / containerd 2.3.5: the same guarantees, now naming the
    # AppArmor profile.
    accept '["name=apparmor,profile=default","name=seccomp,profile=builtin","name=cgroupns"]' ''
    # Order carries no meaning, neither of options nor of attributes.
    accept '["name=cgroupns","profile=builtin,name=seccomp","name=apparmor,profile=default"]' ''
    # A daemon-wide no-new-privileges only adds a restriction every container
    # already imposes. It never stands in for a required property, and it
    # carries no attributes.
    accept '["name=apparmor,profile=default","name=seccomp,profile=builtin","name=cgroupns","name=no-new-privileges"]' ''
    refuse '["name=apparmor","name=seccomp,profile=unconfined","name=cgroupns","name=no-new-privileges"]' 'seccomp is disabled'
    refuse '["name=apparmor","name=seccomp,profile=builtin","name=cgroupns","name=no-new-privileges,mode=all"]' 'no-new-privileges with an attribute this rule does not know: mode'
    refuse '["name=apparmor","name=seccomp,profile=unconfined","name=cgroupns"]' 'seccomp is disabled'
    refuse '["name=apparmor","name=seccomp,profile=/etc/docker/seccomp.json","name=cgroupns"]' 'seccomp applies a substituted profile'
    refuse '["name=apparmor","name=seccomp","name=cgroupns"]' 'seccomp is reported without a profile'
    refuse '["name=apparmor","name=cgroupns"]' 'seccomp cannot be confirmed in force'
    refuse '["name=seccomp,profile=builtin","name=cgroupns"]' 'AppArmor cannot be confirmed enabled'
    refuse '["name=apparmor,profile=unconfined","name=seccomp,profile=builtin","name=cgroupns"]' 'AppArmor does not confine containers'
    refuse '["name=apparmor,profile=/etc/docker/apparmor.tmpl","name=seccomp,profile=builtin","name=cgroupns"]' 'substituted profile template'
    refuse '["name=apparmor","name=seccomp,profile=builtin"]' 'private cgroup namespace cannot be confirmed'
    refuse '["name=apparmor","name=seccomp,profile=builtin","name=cgroupns","name=landlock"]' 'this rule does not know: landlock'
    refuse '["name=apparmor,mode=complain","name=seccomp,profile=builtin","name=cgroupns"]' 'attribute this rule does not know: mode'
    refuse '["name=apparmor","name=seccomp,profile=builtin","name=cgroupns,mode=host"]' 'attribute this rule does not know: mode'
    refuse '["name=apparmor","name=seccomp,profile=builtin","name=seccomp,profile=unconfined","name=cgroupns"]' 'more than once'
    refuse '["name=apparmor","name=seccomp,profile=builtin","name=cgroupns","name=userns"]' 'userns-remap'
    refuse '["name=apparmor","name=seccomp,profile=builtin","name=cgroupns","name=rootless"]' 'runs rootless'
    refuse '["name=apparmor","name=seccomp,profile=builtin","name=cgroupns","name=selinux"]' 'SELinux'
    refuse '["apparmor","name=seccomp,profile=builtin","name=cgroupns"]' 'not a list of key=value attributes'
    refuse '["name=apparmor","name=seccomp,profile=","name=cgroupns"]' 'not a list of key=value attributes'
    refuse '["name=apparmor","name=seccomp,profile=builtin","name=cgroupns,"]' 'not a list of key=value attributes'
    refuse '["name=apparmor","name=seccomp,profile=builtin,profile=builtin","name=cgroupns"]' 'repeats an attribute'
    refuse '["name=apparmor\u002cprofile=unconfined","name=seccomp,profile=builtin","name=cgroupns"]' 'not a JSON array of plain strings'
    refuse 'name=apparmor name=seccomp,profile=builtin name=cgroupns' 'not a JSON array of plain strings'
    refuse '' 'not a JSON array of plain strings'
    refuse 'null' 'AppArmor cannot be confirmed enabled'
    refuse '[]' 'seccomp cannot be confirmed in force'
  )
  printf '%s\x1f%s\x1f%s\n' "${cases[@]}"
}
