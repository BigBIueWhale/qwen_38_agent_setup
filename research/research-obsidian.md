# Obsidian, community plugins and a theme: security and suitability audit for an offline network

Audit date: 2026-09-22. Target: fully offline network. Desktops run Windows 10 Enterprise LTSC 21H2. Linux exists only in Ubuntu 24.04 agent containers with no network.

Legend: **[F]** = fact, with its source (file path, URL, commit, or a command I ran). **[I]** = my inference or recommendation.

---

## 0. Method and limitations

- **[F] Clones.** All repos were cloned into `/tmp/obsidian-audit/<name>` with `--depth 1`. I later deepened some of them to get tags and history (calendar, beautitab, kanban-community). HEAD of each at audit time:

| Repo (clone dir) | HEAD commit | Commit date |
|---|---|---|
| obsidianmd/obsidian-api (`obsidian-api`) | `cc1744324150c632416857c98964f87b1574a5fc` | 2026-07-14 |
| obsidianmd/obsidian-releases (`obsidian-releases`) | `32db40fe1701b1146a1e652e317c49dd1ca84b93` | 2026-09-22 |
| obsidianmd/obsidian-sample-plugin | `07ceb81d1fb3384af611ebf665a1ec42a7e5926d` | 2026-08-02 |
| obsidianmd/obsidian-help | `bc5b4f2b4fb1e0c873912fb0a8b769a1a6450f4a` | 2026-09-15 |
| obsidianmd/obsidian-developer-docs | `c56c7e770ba25dd0ea392aacf4588f9425970d36` | 2026-08-10 |
| liamcain/obsidian-calendar-plugin | `ef3f2696da11aa1d11a272179caea062d6144640` | 2022-11-04 |
| mgmeyers/obsidian-kanban (`obsidian-kanban`; GitHub redirects this URL) and obsidian-community/obsidian-kanban (`obsidian-kanban-community`) | `5134c05ad9a7861f551cd15d57cf50db8394901a` (identical HEAD in both) | 2026-03-06 |
| andrewmcgivery/obsidian-beautitab | `8b796a66bd05e1aaab61b09f48b8a346d323dea7` | 2024-03-26 |
| carbonateb/obsidian-encore-theme | `5365650dceaa5e3ae545017253d6f40316d26755` | 2024-06-09 |
| obsidianmd/jsoncanvas (extra) | `456f843cb293df4f4ab1763e22ccb46a80b307c8` | — |
| microsoft/winget-pkgs (extra, sparse checkout of `manifests/o/Obsidian`) | — | — |

- **[F] Beautitab owner in the brief is wrong.** `https://github.com/andrewbrereton/obsidian-beautitab` does not exist: `git clone` asked for credentials, which is how GitHub answers for a missing repo. The registry entry points to **andrewmcgivery/obsidian-beautitab**.
- **[F] What I could reach.**
  - Reachable: the GitHub API was not; git clone/fetch, `raw.githubusercontent.com`, GitHub release asset downloads, `registry.npmjs.org`, and WebSearch were.
  - Blocked by the egress proxy: `obsidian.md`, `help.obsidian.md`, `forum.obsidian.md`, NVD, `cvedetails`, `starlabs.sg`, `web.archive.org`.
  - Consequences:
    - Help and changelog statements come from the `obsidian-help` git repo, which is the source of help.obsidian.md.
    - CVE facts come from the CVEProject/cvelistV5 JSON on GitHub.
    - I could not read the privacy policy, license or terms pages directly. The privacy statement below is a search-engine summary.
    - Star and issue counts and the GitHub "archived" flag could not be checked because the API was blocked.
- **[F] Closed-source app analysed from binaries.** I downloaded the official 1.13.7 artifacts from `github.com/obsidianmd/obsidian-releases/releases/tag/v1.13.7` into `/tmp/obsidian-audit/app/`:
  - `Obsidian-1.13.7.exe`
  - `obsidian-1.13.7.tar.gz`
  - `obsidian-1.13.7.asar.gz`

  I extracted `resources/app.asar` (the loader) and `resources/obsidian.asar` (the app) with a small Python ASAR reader. Output is in `app/x-app/` and `app/x-obsidian/`. The code is minified; statements about app behaviour come from reading that code.

---

## 1. Key findings

1. **[F] Current version.** Obsidian desktop stable is **1.13.7**.
   - The public channel was set to 1.13.7 on **2026-08-12** (obsidian-releases commit `11fc3ae2`, "Update public to v1.13.7").
   - Early-access (Catalyst) channel: **1.14.2**, 2026-09-15.
   - The 1.13.7 build ships **Electron 43.3.0 / Chromium 150.0.7871.212 / Node 24.18.1**. Source: strings in the Linux `obsidian` binary.
2. **[F] Automatic updates are on by default.** The switch is **Settings → General → "Automatic updates"** (help page `en/Getting started/Update Obsidian.md`). It is stored as `"updateDisabled": true` in the global `obsidian.json` (`%APPDATA%\obsidian\obsidian.json`); see the app.asar and obsidian.asar code.
   - While updates are enabled, the loader runs a check at startup and **every 60 minutes**.
   - Each check first sends `GET https://releases.obsidian.md/desktop?id=<random 128-bit install ID>&v=<version>&p=<platform>`. The ID is stored in `%APPDATA%\obsidian\id`.
   - It then fetches `desktop-releases.json` from raw.githubusercontent.com, falling back to releases.obsidian.md.
   - Obsidian's own team network list does not mention the `releases.obsidian.md/desktop` ping for public updates.
3. **[F] Updates are verified, and can be verified offline.** In-app updates are gzipped ASARs checked with SHA-256 plus an RSA signature against a pinned certificate: `CN=dynalist.io`, valid 2016-05-16 to 2040-05-10. I verified the 1.13.7 asar.gz signature offline with `openssl dgst -verify`: **Verified OK**.
4. **[F] In-app updates only replace JavaScript.** They write `obsidian-<ver>.asar` into `%APPDATA%\obsidian` and never update Electron. Only a new installer updates Electron.
5. **[F] The Windows installer.**
   - Single NSIS "universal" EXE: `Obsidian-1.13.7.exe`, 331,012,528 bytes, SHA-256 `f233dc24896b3f2d5f9e4b01111181a561d0760b2105f0a474024c5f3143a9bc`.
   - Authenticode-signed by **"Dynalist Inc" (Oakville, Ontario, CA)**, through Microsoft ID Verified CS EOC CA 04. The signing certificate is short-lived (2026-08-11 to 2026-08-14).
   - winget ID **`Obsidian.Obsidian`**, switches `/currentuser` or `/allusers`.
   - No MSI and no portable EXE on GitHub Releases.
6. **[F] Community plugins run unsandboxed.** The main windows use `contextIsolation:false, nodeIntegration:true, nodeIntegrationInWorker:true, webviewTag:true` (obsidian.asar `main.js`), and `@electron/remote` 2.1.3 is bundled. Obsidian's own help says plugins "can access files on your computer … connect to internet … install additional programs".
7. **[F] Restricted mode is per device, not per vault.** It is on by default. The flag is stored in the app's Chromium localStorage as `enable-plugin-<vaultAppId>`, not in the vault. The list of enabled plugins, however, is a vault file: `.obsidian/community-plugins.json`.
   - **[I] Risk for the agent workflow:** anyone or anything that can write into a vault's `.obsidian/plugins/` can get code executed the next time a user opens that vault on a desktop where plugins are already trusted. This includes the LLM agents.
8. **[F] Undocumented managed-policy mechanism in 1.13.7.**
   - Trigger: the app name is `obsidian-work`, or `obsidian.json` contains `"isWork": true`.
   - The app then reads `C:\Program Files\Obsidian\policy.json` (Linux: `/etc/<appname>/policy.json`).
   - Keys: `plugins` and `themes` (true, or an allowlist array), `snippets`, `sync`, `publish`, `webViewer`, `devTools`, `insider`.
   - A missing or invalid file means deny-all, and work mode also disables auto-update.
   - **[I]** This looks like a feature still being built. It is not documented in the help repo and `isWork` sits in a user-writable file, so do not rely on it yet.
9. **Plugin and theme verdicts.**
   - **[F] Calendar 1.5.10:** MIT, no network code, **unmaintained since 2021-04**. My rebuild from source is **byte-identical** to the release.
   - **[F] Kanban 2.0.51:** GPL-3.0, no network code, boards are plain Markdown. The last release was 2024-05-31. The author announced on 2026-01-12 that he is looking for new maintainers, and the registry now lists it under "Obsidian Community Archive". My rebuild matches the release except for identifier renaming and one tslib helper.
   - **[F] Beautitab 1.6.1:** MIT; the last release was 2024-03-26. It **makes network calls by default**:
     - a version check to raw.githubusercontent.com on every load, which its README does not disclose;
     - background images from `source.unsplash.com`, the default;
     - quotes from `api.quotable.io`, the default.

     It also uses `fs` and `electron.remote.dialog`. My rebuild is byte-identical.
   - **[F] Encore theme 2.11.0:** CSS only, 90,414 bytes, MIT. No fonts, `@import` or `@font-face`. There is one opt-in remote background image (Unsplash). My rebuild from SCSS is byte-identical. The last commit was 2024-06-09.
10. **[F] License.** Obsidian is proprietary freeware. Since **2025-02-20** it is free for work use and the Commercial license is optional. The help page answers the air-gapped case directly: "In such specialized environments, you don't need to apply the commercial license directly to your installation."
11. **[F] No offline headless mode.**
    - The **Obsidian CLI** (added in 1.12, 2026-02-27; needs installer 1.12.7+) only controls a running GUI instance, through the Unix socket `$XDG_RUNTIME_DIR/.obsidian-cli.sock`.
    - **Obsidian Headless** (`npm obsidian-headless` 0.0.14, UNLICENSED, open beta) is only a client for Sync and Publish, which need an account and network.
    - **[I]** The agent containers should not install Obsidian at all. Vaults are plain files.
12. **[F] Known CVEs in the app.** All are old and fixed:
    - CVE-2022-36450, fixed in 0.15.5
    - CVE-2023-33244, fixed in 1.2.2
    - CVE-2023-2110, fixed in 1.2.8

    Plugin CVE: CVE-2021-42057 in Dataview. There was also real-world abuse of the plugin ecosystem in April 2026 ("PhantomPulse": a Shell Commands plugin delivered through a shared Sync vault), which was not an Obsidian vulnerability.

---

## 2. Obsidian desktop app (closed source)

### 2.1 Versions, dates, Electron

**[F] Channels** (`obsidian-releases/desktop-releases.json` @ `32db40fe`):
- Public: `latestVersion 1.13.7`, `minimumVersion 1.1.9`, download URL `…/v1.13.7/obsidian-1.13.7.asar.gz`, hash `aSU+OaoLmA48+W6eiopL7Wtkge9wIc12L2eHJmLY0lo=` (matches the file I downloaded).
- Beta: `1.14.2`, served from `https://releases.obsidian.md/release/obsidian-1.14.2.asar.gz`.

**[F] Channel history** (blobless history clone `releases-history/`):

| Commit | Date | Change |
|---|---|---|
| `11fc3ae2` | 2026-08-12 | public set to 1.13.7 |
| `dc095efe` | 2026-09-02 | beta set to 1.14.0 |
| `98a99933` | 2026-09-15 | beta set to 1.14.2 |

**[F] Changelog dates** (from `obsidian-help/Release notes/`):
- `v1.13.md`: desktop, 2026-07-30
- `v1.13.7.md`: 2026-08-11
- `v1.13.8.md`: 2026-08-20, "This release is Android-only."
- `v1.14.2.md`: 2026-09-15, insider

**[F] Electron versions stated in the release notes:**

| Release | Electron | Source line |
|---|---|---|
| 1.12 (2026-02-27) | 39.7.0 | `v1.12.md:82` |
| 1.12.7 (2026-03-23) | 39.8.3 | `v1.12.7.md:33` |
| 1.13 (2026-07-30) | 43.1.1 | `v1.13.md:129` |

The actual 1.13.7 Linux binary contains `Electron v43.3.0`, `Chrome/150.0.7871.212` and `v24.18.1` (Node). I extracted these with `strings` from `app/obsidian-1.13.7/obsidian`.

**[I]** Obsidian brings its own Chromium 150. That engine is separate from Edge 138 and Chrome 143 on the desktops, and its CVEs are fixed only by installing a new **installer**. It needs its own patch-tracking line.

### 2.2 Distribution formats

**[F] Windows.**
- One installer: `Obsidian-1.13.7.exe`, NSIS (`InstallerType: nullsoft`), covering x64, x86 and arm64.
- `/currentuser` or `/allusers` (the latter needs elevation).
- It registers the `obsidian:` protocol and the `.md` file extension. Source: `winget-pkgs/manifests/o/Obsidian/Obsidian/1.13.7/Obsidian.Obsidian.installer.yaml`, `ReleaseDate: 2026-08-12`.
- `License: Proprietary`, `LicenseUrl: https://obsidian.md/terms`.
- These names returned 404 on GitHub Releases: `.msi`, `-portable.exe`, `-allusers.exe`, `-arm64.exe`, `-32.exe`.
- Help (`Teams/Deploy Obsidian across your team.md`): "the Universal `.exe` includes the option to install Obsidian for all users."

**[F] Authenticode.**
- Signer: `O=Dynalist Inc, CN=Dynalist Inc`, issued by Microsoft ID Verified CS EOC CA 04.
- Certificate validity: 2026-08-11 to 2026-08-14. I parsed the PE security directory and read it with `openssl pkcs7`.
- **[I]** Because the certificate lives only three days, AppLocker/WDAC rules should be publisher rules on "Dynalist Inc", not certificate-hash rules. I did not check the inner `Obsidian.exe` signature (no 7-Zip in the sandbox).

**[F] Linux assets for v1.13.7** (size / Last-Modified):

| Asset | Size (bytes) | Last-Modified |
|---|---|---|
| `Obsidian-1.13.7.AppImage` | 136,902,072 | 2026-08-12 |
| `Obsidian-1.13.7-arm64.AppImage` | 135,373,906 | 2026-08-12 |
| `obsidian_1.13.7_amd64.deb` | 107,021,628 | 2026-08-12 |
| `obsidian-1.13.7.tar.gz` | 129,462,096 | 2026-08-12 |
| `obsidian-1.13.7-arm64.tar.gz` | 127,754,080 | 2026-08-12 |
| `obsidian-1.13.7.asar.gz` (update payload) | 8,773,048 | 2026-08-12 |

- Also present: `Obsidian-1.13.7.dmg`, and `Obsidian-1.13.7.apk` (15,448,046 bytes).
- Help lists **Snap** (download from obsidian.md, install with `--dangerous --classic`), **AppImage** (`--no-sandbox`) and **Flatpak** (`flatpak install flathub md.obsidian.Obsidian`).
- I found no `.snap` asset under the usual names on GitHub.

### 2.3 Update mechanism (installer version vs app version)

- **[F] Help** (`en/Getting started/Update Obsidian.md`):
  - "Obsidian on desktop devices regularly checks for new updates. If automatic updates are enabled, the application will update on restart."
  - "Open **Settings → General**. … Disable **Automatic updates**."
  - The installer version "is the version of Electron … and it cannot be updated by the automatic update process."
- **[F] Loader code** (`resources/app.asar` → `main.js`, 13,407 bytes, readable):
  - Creates `%APPDATA%\obsidian\id` holding 16 random bytes in hex.
  - `update()` returns early if `disable` is set and the check is not manual.
  - Otherwise it calls `httpGetBinary('https://releases.obsidian.md/desktop?id='+id+'&v='+version+'&p='+process.platform)`. The code comment says: "Hopefully this can help us catch any updater failures".
  - It then GETs `https://raw.githubusercontent.com/obsidianmd/obsidian-releases/master/desktop-releases.json`, falling back to `https://releases.obsidian.md/desktop-releases.json`.
  - It downloads `downloadUrl` and checks SHA-256 plus `RSA-SHA256` against `SIGNATURE_CERT`, then writes `obsidian-<ver>.asar` into `%APPDATA%\obsidian`.
  - It runs `setInterval(queueUpdate, 60*60*1000)`.
  - If the installer (Electron) version is below `minimumVersion`, it emits `update-manual-required`.
- **[F] App main process** (`obsidian.asar/main.js`):
  - Reads `obsidian.json` synchronously at load and runs `if ((isWork || D.updateDisabled)) emit("disable", true)`.
  - The IPC handler `disable-update` sets or deletes `D.updateDisabled`.
  - "Check for updates" briefly re-enables the check for a manual run.
- **[F] At startup the loader loads the highest-versioned `obsidian-*.asar` in `%APPDATA%\obsidian`** if it is at least the installer version. The signature is checked only at download time, not at load time.
  - **[I]** This is a user-writable code path that AppLocker/WDAC do not govern (it is JavaScript inside an ASAR). Treat `%APPDATA%\obsidian\*.asar` as code, and prefer installer-based updates through the approval process.
- **[I] Offline update path.** Bring each new signed installer through the approval process. The asar.gz plus the `signature` in `desktop-releases.json` can be verified offline with openssl; the commands are in section 6. An app-internal `copy-asar` IPC exists, but manual ASAR installation is not documented, so do not rely on it.

### 2.4 Network connections the desktop app can make

Sources: **[F]** code in `app/x-app/main.js`, `app/x-obsidian/main.js` and `app/x-obsidian/app.js`, plus help `en/Teams/Security considerations for teams.md`. Rows marked [I] are inference.

| Trigger | Endpoint(s) | Default |
|---|---|---|
| Update check: startup, every 60 min, and "Check for updates" | `releases.obsidian.md/desktop?id=…&v=…&p=…` (ping), `raw.githubusercontent.com/obsidianmd/obsidian-releases/master/desktop-releases.json`, fallback `releases.obsidian.md/desktop-releases.json`, asar.gz from `github.com/.../releases/download/` (beta: `releases.obsidian.md/release/`) | **On** unless `updateDisabled` |
| Plugin deprecation list | `raw.githubusercontent.com/obsidianmd/obsidian-releases/HEAD/community-plugin-deprecation.json` at startup and every 12 h (`432e5` ms); skipped when no plugin is loaded | Only when community plugins are loaded |
| Automatic plugin update check | GitHub (manifests, releases); every 3 days on window focus | **Off** (localStorage `plugin-automatically-check-for-updates` defaults to false) |
| Plugin and theme browser, install, update | `raw.githubusercontent.com/obsidianmd/obsidian-releases/HEAD/{community-plugins.json, community-plugin-stats.json, community-css-themes.json}`, `github.com/<repo>/releases/download/<ver>/{manifest.json, main.js, styles.css, theme.css}`, fallback `raw…/<repo>/HEAD/…`, proxy fallback `releases.obsidian.md/proxy?url=`, theme stats `releases.obsidian.md/stats/theme` and a POST to `…/stats/theme/<name>/download` on install | Only when the user opens Browse or Manage |
| Release notes view after an update | `raw.githubusercontent.com/obsidianmd/obsidian-docs/HEAD/Release%20notes/v<ver>.md` | After an update |
| Web viewer core plugin | Any URL the user opens; ad-block lists `easylist.to/easylist/easylist.txt` and `easyprivacy.txt`, refreshed every 4 days when a web session is created | Plugin `defaultOn=false` |
| Canvas link cards, `<iframe>`, remote `![](https://…)` images in notes, YouTube or Twitter embeds (`releases.obsidian.md/youtube?v=`, `platform.twitter.com`) | Content-driven | When such content is opened |
| Account, Commercial-license activation, Sync, Publish | `api.obsidian.md`, `sync-xx.obsidian.md`, `publish-main.obsidian.md` / `publish-xx.obsidian.md` (from help) | Only when used |
| Spellcheck dictionaries | [I] Electron's default downloads Hunspell dictionaries from a Google CDN on Windows and Linux (Electron docs: "the hunspell dictionary files are downloaded from a Google CDN by default"). Obsidian calls `setSpellCheckerLanguages` and I found no `setSpellCheckerDictionaryDownloadURL`. | [I] likely on first use of a language; confirm in the pilot |
| Crash reporting | none: the string `crashReporter` is absent from all three JS bundles [F] | — |

**[F] Obsidian's own statements.**
- "Obsidian is designed to function as an offline and standalone application."
- "Obsidian does make network calls based on the services and features you use. These network connections can be disabled via a domain firewall or application lockdown. Obsidian makes these network connections on HTTPS port 443." (Teams security page)
- **[I]** Offline, all of these calls fail. The update code swallows `net::ERR` errors, so core editing is not affected. Expect DNS and proxy log noise for `raw.githubusercontent.com` and `releases.obsidian.md` unless updates are disabled.

**[F] Privacy statement.** I could not fetch `obsidian.md/privacy` (proxy block). A search-engine summary of that page says Obsidian "do[es] not collect any personal data and do[es] not collect any telemetry data … all data is saved locally". This is **not verbatim; verify on the page itself.**
- Developer policy (verbatim, `obsidian-developer-docs/en/Community directory/Developer policies.md`): plugins and themes must not "Include client-side telemetry."
- **[I]** The update ping carries a persistent pseudonymous install ID, version and platform, every hour while updates are enabled. Whether that counts as "telemetry" is a judgement call. Either way, it goes away with `updateDisabled`.

### 2.5 Security model

- **[F] Plugin security, verbatim** (help `en/Extending Obsidian/Plugin security.md`):
  - "By default, Obsidian runs in Restricted Mode to prevent third-party code execution."
  - "Due to technical limitations, Obsidian cannot reliably restrict plugins to specific permissions or access levels. This means that plugins will inherit Obsidian's access levels." They can "access files on your computer … connect to internet … install additional programs."
  - "If you're working with sensitive data and wish to install a community plugin, we recommend that you perform an independent security audit on the plugin before using it."
  - Community plugins page: "Community plugins run third-party code on your behalf that could potentially do harm."
  - Web viewer page: "Obsidian plugins are not sandboxed and have deep control over the app … third-party plugins have full access to cookies in Web viewer."
- **[F] Directory scanning.** "Obsidian automatically scans every plugin version for security vulnerabilities, code quality issues, and malware." Each listing shows a "safety scorecard", including disclosures such as clipboard or network use. The developer docs add that the directory "verifies that the build matches what's committed."
- **[F] Plugin updates.** "For security purposes, community plugins don't update automatically." Themes also don't update automatically.
- **[F] Electron settings.** `webPreferences:{contextIsolation:!1,nodeIntegration:!0,nodeIntegrationInWorker:!0,spellcheck:!0,webviewTag:!0,…,devTools:<policy>}`. Obsidian also exposes `require('electron').remote` through the bundled `@electron/remote`.
- **[F] What Restricted mode covers.**
  - Themes and CSS snippets load regardless of Restricted mode; only the undocumented policy gates them.
  - 1.13 added "the ability to exit Restricted Mode without re-enabling your plugins" (`v1.13.md`).
  - When a vault that contains plugins is opened on a device with no stored trust decision, a trust modal appears (`app.js`: modal opened when `localStorage["enable-plugin-"+appId]` is null).
- **[F] HTML in notes is sanitized** (help `HTML content.md`; DOMPurify is in `app.js`).
- **[F] Obsidian URI.** Actions: `open`, `new` (with `content`, `clipboard`, `append`, `overwrite`, `silent`), `daily`, `unique`, `search`, `choose-vault`, plus Hook integration. The installer registers the `obsidian` protocol.
  - **[I]** Clicking links in browsers or mail can therefore create or overwrite notes. De-register `HKCU\Software\Classes\obsidian` if it is not needed.
- **[F] Undocumented policy.json** (`obsidian.asar/main.js`, functions `wo()`, `ao()`, `vo()`; `app.js` `Db(Ab.plugins,id)` gates `enablePlugin`, and `Ab.themes`/`snippets`/`sync`/`publish`/`webViewer` gate the matching features): see Key finding 8.

### 2.6 Installing plugins and themes by hand (offline)

**[F] Plugins.**
- Help (`Teams/Deploy…`): "Plugins are located in the `.obsidian/plugins` folder within a vault, and can be installed manually at this location."
- The app reads `<configDir>/plugins/<id>/manifest.json`, `main.js` and optional `styles.css` (constants `eL`/`tL`/`nL` in `app.js`).
- The enabled list is `.obsidian/community-plugins.json` (a JSON array of IDs).
- Restricted mode must be turned off once per device and vault: Settings → Community plugins → Turn on community plugins.
- The registry README states that Obsidian downloads `manifest.json`, `main.js` and `styles.css` from the GitHub release whose tag equals `manifest.json.version`.

**[F] Themes.**
- Files: `.obsidian/themes/<Name>/theme.css` plus `manifest.json` (`getThemePath()` in `app.js`).
- The active theme is stored as `"cssTheme": "<Name>"` in `.obsidian/appearance.json`; enabled snippets are in `enabledCssSnippets`.
- On install the app tries release assets first, then falls back to `raw.githubusercontent.com/<repo>/HEAD/{manifest.json, theme.css}`.

**[F] Config folder.** `.obsidian/` inside each vault. Global settings live in `%APPDATA%\Obsidian\` on Windows and `~/.config/obsidian/` on Linux (help `How Obsidian stores data.md`).

**[F] Enterprise locking.** Help FAQ: features can be locked "by blocking edit access to the `.obsidian` folder, or specific files and folders within it."

### 2.7 Vault format and suitability for LLM agents in network-less Linux containers

- **[F] Plain files.** "Obsidian stores your notes as Markdown-formatted plain text files … you can use other text editors and file managers to edit and manage notes. Obsidian automatically refreshes your vault to keep up with any external changes." (help `How Obsidian stores data.md`)
- **[F] File types.** `.md`, `.base` (Bases, **YAML**), `.canvas` (**JSON Canvas 1.0**, spec dated 2024-03-11, MIT; `jsoncanvas/spec/1.0.md`: top-level `nodes[]` and `edges[]`; node types `text`/`file`/`link`/`group`), plus images, audio, video and PDF (help `Accepted file formats.md`).
- **[F] Obsidian-specific syntax** (help `Obsidian Flavored Markdown.md`):

| Syntax | Meaning |
|---|---|
| `[[Link]]`, `[[Note#Heading]]`, `[[Note#^block]]`, `[[Note\|alias]]` | wikilinks |
| `![[embed]]` | embeds |
| `^blockid` | block IDs |
| `%%comment%%` | comments |
| `==highlight==` | highlights; 1.14 adds colour-emoji highlights like `==🔵 …==` |
| `> [!note]` | callouts |
| `[^fn]` | footnotes |
| YAML frontmatter | properties |
| `#tags` | tags |
| ` ```base ` code blocks | embedded Bases queries |

- **[I]** Dataview queries (` ```dataview `, `dataviewjs`) belong to a community plugin, not core. DataviewJS runs JavaScript (see CVE-2021-42057).
- **[F] Kanban boards are plain Markdown** (source `src/parsers/common.ts:9,25-40`, `src/parsers/formats/list.ts:403-451`):
  - frontmatter `kanban-plugin: board`
  - lanes as `## Lane title`
  - cards as task items `- [ ] …` / `- [x] …`
  - optional `**Complete**` marker in a lane
  - archive after `***` under `## Archive`
  - board settings in a `%% kanban:settings` block holding a fenced JSON code block and closing `%%`
  - dates and times inline as `@{…}` and `@@{…}` (default triggers `@` and `@@` in `src/settingHelpers.ts:9-10`)
- **[F] Tested with standard tools only.** In this Ubuntu-based sandbox I parsed a sample vault using only the Python stdlib and regexes: frontmatter, wikilinks, embeds, callouts, Kanban lanes and cards, and `jsoncanvas/sample.canvas` all parsed. PyYAML 6.0.1 is present here, but that does not mean it is in the org's image.
- **[I] Recommendations for agents:**
  - Treat `.obsidian/` as **code and configuration, not content**. Mount it read-only or exclude it from agent write scope, and diff-review any changes to `plugins/`, `snippets/`, `themes/`, `community-plugins.json` and `core-plugins.json`.
  - Keep Restricted mode on for vaults that come back from agents.
  - Canvas and Bases files are safe structured data (JSON/YAML), but validate them against the spec before a human opens them.

### 2.8 CLI and headless use

- **[F] Obsidian CLI** (help `Extending Obsidian/Obsidian CLI.md`, `Release notes/v1.12.md`, `v1.12.7.md`):
  - Added in 1.12 (2026-02-27); needs installer 1.12.7+.
  - Enabled in Settings → General → "Command line interface". It is off by default: the `cli` key is absent from `obsidian.json` until set.
  - "Obsidian CLI requires the Obsidian app to be running."
  - Commands include `read`, `create`, `search`, `daily:append`, `base:query`, `plugin:install`, `eval code=…` (runs JavaScript in the app), `dev:cdp` and `dev:screenshot`.
  - On Linux, `obsidian-cli` (18,576 bytes, libc-only ELF) is a thin client that connects to `$XDG_RUNTIME_DIR/.obsidian-cli.sock` and otherwise prints "The CLI is unable to find Obsidian. Please make sure Obsidian is running". On Windows the redirector is `Obsidian.com`.
- **[F] Obsidian Headless** (help `Obsidian Headless.md`): "a headless client for Obsidian services" for Sync and Publish. It needs Node 22+ and `ob login`. npm `obsidian-headless` is version 0.0.14, `license: UNLICENSED`, modified 2026-07-30.
- **[I]** Neither is useful offline. The containers need no Obsidian binary. On desktops, keep the CLI off, because its `eval` makes it a local code-execution surface.

### 2.9 License

- **[F] Free for work.** Obsidian announcement on X (@obsdmd, status 1892586092882276352), dated 2025-02-20 per AlternativeTo and other coverage: "Obsidian is now free for work. Starting today, the Obsidian Commercial license is optional. Anyone can use Obsidian for work, for free. If Obsidian benefits your organization, you can still purchase Commercial licenses to support development. Nothing else is changing." The blog post is `https://obsidian.md/blog/free-for-work/` (could not fetch).
- **[F] Help** (`en/Teams/Commercial license.md`):
  - "Obsidian is free to use and is 100% user-supported. Organizations can choose to support Obsidian by purchasing Commercial licenses."
  - "Note that applying a commercial license does not provide any functional benefits within the app."
  - "Am I required to purchase a commercial license for every employee? No. Obsidian is 100% free to use".
  - Air-gapped FAQ: "In such specialized environments, you don't need to apply the commercial license directly to your installation."
  - The Commercial license still exists as an optional purchase; 25+ seats earn an Enterprise-page listing.
- **[F] Embedded terms.** The app's `package.json` says `"license": "UNLICENSED"`. `app.js` contains this string: "I understand and agree that I am not allowed to distribute the Obsidian application, in any form, without explicit approval from the Obsidian team. I also understand that Obsidian is a registered trademark…" The renderer refuses to start unless the main process returns exactly this text (IPC `terms`).
  - **[I]** Deploying inside the organisation is what Obsidian's own "Deploy Obsidian across your team" page describes. Still, legal should read `obsidian.md/terms` and `obsidian.md/license`, which I could not fetch, especially for mirroring the installer on an internal repository.
- **[F] Licenses of the public repos:**

| Repo | License |
|---|---|
| obsidian-api | MIT ("Copyright 2022 Dynalist Inc.") |
| obsidian-sample-plugin | 0-BSD |
| jsoncanvas | MIT |
| obsidian-help, obsidian-developer-docs, obsidian-releases | no LICENSE file |

  **[I]** Mirroring obsidian-help internally as offline docs, which is a ready-made vault, needs a legal check.

### 2.10 CVEs and advisories

**[F]** From `CVEProject/cvelistV5` JSON fetched via raw.githubusercontent.com (saved in `/tmp/obsidian-audit/cve/`):

| CVE | Published | Affected / fixed | CVSS 3.1 | Summary |
|---|---|---|---|---|
| CVE-2022-36450 | 2022-07-25 | 0.14.x and 0.15.x before **0.15.5** | 8.0 | `obsidian://hook-get-address` RCE, because `window.open` was used without checking the URL |
| CVE-2023-33244 | 2023-05-20 | before **1.2.2** | 8.2 (ADP) | An embedded web page could call unintended APIs (microphone, camera, notifications) |
| CVE-2023-2110 | 2023-08-19 | before **1.2.8** | 8.2 (`AV:L/AC:L/PR:N/UI:R/S:C/C:H/I:H`) | `app://local/<absolute-path>` local file disclosure through a crafted note or paste (STAR Labs) |
| CVE-2021-42057 (Dataview plugin) | 2021-11-04 | Dataview ≤ 0.4.12-hotfix1 | — | `evalInContext` eval injection from Markdown |

- **[F] Excluded false positives:**
  - CVE-2025-56449 is "Obsidian Scheduler", a different product.
  - CVE-2026-20841 is Windows Notepad. A forum post (`forum.obsidian.md/t/…/111160`) claimed a similar issue in Obsidian and sits in the forum's "Bug graveyard"; it is unverified.
- **[F] Audits** (secondary sources via search):
  - Cure53's second client audit concluded in October 2024. Its six issues were in the then-unreleased Web viewer and were fixed before 1.8.0.
  - The help Web viewer page links to `obsidian.md/blog/cure53-second-client-audit/`.
  - A Trail of Bits audit of Sync is reported for December 2025 (not verified).
- **[F] Threat intel.** Elastic Security Labs, "Phantom in the vault" (April 2026): social engineering led victims to open an attacker's shared Sync vault and enable community-plugin sync. The Shell Commands plugin then ran PowerShell or AppleScript loaders. This was plugin abuse, not an Obsidian vulnerability.
- **[I]** No app CVE after 2023 turned up in my searches, but NVD and cvedetails were blocked, so coverage may be incomplete. Re-run an NVD search from the approval network.

---

## 3. The public Obsidian repositories

- **[F] obsidian-api** (`obsidian.d.ts`, 8,498 lines; `package.json` version 1.13.2, MIT; dependencies `@types/codemirror 5.60.8`, `moment 2.30.1`):
  - `request()` and `requestUrl()` are "Similar to `fetch()`, request a URL using HTTP/HTTPS, without any CORS restrictions" (lines 5431-5442).
  - `Platform.isDesktopApp`, `isMobileApp`, `isWin`, `isLinux` (line 4823).
  - `FileSystemAdapter.getBasePath()` and `getFullPath()` return absolute OS paths (lines 2996-3095).
  - `SecretStorage`, since 1.11.4 (line 5635).
  - `registerObsidianProtocolHandler` (line 5028).
  - The README says: "Import NodeJS or Electron API using `require('fs')` or `require('electron')`" and describes `isDesktopOnly` as "whether your plugin uses NodeJS or Electron APIs".
- **[F] obsidian-releases.**
  - `community-plugins.json`: 7,919 entries. `community-css-themes.json`: 789.
  - Both are now mirrored hourly from `community.obsidian.md/assets/…` (`.github/workflows/mirror-community-json.yml`). Stats are pulled daily (`plugin-stat.yml`).
  - `plugin-review.md` now says the content moved to `docs.obsidian.md/Plugins/Releasing/Plugin+guidelines`.
  - `.github/pull_request_template.md` only links the plugin and theme templates.
  - Registry entries found:
    - `{"id":"calendar","author":"Liam Cain","repo":"liamcain/obsidian-calendar-plugin"}`
    - `{"id":"obsidian-kanban","author":"Obsidian Community Archive","repo":"obsidian-community/obsidian-kanban"}`
    - `{"id":"beautitab","author":"andrewmcgivery","repo":"andrewmcgivery/obsidian-beautitab"}`
    - theme `{"name":"Encore","author":"carbonateb","repo":"carbonateb/obsidian-encore-theme","modes":["dark","light"]}`. This is the **only** Encore match.
- **[F] obsidian-sample-plugin.**
  - 0-BSD; esbuild 0.25.5; TypeScript ^5.8.3; `eslint-plugin-obsidianmd` ^0.4.0.
  - Its `.gitignore` says: "Don't include the compiled main.js file in the repo. They should be uploaded to GitHub releases instead."
- **[F] obsidian-developer-docs, Developer policies.** Plugins and themes must not:
  - "Obfuscate code"
  - "Include client-side telemetry"
  - "Install or update themselves or their dependencies"

  Also: "Themes may not load assets from the network." Network use must be disclosed in the README.
  - Submission requirement: Node or Electron use requires `isDesktopOnly: true`.
  - Theme guidelines: no remote fonts or images.
  - Contradiction: the Teams security help page says there is "an exception for Google Fonts".

---

## 4. Plugins and theme

Released assets were downloaded to `/tmp/obsidian-audit/releases/<plugin>-<ver>/`. Rebuilds are in `/tmp/obsidian-audit/build/`. Toolchain: Node 22.22.2, npm 10.9.7, yarn 1.22.22.

**Grep patterns.** Source and released `main.js` were searched for `fetch(`, `XMLHttpRequest`, `requestUrl(`, `WebSocket`, `EventSource`, `http://`, `https://`, `new Function`, `eval(`, `child_process`, `require('fs')`, `require('electron')`, `openExternal`, `window.open`, telemetry/analytics/sentry/posthog/google, `@import url`, `@font-face`, `localStorage`, `sessionStorage` and `sendBeacon`. Plain `https://` hits in READMEs, manifests or comments are documentation, not network calls, and are listed only where relevant.

### 4.1 Calendar (liamcain)

**[F] Summary**

| Field | Value |
|---|---|
| Repository | https://github.com/liamcain/obsidian-calendar-plugin |
| License | MIT (`LICENSE`, "Copyright (c) 2021 Liam Cain") |
| Latest stable | **1.5.10**, tag commit `7d2aebd`, 2021-04-01 |
| Latest pre-release | 2.0.0-beta.2, 2021-04-20 (stats `updated` 2021-04-20T18:50Z) |
| Last commit | 2022-11-04 (`ef3f269`, "Add funding URL") |
| Releases | 54 tags; 52 have release `main.js` (1.0.0 and 1.1.1 have none) |
| Downloads | 3,126,086 (stats file @ `32db40fe`) |
| manifest | id `calendar`, version 1.5.10, minAppVersion 0.9.11, `isDesktopOnly: false`; `versions.json`: 1.5.10 → 0.11.0 |

**[F] Dependencies**
- `dependencies`: `obsidian: obsidianmd/obsidian-api#master` (a git dependency), `obsidian-calendar-ui 0.3.12`, `obsidian-daily-notes-interface 0.9.0`, `svelte 3.35.0`, `tslib 2.1.0`.
- `devDependencies`: `rollup 2.44.0`, `@rollup/plugin-commonjs 18.0.0`, `@rollup/plugin-node-resolve 11.2.1`, `@rollup/plugin-typescript 8.2.1`, `rollup-plugin-svelte 7.1.0`, `svelte-preprocess 4.7.0`, `svelte-check 1.3.0`, `@tsconfig/svelte 1.0.10`, `typescript 4.2.3`, `eslint 7.23.0`, `@typescript-eslint/* 4.20.0`, `jest 26.6.3`, `ts-jest 26.5.4`, `svelte-jester 1.3.2`, `@types/jest 26.0.22`, `@types/moment 2.13.0`, `moment 2.29.1`.

**[F] Build artifacts**
- `main.js` is **not committed**; it exists only as a release asset.
- Released `main.js` for 1.5.10: **141,498 bytes**, SHA-256 `7fb339e9cf9fdbe5a801fa2b8ab85b366b5b3777fbd193cbc8728bc27711d125`. The release has no `styles.css` (404).
- Toolchain: **rollup** (`rollup.config.js`), with externals `obsidian`, `fs`, `os`, `path`.

**[F] Offline rebuild: byte-identical.** Worktree at tag 1.5.10; `yarn install --ignore-scripts`; `npx rollup -c`. The lockfile pins `obsidian` to a `codeload.github.com` tarball (commit `dbfa19ad`), which the proxy blocked. I replaced it with npm `obsidian@0.11.13` via `resolutions`. This has no effect on the bundle, because `obsidian` is external and used only for types. `node_modules` is 223 MB.

**[F] Grep results**
- Source: only documentation URLs.
  - `README.md:3,5,25,33,109,163,165,190-202`
  - `manifest.json:7-8`
  - `.github/FUNDING.yml:2`
- Released `main.js` 1.5.10:
  - `navigator.language` (lines 894, 3818) and `navigator.appVersion` (2088)
  - `localStorage.getItem("language")` (3817), which reads Obsidian's UI language for locale
  - four `http://www.w3.org` SVG namespaces
  - **No** `fetch`, XHR, `requestUrl`, WebSocket, eval, `new Function`, `fs`, `electron`, `child_process` or analytics.
- The 2.0.0-beta.2 build is not what gets installed; Obsidian installs the version in `manifest.json`. It does embed `<img src="https://cdn.buymeacoffee.com/…">` (line 9117), a remote image in its settings UI.

**[F] Dependencies on other plugins**
- README: "The plugin reads your Daily Note settings to know your date format, your daily note template location…". `main.js` uses `internalPlugins.getPluginById("daily-notes")` (the **Daily notes core plugin**).
- Weekly notes: README lines 163-167 say they were split out into the **Periodic Notes** community plugin; the code calls `plugins.getPlugin("periodic-notes")`.
- `window.moment` is Obsidian's bundled moment.

**[I] Assessment.** Offline-safe with a small attack surface. Maintenance risk is high: no release for about 5.5 years, and it relies on internal APIs such as `internalPlugins` that may break in future Obsidian versions. Pin the SHA-256 above and test against 1.13.7 in the pilot.

### 4.2 Kanban (mgmeyers → obsidian-community → community-archive)

**[F] Summary**

| Field | Value |
|---|---|
| Repository | https://github.com/mgmeyers/obsidian-kanban. Registry now lists `obsidian-community/obsidian-kanban`. Both URLs currently HTTP-301 to `github.com/community-archive/obsidian-kanban` (observed on release downloads). |
| License | **GPL-3.0** (`LICENSE.md`, added 2021-04-23 in commit `56b6499`). `package.json` says `"license": "MIT"`, which is inconsistent. [I] Treat it as GPL-3.0: no obligations for internal use unless modified versions are distributed. |
| Latest release | **2.0.51**, tag commit `8501981`, 2024-05-30 18:07 −07:00 (stats `updated` 2024-05-31T01:08Z) |
| Last commit | 2026-03-06 (`5134c05`, "Remove funding"). The only commits after 2.0.51: `e833d34` "Create MAINTAINERS.md" (2026-01-12), README, and a CSS fix (`7d534fd`, authored 2024-06-02). None were released. |
| Releases | 218 tags; 205 have release `main.js` |
| Downloads | 2,688,913 total; 1,425,709 on 2.0.51 |
| manifest | id `obsidian-kanban`, version 2.0.51, minAppVersion 1.0.0, `isDesktopOnly: false` |

**[F] Maintenance status**
- `MAINTAINERS.md` (2026-01-12): "I no longer have the bandwidth to maintain the Kanban plugin (and haven't for a while) and I'm looking for new maintainers".
- Registry change: commit `375e6140`, 2026-03-07, by Matthew Meyers, "Migrate mgmeyers plugins to obsidian-community (#10818)".
- Author label "Obsidian Community Archive" since mirror commit `252a322f` (2026-08-13).

**[F] Dependencies**
- `dependencies`:
  - `@tanstack/match-sorter-utils ^8.15.1`, `@tanstack/react-table ^8.11.2`
  - `animated-scroll-to ^2.2.0`, `box-intersect ^1.0.2`, `choices.js 9.0.1`, `classcat ^5.0.3`, `colord ^2.9.3`
  - `deepmerge ^4.2.2`, `eventemitter3 ^5.0.1`, `fast-json-patch ^3.1.0`, `file-selector ^0.2.4`
  - `immutability-helper ^3.1.1`, `immutable-json-patch ^1.1.2`, `is-plain-object ^5.0.0`, `mark.js ^8.11.1`
  - `mdast ^3.0.0`, `mdast-util-from-markdown ^1.0.2`, `mdast-util-frontmatter ^1.0.0`, `mdast-util-to-markdown ^1.2.6`, `mdast-util-to-string ^3.1.0`
  - `micromark-util-character ^1.1.0`, `micromark-util-types ^1.0.1`, `monkey-around ^2.1.0`, `obsidian-daily-notes-interface ^0.9.4`
  - `preact ^10.8.2`, `react` and `react-dom` → `npm:@preact/compat`, `react-colorful ^5.6.1`, `react-cool-onclickoutside ^1.6.1`, `react-fast-compare ^3.2.2`
  - `rrule ^2.8.1`, `unist-util-visit ^4.1.0`
- `devDependencies`:
  - `esbuild ^0.19.11` (resolved 0.19.12), `esbuild-plugin-less ^1.3.1`, `typescript ^5.3.3`, `obsidian ^1.5.7-1`, `obsidian-dataview ^0.5.66`
  - `@codemirror/{commands,state,view} ^6.0.0`, `eslint ^8.56.0`, `prettier ^3.1.1`, `magic-string ^0.30.10`, `builtin-modules ^3.3.0`
  - `obsidian-plugin-cli ^0.9.0`, `replace ^1.2.2`, `tslib ^2.6.2`, plus `@types` packages
- Resolved and bundled versions: moment 2.29.4, preact 10.20.2, rrule 2.8.1, tslib 2.6.2.

**[F] Build artifacts**
- `main.js` is not committed; it is a release asset.
- `main.js` for 2.0.51: **990,749 bytes** (minified, 151 lines), SHA-256 `a7e3bd4c…c39ac`. `styles.css`: 60,802 bytes, SHA-256 `ecf6dd31…d9a7`.
- The mgmeyers and obsidian-community release URLs serve identical files.
- Toolchain: **esbuild** (`esbuild.config.mjs`, cjs, es2018, minify), LESS, with `obsidian`, `electron`, `@codemirror/*` and Node builtins external.

**[F] Offline rebuild: equivalent, not byte-identical.**
- The lockfile has two git-hosted dependencies:
  - `obsidian@obsidianmd/obsidian-api#master` → codeload tarball `8b2eda0` (blocked by the proxy)
  - `@codemirror/language` from `github.com/lishid/cm-language#2644bfc`, which has a `prepare` build step that failed under Node 22
- Workaround: `resolutions: {obsidian: "1.5.7-1"}` and `yarn install --ignore-scripts`. `node_modules` is 205 MB.
- Result: `styles.css` byte-identical. `main.js` 990,728 bytes vs 990,749.
- After normalising minified identifiers, the only differences are module ordering and one TypeScript `__generator` helper variant from a different tslib copy. No other code differs.

**[F] Grep results.** Source:
- `src/components/Item/helpers.ts:332` and `:533`: `win.require('electron').remote.clipboard`. Reads the OS clipboard when files are pasted into a board (desktop only); `EN()` then uses `require("fs/promises")` and `require("path")` to read the pasted files into the vault.
- `src/helpers/renderMarkdown.ts:104`: `window.open(link.href)`, only when the user clicks an external link.
- `src/lang/helpers.ts:53`: `localStorage.getItem('language')`, Obsidian's UI language.
- `src/Settings.ts:701/749/799/1084`: help links to momentjs.com.
- flatpickr and `buffer-es6.mjs` URLs are comments.
- `esbuild.config.mjs:4` imports `fs` at build time.

Released `main.js`:
- `new Function` at 1:45048 is box-intersect's brute-force code generator (template code, no external input). At 33:5331 it is the `Function("return this")` global shim.
- `require("electron")` at 47:7260 and 47:9733 (clipboard); `window.open` at 40:18898; `localStorage` at 39:108007.
- URLs are only license banners (mark.js, choices.js 9.0.1, Fuse.js 3.4.5, feross/buffer, is-plain-object, match-sorter) and moment deprecation strings.
- **No** `fetch`, XHR, `requestUrl`, WebSocket, `sendBeacon` or analytics. Nothing phones home.

**[F] Date and time features**
- Bundled moment 2.29.4 (`src/helpers/patch.ts` imports `moment`), plus Obsidian's own moment.
- A vendored flatpickr (`src/components/Editor/flatpickr/`) and rrule.
- Integrations detected in `main.js`: `enabledPlugins.has("dataview" | "obsidian-tasks-plugin" | "templater-obsidian")`, `getPluginById("daily-notes")`, `plugins.getPlugin("periodic-notes" | "calendar")`.

**[F] Board format:** plain Markdown; see section 2.7.

**[I] Assessment.** Offline-safe with no network calls. Maintenance risk is high: effectively orphaned and in the archive org, and a large 1 MB bundle with about 1,700 transitive packages. Boards stay readable Markdown even if the plugin is dropped, which lowers lock-in. The core **Bases "Kanban" layout** needs Obsidian 1.14, which is early access (help `Bases/Layouts/Kanban view.md`). It is a first-party alternative, but uses a different model: one note per card, grouped by a property.

### 4.3 Beautitab (andrewmcgivery)

**[F] Summary**

| Field | Value |
|---|---|
| Repository | https://github.com/andrewmcgivery/obsidian-beautitab. `andrewbrereton/…` does not exist. |
| License | MIT ("Copyright (c) 2023 andrewmcgivery") |
| Latest release | **1.6.1**, tag commit `8b796a6`, 2024-03-26 17:27 −04:00 (stats `updated` 2024-03-26T21:28Z) |
| Last commit | Same commit and date |
| Releases | 21 tags (including betas), all with release `main.js`; 13 non-beta versions in stats |
| Downloads | 66,110 |
| manifest | id `beautitab`, version 1.6.1, minAppVersion 0.15.0, `isDesktopOnly: false`, even though it uses `fs` and `electron`. This violates the submission requirement, though the desktop-only code paths are guarded. |

**[F] XDA's "hasn't been updated in about 2 years" is confirmed:** the last release and commit were 2024-03-26, about 2.5 years before 2026-09-22.

**[F] Dependencies**
- `dependencies`: `electron ^25.8.1` (unnecessary at runtime, since it is external), `esbuild-copy-static-files ^0.1.0`, `esbuild-sass-plugin ^2.16.0`, `react ^18.2.0`, `react-dom ^18.2.0`.
- `devDependencies`: `esbuild ^0.19.8`, `typescript 4.7.4`, `tslib 2.4.0`, `obsidian latest` (lock: 1.4.11), `builtin-modules 3.3.0`, `@typescript-eslint/* 5.29.0`, `@types/node ^16.11.6`, `@types/react ^18.2.42`, `@types/react-dom ^18.2.17`.

**[F] Build artifacts**
- `main.js` is not committed (built to `dist/`); it is a release asset.
- `main.js` 1.6.1: **276,842 bytes**, SHA-256 `b7e8f4d19df1f4e387eeecc682e3a19a192b967dff3c0cba2dadb6568729cf7f`. `styles.css`: 9,585 bytes.
- Toolchain: **esbuild** plus Sass.

**[F] Offline rebuild: byte-identical** for both `main.js` and `styles.css`. It needed a one-line patch: Node 22 removed `import … assert {type:"json"}`, so I changed `assert` to `with` in `esbuild.config.mjs:7`. `npm ci` also **downloads a 239 MB Electron 25 binary** in postinstall, because `electron` is listed in `dependencies`. Offline, set `ELECTRON_SKIP_BINARY_DOWNLOAD=1` or use `--ignore-scripts`. `node_modules` is 368 MB.

**[F] Grep results.** Source (and the matching lines in released `main.js`):
- `main.ts:84-93` → `main.js:7137-7143`: `requestUrl("https://raw.githubusercontent.com/andrewmcgivery/obsidian-beautitab/main/package.json")` and `…/beta/package.json`. This is a **version check on every plugin load** (`onload` → `versionCheck()`), and it is **not disclosed in the README**.
- `React/Utils/getBackground.ts:158,171` → `main.js:6384,6393`: `https://source.unsplash.com/random?<theme>&cachetag=<date>`. The default background theme is "seasons and holidays", which remote-loads an image (`Settings.ts:53`).
- `React/Utils/getQuote.ts:25` → `main.js:6447`: `requestUrl("https://api.quotable.io/random")`. The default `quoteSource` is **Quoteable** (`Settings.ts:69`).
- `src/Settings.ts:1,15,140` → `main.js:6699,6819`: `import fs`, `import electron`, `electron.remote.dialog.showOpenDialog` plus `fs.readFileSync`. The "Add local image" button lets the user pick files, which are stored base64 in `data.json`.
- `main.ts:14-19`: `new EventSource("http://127.0.0.1:8000/esbuild")` runs in **development only**. It is removed in the release build (no `EventSource` in `main.js`).
- README Credits disclose "Images are pulled from http://www.unsplash.com" and "Quotes are pulled from the quoteable API".
- No telemetry or analytics SDKs, no eval, no `child_process`.

**[F] What "search" is.** It is not a search engine. The two search buttons run an Obsidian command (`app.commands.executeCommandById`). The default is `switcher:open`, the core Quick Switcher. Allowed providers are `switcher`, `omnisearch`, `darlal-switcher-plus` and `obsidian-another-quick-switcher` (`src/Settings.ts:19-29`). It is local only.

**[F] Behaviour.** It takes over every empty tab (`layout-change` → replaces leaves of type `empty` with the Beautitab view).

**[I] Offline behaviour**
- Defaults produce a broken or blank background; secondary sources say Unsplash Source was shut down in 2024, so it is likely broken even online.
- No quote appears; `requestUrl` rejects without a catch.
- Two failed GitHub requests happen on every start.
- All of these appear as egress attempts in DNS and proxy logs.
- It can be configured with Local, Transparent, or "My quotes" backgrounds and quotes, but the version check cannot be turned off without patching the code.

**[I] Assessment.** Cosmetic value only. It makes network calls by default, has an undisclosed update check, and is unmaintained. **Reject**, or accept only an internally patched build that removes `versionCheck` and the remote sources.

### 4.4 Encore theme (carbonateb)

**[F] Summary**

| Field | Value |
|---|---|
| Repository | https://github.com/carbonateb/obsidian-encore-theme. The only registry match. |
| License | MIT ("Copyright (c) 2022 Lucas Champagne") |
| Version | `manifest.json` version **2.11.0**, minAppVersion 1.1.9, author Carbonateb |
| Releases | **0 tags or releases.** Obsidian installs through the legacy path from `raw.githubusercontent.com/carbonateb/obsidian-encore-theme/HEAD/{manifest.json, theme.css}`. The raw file I fetched is identical to the repo's `theme.css`. |
| Last commit | 2024-06-09 (`5365650`, "Support Obsidian version 1.6"); 118 commits |

**[F] Content**
- **CSS only.** `theme.css` is **90,414 bytes**, SHA-256 `65e4236200b4703c2dab136b2220558435af438de733e15adb721e32f76bbb3a`. `obsidian.css` is an identical legacy copy.
- No JavaScript.
- Build: `sass` (`^1.53.0`; pnpm lock 1.69.3) plus `BuildScript.js`, which prepends `source/style-settings.yaml`.

**[F] Offline rebuild: byte-identical** with sass 1.53.0.

**[F] Grep results**
- **No** `@import`, **no** `@font-face`, **no** `font-family` declarations.
- README: "Font: Rubik (not included)". [I] It falls back to the fonts set in Obsidian.
- The only remote reference is `url("https://images.unsplash.com/photo-1707494966495-…")`:
  - as the Style-Settings default at `theme.css:151`
  - as `--encore-translucency-image` at `:2100`
  - applied only under `body.encore-translucency.encore-bg-image .app-container` at `:814`, meaning only when the user picks "Custom Image" translucency through the **Style Settings** plugin. It is off by default.
- Every other `url()` is a `data:` URI: lines 405-408, 828, 978, 982-983, 1060-1061, 1134-1143, 2162-2166, 2337-2338.

**[I] Assessment.** Offline-safe with a very small attack surface (CSS). Pin the commit and hash. Style Settings, another community plugin now also under community-archive, is optional; without it the defaults apply.

---

## 5. Whitelisting checklist

| Item | Source of truth | Offline-safe? | Network calls | Maintenance risk | License | Recommended action |
|---|---|---|---|---|---|---|
| Obsidian desktop 1.13.7, Windows | `github.com/obsidianmd/obsidian-releases/releases/tag/v1.13.7` → `Obsidian-1.13.7.exe`, SHA-256 `f233dc24…a9bc`, Authenticode "Dynalist Inc"; winget `Obsidian.Obsidian` | [F] Yes: designed offline-first; network failures are swallowed | [F] Update check and install-ID ping hourly (on by default); deprecation check if plugins exist; plugin/theme browser; release notes; Web viewer; Canvas web cards; remote embeds; Sync and Publish if used. [I] Spellcheck dictionaries. | Low for the vendor (active releases). Medium in operation: closed source, bundled Chromium 150 needs installer refreshes. | Proprietary freeware; free for work since 2025-02-20; commercial license optional | **Approve.** Install with `/allusers`. Pre-seed `%APPDATA%\obsidian\obsidian.json` with `{"updateDisabled": true}` or tell users to switch off "Automatic updates". Standard `.obsidian/core-plugins.json` with Sync, Publish and Web viewer off. Keep the firewall deny-all. Replace installers through the approval process on each Electron bump. Use AppLocker/WDAC publisher rules on "Dynalist Inc". Legal to review the terms. |
| Obsidian in-app update payload `obsidian-<ver>.asar.gz` | `desktop-releases.json` (hash and signature) | [F] Verifiable offline with openssl | — | — | Proprietary | **Do not use as the primary path.** Prefer installers. If used, verify the signature and treat `%APPDATA%\obsidian\*.asar` as code. |
| Obsidian Linux (AppImage / .deb / tar.gz) | Same GitHub release | Yes | Same as Windows | Same as Windows | Proprietary | **Not needed** in agent containers. Do not install. |
| Obsidian CLI (bundled) | help `Obsidian CLI.md` | Yes (local socket) | None itself | — | Proprietary | **Keep disabled** (off by default). It needs the GUI and has `eval`. |
| Obsidian Headless (npm `obsidian-headless` 0.0.14) | npm registry; help `Obsidian Headless.md` | **No**: Sync and Publish only | Obsidian services | Beta | UNLICENSED | **Reject / not applicable.** |
| Undocumented `policy.json` / `isWork` | `obsidian.asar/main.js` (1.13.7) | Yes | — | Unknown (unreleased) | — | **Monitor.** Ask Obsidian about an enterprise build; do not rely on it yet. |
| Calendar 1.5.10 | `liamcain/obsidian-calendar-plugin` tag 1.5.10; release `main.js` SHA-256 `7fb339e9…d125`; rebuild byte-identical | [F] Yes | [F] None | **High:** no release since 2021-04; relies on internal APIs | MIT | **Approve with conditions.** Carry `manifest.json` and `main.js` (or build internally). Needs the Daily notes core plugin enabled. Test on 1.13.7. Re-assess on each Obsidian upgrade. |
| Kanban 2.0.51 | `mgmeyers/obsidian-kanban`, now `community-archive/obsidian-kanban`, tag 2.0.51; `main.js` SHA-256 `a7e3bd4c…39ac`, `styles.css` `ecf6dd31…d9a7`; rebuild equivalent | [F] Yes | [F] None (clipboard and `fs` only for user paste; `window.open` on click) | **High:** maintainer stepped back in 2026-01; archive org; last release 2024-05; 1 MB bundle | **GPL-3.0** (package.json says MIT) | **Approve with conditions** if the business needs it. Boards are plain Markdown, so there is no lock-in. Plan migration to the core Bases Kanban layout once Obsidian 1.14 is stable. |
| Beautitab 1.6.1 | `andrewmcgivery/obsidian-beautitab` tag 1.6.1; `main.js` SHA-256 `b7e8f4d1…727f` | [F] **No** by default | [F] `raw.githubusercontent.com` version check on every load (undisclosed); `source.unsplash.com` backgrounds (default); `api.quotable.io` quotes (default) | **High:** no updates since 2024-03; remote services dead or unstable | MIT | **Reject**, or approve only an internally patched build with the network code removed and local backgrounds and custom quotes. |
| Encore theme 2.11.0 | `carbonateb/obsidian-encore-theme` @ `5365650`; `theme.css` SHA-256 `65e42362…bb3a`; rebuild from SCSS byte-identical | [F] Yes (CSS only; no fonts or imports) | [F] None by default. One opt-in Unsplash image only through the Style Settings "Custom Image" option. | Medium: last commit 2024-06; no releases | MIT | **Approve.** Copy to `.obsidian/themes/Encore/{theme.css, manifest.json}`. Optionally strip the Unsplash default URL. |
| Style Settings plugin (optional, for Encore) | `obsidian-community/obsidian-style-settings` (redirects to community-archive) | Not audited | Not audited | Archive org | Not checked | Only if customisation is required; audit separately. |
| obsidian-help docs (as an offline help vault) | `obsidianmd/obsidian-help` @ `bc5b4f2` | Yes (Markdown vault; some pages embed remote video) | None when opened in Restricted mode | Active | **No LICENSE file** | Legal check before mirroring internally. |
| JSON Canvas spec | `obsidianmd/jsoncanvas` `spec/1.0.md` | Yes | — | Stable (1.0, 2024-03-11) | MIT | **Approve** as reference for agent tooling. |
| Vaults processed by LLM agents | help `How Obsidian stores data.md`; Kanban and Canvas specs above | Yes: plain `.md`, `.canvas` (JSON), `.base` (YAML) | — | — | — | **Approve**, with controls: agents must not write to `.obsidian/`; review diffs to plugin and snippet folders; keep Restricted mode on for agent-touched vaults. |
| Plugin build toolchain, if rebuilding from source | Node 22 plus vendored `node_modules` (223 / 205 / 368 MB) | Yes once vendored | Only at vendoring time; git-hosted dependencies need pre-fetching | — | Various OSS | **Optional.** Rebuild internally to remove trust in GitHub release assets. Mirror npm tarballs (for example with Verdaccio). Pin `obsidian` typings to npm versions. Set `ELECTRON_SKIP_BINARY_DOWNLOAD=1`. |

---

## 6. Reproduction pointers

- Workspace: `/tmp/obsidian-audit/`
  - `app/`: Obsidian 1.13.7 binaries and extracted ASARs (`x-app/`, `x-obsidian/`)
  - `releases/`: plugin release assets
  - `build/`: rebuild worktrees and logs
  - `cve/`: CVE JSON
  - `winget-pkgs/manifests/o/Obsidian/Obsidian/1.13.7/`
  - `releases-history/`: blobless history of obsidian-releases
- Verify an update payload offline:
  1. Extract `SIGNATURE_CERT` from `resources/app.asar/main.js` into `cert.pem`.
  2. `openssl x509 -in cert.pem -pubkey -noout > pub.pem`
  3. base64-decode `signature` from `desktop-releases.json` into `sig.bin`.
  4. `openssl dgst -sha256 -verify pub.pem -signature sig.bin obsidian-1.13.7.asar.gz` → `Verified OK`.
- Sources I could not reach, to re-check from the approval network: obsidian.md/privacy, /terms, /license, /security, /blog/free-for-work, /changelog; NVD search "Obsidian"; plugin listing scorecards on community.obsidian.md.
