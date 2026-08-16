# Third-party licensing and scope

The repository-level [`LICENSE`](LICENSE) is The Unlicense
(`SPDX-License-Identifier: Unlicense`). It applies only to original material in
this repository for which the repository author owns the copyright and can make
the public-domain dedication.

It does not relicense third-party material. In particular:

- Qwen3.8 model, tokenizer, processor, configuration, and model-card material
  copied or derived from `Qwen/Qwen3.8-27B` or
  `unsloth/Qwen3.8-27B-NVFP4` retains its upstream license and notices. The
  tracked upstream Qwen material identifies Apache License 2.0; a copy is in
  [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
- The `vllm` Git submodule remains under the licenses and notices in that exact
  upstream source tree. The pinned revision identifies Apache License 2.0.
- Review patches and generated transformations that contain or modify vLLM
  source remain subject to the applicable vLLM license and notices. The
  Unlicense applies only to separable original material where the author owns
  the relevant rights.
- The tracked release-page screenshot is reference evidence and is not offered
  as original Unlicensed artwork.
- Container base images, system packages, Python/Rust/JavaScript dependencies,
  NVIDIA components, and other bundled tools retain their respective upstream
  licenses. Their inclusion or pinning does not change those terms.

The checkpoint weights and local Docker image archives are deliberately not
committed to this Git repository. Their local presence does not place them under
The Unlicense.

Nothing in this notice grants trademark rights or changes any upstream license.
