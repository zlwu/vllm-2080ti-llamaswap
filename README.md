# vllm-2080ti-llamaswap

An image that adds the **vLLM 2080 Ti Definitive Edition** (SM75) runtime to the
official [`llama-swap`](https://github.com/mostlygeek/llama-swap) image, so vLLM
can be served as one of llama-swap's on-demand model groups.

## Security boundary

This public repository and its GHCR image contain only generic code and
placeholders. Do not commit real hostnames, IP addresses, domains, production
stack files, credentials, `.env` files, topology documents, or model weights.
Model weights and runtime caches are mounted at run time.

## What is in the image

| Layer | Content |
|---|---|
| Base | `ghcr.io/mostlygeek/llama-swap:cuda` (Ubuntu 24.04, CUDA 12.8 runtime, llama.cpp + llama-swap) |
| Added | A source-built [vLLM 2080 Ti Definitive Edition](https://github.com/weicj/vLLM-2080Ti-Definitive) runtime at `/opt/vllm-2080ti` |
| Added | `/usr/local/cuda-13.0` for run-time kernel JIT |

`/usr/local/cuda` (12.8) and `LD_LIBRARY_PATH` are deliberately left untouched so
llama.cpp keeps working; the launcher resolves `/usr/local/cuda-13.0` by name
because that is the CUDA version the bundled torch reports.

### Why a source build is unavoidable

* The runtime is a hardware-specific fork: SM75 CUDA/C++ kernels, a FlashQLA
  SM70/SM75 Gated-DeltaNet extension and a Rust front end. There is no wheel on
  PyPI and no prebuilt image upstream, so `pip install vllm` cannot reproduce it.
* The build must run on Ubuntu 24.04 (glibc 2.39) so the result can execute in
  the llama-swap base image. Compiling on a newer glibc host produces a runtime
  that the container cannot load.
* The container also needs a C++ compiler and `nvcc` **at run time**: vLLM's
  `--compilation-config` / torch.compile and FlashInfer's AOT kernels compile on
  first use and cache under `TORCHINDUCTOR_CACHE_DIR` / `FLASHINFER_WORKSPACE_BASE`.
  The official llama-swap image ships neither.

## When the image is rebuilt

The build compiles vLLM from source and takes hours, so it is deliberately
**not** triggered by every push or by a blind weekly rebuild. The weekly job is
a *check* that escalates to a build only when something upstream moved:

| Trigger | Effect |
|---|---|
| New `ghcr.io/mostlygeek/llama-swap:cuda` digest | Builds with the new base, then records the digest in `.github/llama-swap-base` |
| New stable release of `weicj/vLLM-2080Ti-Definitive` | Opens a pull request bumping `.github/vllm-ref`; merging it builds |
| Change to `Dockerfile` / `serve-foreground.sh` | Builds |
| `workflow_dispatch` | Always builds (optionally with a `vllm_ref` override) |
| Pull request | Validation only: `shellcheck` + `docker build --target toolchain` |

Keeping the base image current matters more than usual here: because this image
*replaces* the official llama-swap image for the service, new llama.cpp builds
only arrive through it. A new upstream vLLM release changes runtime behaviour,
so it goes through review first.

## Tags

| Tag | Meaning |
|---|---|
| `latest` | Latest build from `main` |
| `sha-<commit>` | Build for a specific commit |
| `buildcache` | BuildKit cache only; not a runnable image |

## Usage

The image expects to be used as a llama-swap model group. Mount your model
directory and (optionally) a persistent cache directory:

```yaml
services:
  llama-swap:
    image: ghcr.io/zlwu/vllm-2080ti-llamaswap:latest
    command: ["-config", "/app/config.yaml", "-watch-config", "-listen", ":8080"]
    volumes:
      - /path/to/llama-swap/config.yaml:/app/config.yaml:ro
      - /path/to/models:/models
      - /path/to/data:/data
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - CUDA_VISIBLE_DEVICES=0,1
    ports:
      - "8088:8080"
```

Model group in `config.yaml`:

```yaml
healthCheckTimeout: 1800   # a cold vLLM start is far slower than llama.cpp

models:
  my-vllm-model:
    cmd: >
      /opt/vllm-2080ti/serve-foreground.sh
      --model-dir /models/vllm/MyCheckpoint
      --speculative-model /models/vllm/MyDraft
      --profile 2x2080Ti/qwen27b/w4a16/dflash2-fp8kv-1x262K-text-image.env
      --mode fast
      --gpu-devices 0,1
      --tp-size 2
      --pp-size 1
      --served-name my-vllm-model
      --port ${PORT}
    proxy: "http://127.0.0.1:${PORT}"
    useModelName: my-vllm-model
    checkEndpoint: /health
    context: 262144
    unloadTimeout: 180
    env:
      - "HF_ENDPOINT=https://hf-mirror.com"
      - "TORCHINDUCTOR_CACHE_DIR=/data/vllm/torchinductor"
      - "TRITON_CACHE_DIR=/data/vllm/triton"
      - "FLASHINFER_WORKSPACE_BASE=/data/vllm/flashinfer"
      - "VLLM_CACHE_ROOT=/data/vllm/vllm-cache"
```

### Why `serve-foreground.sh` exists

llama-swap supervises exactly one foreground process per model group: it starts
`cmd`, polls `checkEndpoint`, and signals the whole process group on unload.
Upstream `launcher.sh` daemonises (`nohup` + `setsid` + pid file) and exits,
which llama-swap would read as "the model died". `serve-foreground.sh` reuses
`launcher.sh`'s validated configuration resolution (profiles, startup mode,
SM75 runtime environment, speculative decoding, CUDA graph sizing) and then
`exec`s the server itself.

No `cmdStop` is needed: llama-swap starts commands with `Setpgid` and sends
`SIGTERM` to the process group, and after `exec` the group leader *is* vLLM.

### Operational notes

* vLLM takes essentially the whole GPU pair (the reference profile uses
  `GPU_UTIL=0.96`). llama-swap's swap semantics keep it mutually exclusive with
  the llama.cpp model groups; that is the intended behaviour.
* Point `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`,
  `FLASHINFER_WORKSPACE_BASE` and `VLLM_CACHE_ROOT` at a persistent mount, and
  clear them when you change the pinned upstream ref.
* The upstream launcher refuses to start when the host runs
  `vm.overcommit_memory=0` and the largest `safetensors` file exceeds the commit
  headroom, because the container cannot change a host sysctl. Set
  `vm.overcommit_memory=1` on the host.

## Building locally

```bash
docker build -t vllm-2080ti-llamaswap:local \
  --build-arg VLLM_REF="$(tr -d '[:space:]' < .github/vllm-ref)" \
  --build-arg MAX_JOBS=4 .
```

This compiles vLLM from source and takes hours on a 4-core machine. To validate
only the builder prerequisites (apt/PPA, GCC 15, Rust, uv) in a few minutes:

```bash
docker build --target toolchain -t vllm-2080ti-llamaswap:toolchain .
```

## Attribution and license

This image packages and redistributes the work of others:

* [vLLM](https://github.com/vllm-project/vllm) — Apache-2.0.
* [vLLM 2080 Ti Definitive Edition](https://github.com/weicj/vLLM-2080Ti-Definitive)
  by [github.com/weicj](https://github.com/weicj) — the SM75 fork, launcher,
  profiles and validation evidence this image builds.
* [FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75) — SM70/SM75
  Gated-DeltaNet prefill backend.
* [llama-swap](https://github.com/mostlygeek/llama-swap) — the model-swapping
  proxy used as the base image.

Everything here is licensed under Apache-2.0; see [LICENSE](LICENSE). This
repository is an independent packaging project, not affiliated with or endorsed
by the upstream authors. Route parameters, performance figures and support
statements belong to the upstream project — consult its documentation for the
authoritative list of validated profiles.
