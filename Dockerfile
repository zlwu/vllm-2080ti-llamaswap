# syntax=docker/dockerfile:1.7
#
# vLLM 2080 Ti Definitive Edition runtime added to the official llama-swap image.
#
# Two stages:
#
#   1. `runtime-builder` compiles the upstream SM75 runtime from source on an
#      Ubuntu 24.04 + CUDA 13.0.3 devel base with GCC 15. Ubuntu 24.04 (glibc
#      2.39) is mandatory, not cosmetic: the final image inherits the official
#      llama-swap base (also Ubuntu 24.04), and a runtime built against a newer
#      glibc cannot be loaded there.
#   2. `runtime` copies that tree into the official llama-swap image. The
#      image's own `/usr/local/cuda` (12.8) and `LD_LIBRARY_PATH` are left
#      untouched so llama.cpp keeps working; CUDA 13.0 is added alongside it.
#
# Build:
#   docker build -t vllm-2080ti-llamaswap:local .
#
# Validate only the builder prerequisites (minutes instead of hours):
#   docker build --target toolchain -t vllm-2080ti-llamaswap:toolchain .

ARG CUDA_DEVEL_BASE=nvidia/cuda:13.0.3-devel-ubuntu24.04
ARG LLAMA_SWAP_BASE=ghcr.io/mostlygeek/llama-swap:cuda


# --------------------------------------------------------------------------
# Stage 1: compile the SM75 vLLM runtime (source build, hours on 4 cores)
# --------------------------------------------------------------------------
FROM ${CUDA_DEVEL_BASE} AS runtime-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG VLLM_REF=v0.2.1
ARG REPO_URL=https://github.com/weicj/vLLM-2080Ti-Definitive.git
# Roughly (RAM_GiB - 3) / 3, capped by the CPU count. Raise it only on a machine
# with enough RAM: nvcc front-end jobs are memory hungry.
ARG MAX_JOBS=4
# build.sh only pre-installs torch when its network preflight selects a domestic
# PyPI mirror. On the official route it relies on `uv pip install
# --torch-backend`, but build isolation is disabled, so the build backend's own
# torch requirement goes unmet and the build dies with "No module named
# 'torch'". Pin the PyTorch index so torch is installed first on any route.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG RUSTUP_DIST_SERVER=
ARG RUSTUP_UPDATE_ROOT=
ARG RUST_TOOLCHAIN=1.95

ENV RUNTIME_TREE=/opt/vllm-2080ti \
    TORCH_CUDA_ARCH_LIST=7.5 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/bin:/usr/local/sbin:/usr/local/cuda/bin:/usr/sbin:/usr/bin:/sbin:/bin

# GCC 15 is what the 0.2.x line validates; Ubuntu 24.04 only ships gcc-13/14.
# Symlinks in /usr/local/bin win over /usr/bin on PATH, so `gcc -dumpversion`
# (used by the upstream build script) and nvcc's default host compiler both
# resolve to 15.
#
# `apt_install` retries the update+install pair together. The Ubuntu archive
# rotates packages while a CI job runs, so a freshly fetched index can already
# point at a pool file that 404s seconds later; retrying the update alone is
# not enough, and the whole build would fail on a hiccup that clears in seconds.
RUN set -eux; \
    apt_install() { \
      for i in 1 2 3 4 5; do \
        if apt-get update -o Acquire::Retries=5; then \
          if apt-get install -y --no-install-recommends -o Acquire::Retries=5 "$@"; then return 0; fi; \
        fi; \
        echo "apt attempt $i failed; retrying in 20s"; sleep 20; \
      done; return 1; }; \
    apt_install software-properties-common gnupg ca-certificates curl git make pkg-config perl; \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test; \
    apt_install gcc-15 g++-15 \
      python3.12 python3.12-venv python3.12-dev \
      ninja-build libnuma-dev; \
    ln -sf /usr/bin/gcc-15 /usr/local/bin/gcc; \
    ln -sf /usr/bin/g++-15 /usr/local/bin/g++; \
    ln -sf /usr/bin/gcc-15 /usr/local/bin/cc; \
    ln -sf /usr/bin/g++-15 /usr/local/bin/c++; \
    rm -rf /var/lib/apt/lists/*; \
    gcc -dumpversion; \
    python3.12 --version

# The upstream runtime builds Rust artifacts (a `vllm-rs` binary and a PyO3
# parser module) through setuptools-rust, so cargo must exist before build.sh.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain none --no-modify-path \
 && rustup toolchain install "${RUST_TOOLCHAIN}" --profile minimal \
 && rustup default "${RUST_TOOLCHAIN}" \
 && cargo --version \
 && rustc --version

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv \
 && install -m 0755 /root/.local/bin/uvx /usr/local/bin/uvx \
 && uv --version


# --------------------------------------------------------------------------
# Stage 1a: toolchain only, so prerequisite changes fail fast
# --------------------------------------------------------------------------
FROM runtime-builder AS toolchain

RUN gcc -dumpversion && g++ -dumpversion && cargo --version && uv --version


# --------------------------------------------------------------------------
# Stage 1b: source build
# --------------------------------------------------------------------------
FROM toolchain AS runtime-builder-src

# A shallow clone made from a release tag still carries that tag, so
# setuptools_scm can derive the version. `describe --exact-match` fails the
# build early (instead of deep inside setup.py) if that assumption breaks.
RUN git clone --depth 1 --branch "${VLLM_REF}" "${REPO_URL}" "${RUNTIME_TREE}" \
 && git -C "${RUNTIME_TREE}" describe --tags --exact-match

# Upstream's FlashQLA loader refuses to build unless torch.cuda.is_available(),
# even though the extension is compiled by nvcc from TORCH_CUDA_ARCH_LIST and
# never touches a device. Build hosts and CI runners have no GPU, so relax
# exactly that guard behind an explicit opt-in that is only set for the build.
COPY relax-flashqla-build-guard.py /tmp/relax-flashqla-build-guard.py
RUN python3 /tmp/relax-flashqla-build-guard.py \
      "${RUNTIME_TREE}/tools/flashqla_sm75_patches/sm_legacy.py"

# Keep the two host gates that actually shape the artifacts, since
# ALLOW_HOST_MISMATCH below relaxes build.sh's own checks wholesale.
RUN set -eux; \
    nvcc --version | grep -q 'release 13\.'; \
    test "$(gcc -dumpversion | cut -d. -f1)" = "15"; \
    echo "toolchain preflight ok: $(nvcc --version | tail -1)"; \
    echo "python: $(python3.12 --version)"; \
    echo "build kernel: $(uname -r)"

# The long step. build.sh runs its own host checks, picks PyPI/Git mirrors,
# creates the venv, compiles vLLM, patches torch inductor for E8M0, fetches and
# builds the FlashQLA SM70/SM75 extension, and finally validates the runtime.
#
# ALLOW_HOST_MISMATCH waives build.sh's `kernel >= 7` requirement. That check
# describes the deployment host (the 0.2.x line is validated on Ubuntu 26.04 /
# kernel 7, which the target GPU node runs); this builder is Ubuntu 24.04, the
# same platform upstream's own Dockerfile uses as its final base, and the host
# kernel cannot influence the compiled artifacts. nvcc and GCC are asserted
# explicitly just above instead.
RUN cd "${RUNTIME_TREE}" \
 && ASSUME_YES=1 NON_INTERACTIVE=1 MAX_JOBS="${MAX_JOBS}" \
    BUILD_TORCH_INDEX="${TORCH_INDEX}" \
    FLASHQLA_ALLOW_GPU_LESS_BUILD=1 ALLOW_HOST_MISMATCH=1 ./build.sh

RUN "${RUNTIME_TREE}/.venv/bin/python" -c \
      'import torch, vllm; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'


# --------------------------------------------------------------------------
# Stage 2: official llama-swap image + the SM75 vLLM runtime
# --------------------------------------------------------------------------
FROM ${LLAMA_SWAP_BASE} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG RUNTIME_TREE=/opt/vllm-2080ti

ENV RUNTIME_TREE=${RUNTIME_TREE} \
    CUDA_HOME=/usr/local/cuda-13.0 \
    CUDA_PATH=/usr/local/cuda-13.0 \
    TORCH_CUDA_ARCH_LIST=7.5 \
    FLASHINFER_ENABLE_AOT=1 \
    VLLM_DISABLE_TILELANG=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONSAFEPATH=1

# Same apt retry rationale as the builder stage above.
#
# A compiler is needed at *runtime* too: FlashInfer compiles SM75 kernels and
# torch.compile generates C++ wrappers on first use (then cached under
# FLASHINFER_WORKSPACE_BASE / TORCHINDUCTOR_CACHE_DIR). Triton additionally
# compiles a tiny `cuda_utils.c` on import, so the Python headers must be
# present as well.
RUN set -eux; \
    apt_install() { \
      for i in 1 2 3 4 5; do \
        if apt-get update -o Acquire::Retries=5; then \
          if apt-get install -y --no-install-recommends -o Acquire::Retries=5 "$@"; then return 0; fi; \
        fi; \
        echo "apt attempt $i failed; retrying in 20s"; sleep 20; \
      done; return 1; }; \
    apt_install software-properties-common gnupg ca-certificates curl; \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test; \
    apt_install gcc-15 g++-15 libnuma1 libgomp1 python3.12 python3.12-dev; \
    ln -sf /usr/bin/gcc-15 /usr/local/bin/gcc; \
    ln -sf /usr/bin/g++-15 /usr/local/bin/g++; \
    ln -sf /usr/bin/gcc-15 /usr/local/bin/cc; \
    ln -sf /usr/bin/g++-15 /usr/local/bin/c++; \
    rm -rf /var/lib/apt/lists/*

# CUDA 13.0 toolkit for run-time JIT, placed next to (not over) the image's
# CUDA 12.8 install. The upstream launcher resolves CUDA_HOME by the CUDA
# version torch reports, so /usr/local/cuda-13.0 is what it picks up.
COPY --from=runtime-builder-src /usr/local/cuda-13.0 /usr/local/cuda-13.0
COPY --from=runtime-builder-src ${RUNTIME_TREE} ${RUNTIME_TREE}

# llama-swap supervises one foreground process per model group; this wrapper
# reuses the upstream launcher's validated argument/env resolution but execs the
# server instead of letting the launcher daemonise it.
COPY serve-foreground.sh ${RUNTIME_TREE}/serve-foreground.sh
RUN chmod 0755 ${RUNTIME_TREE}/serve-foreground.sh

RUN ${RUNTIME_TREE}/.venv/bin/python -c \
      'import torch, vllm; print("runtime ok:", vllm.__version__, torch.__version__, torch.version.cuda)'

# ENTRYPOINT/CMD are inherited from the llama-swap base image on purpose.
