#!/usr/bin/env python3
"""Relax FlashQLA's build-time GPU-presence guard for GPU-less build hosts.

Upstream `build.sh` builds the FlashQLA SM70/SM75 Gated-DeltaNet extension by
importing `flash_qla.ops.gated_delta_rule.legacy.sm_legacy` and calling
`_load_ext()`. That loader starts with:

    if not torch.cuda.is_available():
        raise RuntimeError("SM70/SM75 legacy GDN backend requires CUDA")

The guard is about *serving*, not about compiling: immediately after it the
loader pins the architecture from `TORCH_CUDA_ARCH_LIST` and hands
`gdn_forward.cu` to nvcc. It never queries a device. Build hosts and CI runners
have no GPU, so without this relaxation `build.sh` cannot finish there at all.

This script edits the repository's own SM75 patch file
(`tools/flashqla_sm75_patches/sm_legacy.py`, which `build.sh` copies over the
cloned FlashQLA tree) so the guard also accepts an explicit opt-in:

    FLASHQLA_ALLOW_GPU_LESS_BUILD=1

The variable is only set for the build. At serving time it is unset, so the
shipped behaviour is byte-for-byte upstream's.

Usage:
    relax-flashqla-build-guard.py <path to tools/flashqla_sm75_patches/sm_legacy.py>
"""

from __future__ import annotations

import sys
from pathlib import Path

NEEDLE = (
    "    if not torch.cuda.is_available():\n"
    '        raise RuntimeError("SM70/SM75 legacy GDN backend requires CUDA")\n'
)

REPLACEMENT = (
    "    if (\n"
    "        not torch.cuda.is_available()\n"
    '        and os.environ.get("FLASHQLA_ALLOW_GPU_LESS_BUILD") != "1"\n'
    "    ):\n"
    '        raise RuntimeError("SM70/SM75 legacy GDN backend requires CUDA")\n'
)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <sm_legacy.py path>")

    path = Path(sys.argv[1])
    if not path.is_file():
        raise SystemExit(f"missing FlashQLA patch file: {path}")

    source = path.read_text(encoding="utf-8")
    if "FLASHQLA_ALLOW_GPU_LESS_BUILD" in source:
        print(f"{path}: guard already relaxed")
        return 0

    occurrences = source.count(NEEDLE)
    if occurrences != 1:
        raise SystemExit(
            f"{path}: expected exactly one GPU guard, found {occurrences}. "
            "Upstream changed the FlashQLA loader; re-check the patch."
        )

    path.write_text(source.replace(NEEDLE, REPLACEMENT, 1), encoding="utf-8")
    print(f"{path}: relaxed the build-time GPU guard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
