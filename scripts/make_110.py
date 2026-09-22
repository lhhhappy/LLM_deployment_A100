#!/usr/bin/env python3
"""Generate patches/110-sm80-dsa-indexer.patch against build/base_exact (= the L3 code, F54).

Why: on sm80 (A100) the stock base cannot start (F56). Two independent causes on the GLM DSA path:
  1. DeepGEMM MQA-logits kernels (fp8_mqa_logits, fp8_paged_mqa_logits, get_paged_mqa_logits_metadata)
     are Hopper+ only  ->  new module dsa/sm80_deep_gemm.py (torch implementations, installed as lazy
     dispatchers on the deep_gemm module, so every `deep_gemm.<fn>` call site is covered).
  2. Triton cannot convert fp32 -> fp8e4nv on sm80; the index-K cache writers (kpool_fp8_index.py) and
     the indexer query quantizer act_quant (triton_kernel.py) store fp32 values into fp8 buffers  ->  on devices below sm89 the buffer is passed as
     uint8 and values are encoded to e4m3fn bits in software (RNE; values are pre-clamped to +-448).
Every edit is a line-anchored regex with a declared hit count; the script fails if any count differs,
and post-checks that no direct fp8 store remains. Plan: plans/active/110-sm80-dsa.md.
"""
import difflib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "build/base_exact/sglang"
SOFT = ROOT / "build/p110/ax_soft_fp8.py"
SHIM = ROOT / "build/p110/sm80_deep_gemm.py"

WRAP = (r"\1# [ax] 110: sm80 has no DeepGEMM MQA-logits kernels; see dsa/sm80_deep_gemm.py\n"
        r"\1from sglang.srt.layers.attention.dsa.sm80_deep_gemm import maybe_wrap as _ax_wrap\n\n"
        r"\1deep_gemm = _ax_wrap(deep_gemm)\n")

SOFT_IMPORT = ("from sglang.kernels.ops.attention.dsa.ax_soft_fp8 import _ax_fp8_view, _ax_store_fp8  "
               "# [ax] 110: sm80 soft fp8\n")

# (file, regex with MULTILINE, replacement, expected hits)
EDITS = [
    ("srt/layers/attention/dsa_backend.py",
     r"^(    )import deep_gemm\n(?=\nif TYPE_CHECKING)", r"\1import deep_gemm\n\n" + WRAP, 1),
    ("srt/layers/attention/dsa/dsa_indexer.py",
     r"^(    )try:\n        import deep_gemm\n    except ImportError as e:\n        deep_gemm = e\n",
     r"\1try:\n        import deep_gemm\n    except ImportError as e:\n        deep_gemm = e\n" + WRAP, 1),
    ("srt/layers/attention/dsa/dsa_indexer_kpool.py",
     r"^(    )try:\n        import deep_gemm\n    except ImportError as e:\n        deep_gemm = e\n",
     r"\1try:\n        import deep_gemm\n    except ImportError as e:\n        deep_gemm = e\n" + WRAP, 1),
    # kpool_fp8_index.py: helpers after the imports
    ("srt/layers/attention/dsa/kpool_fp8_index.py",
     r"^import triton\.language as tl\n", "import triton.language as tl\n\n" + SOFT_IMPORT, 1),
    # the four launchers view the cache buffer as fp8
    ("srt/layers/attention/dsa/kpool_fp8_index.py",
     r"^(\s+)buf_fp8 = buf\.view\(torch\.float8_e4m3fn\)$", r"\1buf_fp8 = _ax_fp8_view(buf)  # [ax] 110", 4),
    # the RETURN_COMPRESSED output tensor passed to _kpool_softmax_rotate_write_cache_kernel
    ("srt/layers/attention/dsa/kpool_fp8_index.py",
     r"^(\s+)compressed_k,\n(\s+)compressed_scale,\n(\s+)slot_k\.stride\(0\),$",
     r"\1_ax_fp8_view(compressed_k),  # [ax] 110\n\2compressed_scale,\n\3slot_k.stride(0),", 1),
    # the five fp8 stores of quantized values (lines 961, 965-969, 1126, 1234, 1614 of the base)
    ("srt/layers/attention/dsa/kpool_fp8_index.py",
     r"^(\s+)tl\.store\(buf_fp8_ptr \+ out_k_offsets, quantized, mask=(\w+)\)$",
     r"\1_ax_store_fp8(buf_fp8_ptr + out_k_offsets, quantized, \2)  # [ax] 110", 4),
    ("srt/layers/attention/dsa/kpool_fp8_index.py",
     r"^(\s+)tl\.store\(\n\s+compressed_k_ptr \+ row \* HEAD_DIM \+ offs,\n\s+quantized,\n\s+mask=offs < HEAD_DIM,\n\s+\)$",
     r"\1_ax_store_fp8(compressed_k_ptr + row * HEAD_DIM + offs, quantized, offs < HEAD_DIM)  # [ax] 110", 1),
    # act_quant (indexer query -> fp8), kernels/ops/attention/dsa/triton_kernel.py
    ("kernels/ops/attention/dsa/triton_kernel.py",
     r"^import triton\.language as tl\n", "import triton.language as tl\n\n" + SOFT_IMPORT, 1),
    ("kernels/ops/attention/dsa/triton_kernel.py",
     r"^(\s+)tl\.store\(y_ptrs, y, mask=mask\)$", r"\1_ax_store_fp8(y_ptrs, y, mask)  # [ax] 110", 1),
    ("kernels/ops/attention/dsa/triton_kernel.py",
     r"^(\s+)x_flat,\n(\s+)y_flat,\n(\s+)s_flat,$", r"\1x_flat,\n\2_ax_fp8_view(y_flat),  # [ax] 110\n\3s_flat,", 1),
]


def main():
    files = {}
    for rel, pat, rep, want in EDITS:
        text = files.get(rel, (BASE / rel).read_text())
        new, n = re.subn(pat, rep, text, flags=re.M)
        assert n == want, f"{rel}: pattern {pat[:60]!r} hit {n}, expected {want}"
        files[rel] = new
    kp = files["srt/layers/attention/dsa/kpool_fp8_index.py"]
    leftover = re.findall(r"^\s+tl\.store\(\s*(?:buf_fp8_ptr|compressed_k_ptr)", kp, flags=re.M)
    leftover += re.findall(r"^\s+tl\.store\(y_ptrs, y,", files["kernels/ops/attention/dsa/triton_kernel.py"], flags=re.M)
    assert not leftover, f"direct fp8 stores remain: {leftover}"
    out = []
    for rel, new in files.items():
        p = "python/sglang/" + rel
        out += difflib.unified_diff((BASE / rel).read_text().splitlines(True), new.splitlines(True), "a/" + p, "b/" + p)
    for src, p in ((SHIM, "python/sglang/srt/layers/attention/dsa/sm80_deep_gemm.py"),
                   (SOFT, "python/sglang/kernels/ops/attention/dsa/ax_soft_fp8.py")):
        out += difflib.unified_diff([], src.read_text().splitlines(True), "/dev/null", "b/" + p)
    (ROOT / "patches/110-sm80-dsa-indexer.patch").write_text("".join(out))
    print(f"wrote patches/110-sm80-dsa-indexer.patch ({len(files)} files edited + 2 new)")


if __name__ == "__main__":
    main()
