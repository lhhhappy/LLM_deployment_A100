#!/usr/bin/env python3
"""Private Humming 0.1.12 headers for an exponent/scale folding experiment.

Never changes installed headers, weights, scales, tuning or raster order.
The caller MUST restrict folded kernels to finite, nonnegative BF16 scales
at most 255. This research guard is checked once before graph capture.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time


RELATIVE_HEADER = "humming/arith/mainloop_arith.cuh"
EXPECTED_HEADER_SHA = "315d2c6b0bc965dd77485a15b522c96585b11de15e39a69108cdb3b3de02f75b"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fold_header(original):
    assert sha(original.encode()) == EXPECTED_HEADER_SHA, "unexpected Humming source"
    text = original
    needle = "  static constexpr uint32_t kDequantBSBits ="
    assert text.count(needle) == 1
    text = text.replace(needle, """  // The probe verifies every scale is finite and in [0, 255].
  static constexpr bool kAxFoldFp8Exp =
      Ctx::kIsIndexedGemm && !kUseWgmma && !kUseMxmma &&
      std::is_same<ElementA, BFloat16>::value &&
      std::is_same<ElementB, Float8E4M3>::value &&
      std::is_same<ElementBS, BFloat16>::value &&
      kIsGroupWeightScale && !kHasZeroPoint && !kIsF16Accum &&
      kWeightScaleGroupSize == 128 && kExpOffset.x == 120 && kExpOffset.y == 0;

  static constexpr uint32_t kDequantBSBits =""", 1)
    needle = """    if (j == 0) {
      if constexpr (ElementA::kBits == 16 && kIsBlockWeightScale) {"""
    assert text.count(needle) == 1
    text = text.replace(needle, """    if (j == 0) {
      if constexpr (kAxFoldFp8Exp) {
        // S2R reloads all scale registers before each transform_b call.
        // Bias once per scale pair, not once per dequantized weight pair.
        scalar_t2 *scale_pairs = reinterpret_cast<scalar_t2 *>(bs[buffer_id]);
        constexpr uint32_t pairs = kNumBSPerGroup / 2;
        const scalar_t2 exp_factor = prepare_exp_scale_factor<scalar_t2, kExpOffset.x>();
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < pairs; ++i) {
          scale_pairs[i] = __hmul2(scale_pairs[i], exp_factor);
        }
      }
      if constexpr (ElementA::kBits == 16 && kIsBlockWeightScale) {""", 1)
    needle = "if constexpr (ElementA::kBits == 16 && kExpOffset.x) {"
    assert text.count(needle) == 1
    text = text.replace(needle, "if constexpr (ElementA::kBits == 16 && kExpOffset.x && !kAxFoldFp8Exp) {", 1)
    return text


def prepare(source, cache_root):
    source = Path(source).resolve()
    originals = {str(p.relative_to(source)): p.read_bytes() for p in sorted(source.rglob("*.cuh"))}
    assert RELATIVE_HEADER in originals
    folded = dict(originals, **{RELATIVE_HEADER: fold_header(originals[RELATIVE_HEADER].decode()).encode()})
    digests = {name: sha(json.dumps({p: sha(v) for p, v in files.items()}, sort_keys=True).encode())
               for name, files in (("native", originals), ("fold", folded))}
    paths = {}
    for name, files in (("native", originals), ("fold", folded)):
        destination = Path(cache_root).resolve() / digests[name] / name
        for relative, data in files.items():
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                assert output.read_bytes() == data, f"private header changed: {output}"
            else:
                output.write_bytes(data)
        paths[name] = destination
    changed = [p for p in originals if originals[p] != folded[p]]
    assert changed == [RELATIVE_HEADER]
    return dict(paths=paths, tree_sha256=digests, changed_headers=changed,
                mainloop_sha256={"native": sha(originals[RELATIVE_HEADER]), "fold": sha(folded[RELATIVE_HEADER])},
                installed_source=str(source))


@contextlib.contextmanager
def private_compiler(metadata, emit):
    """Two actual binaries in one process, with unchanged parsed config values."""
    from humming.jit.compiler import Compiler
    from humming.jit.runtime import KernelRuntime

    include_descriptor = Compiler.__dict__["humming_include_dir"]
    compile_descriptor = Compiler.__dict__["compile"]
    original_compile = compile_descriptor.__func__
    state = {"variant": "native", "compiles_allowed": True}

    def choose(name):
        assert name in metadata["paths"]
        state["variant"] = name
        # Registered old kernel IDs remain strongly referenced by _id2kernel.
        # Evict only constructor memoization, not the dispatch table cache.
        removed = [k for k in KernelRuntime._instances if k[0] == "HummingKernel"]
        for key in removed:
            del KernelRuntime._instances[key]
        emit(kind="private_variant", variant=name, constructor_entries_evicted=len(removed))

    def compile_stamped(cls, code, *args, **kwargs):
        assert state["compiles_allowed"], "unexpected JIT after both variants were prepared"
        name = state["variant"]
        code = f"// ax-scale-fold-probe {name} {metadata['tree_sha256'][name]}\n" + code
        filename = original_compile(cls, code, *args, **kwargs)
        emit(kind="compile_receipt", variant=name, cubin=filename,
             cubin_sha256=sha(Path(filename).read_bytes()))
        return filename

    Compiler.humming_include_dir = staticmethod(lambda: str(metadata["paths"][state["variant"]]))
    Compiler.compile = classmethod(compile_stamped)
    try:
        yield state, choose
    finally:
        Compiler.humming_include_dir = include_descriptor
        Compiler.compile = compile_descriptor


def cpu_check():
    """Exact dyadic RN-even, including BF16 subnormals; not a GPU FTZ test."""
    def unpack(bits):
        e, m = (bits >> 7) & 255, bits & 127
        return (m if e == 0 else m + 128, -133 if e == 0 else e - 134)

    def rnd(m, e):
        if not m:
            return 0
        h = m.bit_length() - 1 + e
        shift = max(h - 7, -133) - e
        if shift > 0:
            remainder, n = m & ((1 << shift) - 1), m >> shift
            halfway = 1 << (shift - 1)
            n += int(remainder > halfway or (remainder == halfway and n & 1))
        else:
            n = m << -shift
        if h < -126:
            return n
        if n == 256:
            n, h = 128, h + 1
        return 0x7f80 if h > 127 else ((h + 127) << 7) | (n - 128)

    def mul(a, b):
        ma, ea = unpack(a & 0x7fff)
        mb, eb = unpack(b & 0x7fff)
        return ((a ^ b) & 0x8000) | rnd(ma * mb, ea + eb)

    start = time.monotonic()
    factor = (127 + 120) << 7
    weights = []
    for q in range(256):
        if q & 127 == 127:
            continue
        raw = ((q & 128) << 8) | ((q & 127) << 4)
        weights.append((q, raw, mul(raw, factor)))
    pairs = 0
    for scale in range(0x4380):
        folded = mul(scale, factor)
        assert folded < 0x7f80
        for q, raw, weight in weights:
            assert mul(weight, scale) == mul(raw, folded), (q, scale)
            pairs += 1
    assert mul(0x4380, factor) == 0x7f80
    return dict(kind="cpu_exact_dyadic_bf16_check", finite_fp8_patterns=len(weights),
                nonnegative_bf16_scale_patterns=0x4380, tested_pairs=pairs, mismatches=0,
                scale_max=255, first_overflow_scale=256, elapsed_s=time.monotonic() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    assert args.cpu_check
    print(json.dumps(cpu_check()), flush=True)
