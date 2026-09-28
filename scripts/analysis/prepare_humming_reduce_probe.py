#!/usr/bin/env python3
"""Isolated feasibility probe, NEVER a serving patch.

Copy Humming headers into this experiment's cache. Reserved tuning values
select rejected atomic, row-lock or last-writer down-GEMM epilogues.
The existing launcher still receives its full-sized BF16 scratch allocation;
only its first M*H FP32 elements are used. No memory-capacity saving is claimed.
For this prototype ONLY, the otherwise-unused BF16 input-scale pointer carries
router weights. A production implementation would need an explicit ABI.
"""
import hashlib
import json
from pathlib import Path
import shutil


def prepare(root):
    import humming.jit.compiler as compiler
    source=Path(compiler.Compiler.humming_include_dir())
    target=root/'cache/humming_reduce_headers'
    if not target.exists():
        shutil.copytree(source,target)
    writer=target/'humming/epilogue/gmem_writer.cuh'
    scheduler=target/'humming/scheduler.cuh'
    # Always derive from the unmodified package, never apply a patch twice.
    w=(source/'humming/epilogue/gmem_writer.cuh').read_text()
    assert w.count('#pragma once') == 1
    w=w.replace('#pragma once', '#pragma once\n#include <cuda/atomic>', 1)
    s=(source/'humming/scheduler.cuh').read_text()
    needle='''        if (!kUseStreamK || slice_count == 1 || slice_id == 0) {'''
    assert w.count(needle)==1
    w=w.replace(needle,'''        if constexpr (Ctx::kIsIndexedGemm && (Ctx::kRasterGroupM == 117 || Ctx::kRasterGroupM == 118)) {
          static_assert(ProblemShape::N == 4096 && ProblemShape::K == 256);
          static_assert(!kUseStreamK && PadShape::N == 0);
          // Keep the GEMM's BF16 rounding before router weighting.
          const auto *rounded = reinterpret_cast<const __nv_bfloat16 *>(&val);
          const float weight = reinterpret_cast<const float *>(ctx.params.as)[gmem_row] * 2.5f;
          auto *result = reinterpret_cast<float *>(const_cast<void *>(ctx.params.c));
          const uint32_t offset = (gmem_row / 9) * 4096 + col_offset + gmem_col * 8;
          PRAGMA_UNROLL
          for (uint32_t j = 0; j < 8; ++j) {
            atomicAdd(result + offset + j, __bfloat162float(rounded[j]) * weight);
          }
        } else if (!kUseStreamK || slice_count == 1 || slice_id == 0) {''')
    needle='''    constexpr uint32_t total_write_int4s = BlockShape::M * BlockShape::N * 2 / 16 / kNumWriteSplits;'''
    assert w.count(needle)==1
    w=w.replace(needle,'''    if constexpr (Ctx::kIsIndexedGemm && (Ctx::kRasterGroupM == 119 || Ctx::kRasterGroupM == 120)) {
      static_assert(ProblemShape::N == 4096 && ProblemShape::K == 256 && BlockShape::N == 256);
      static_assert(!kUseStreamK && PadShape::N == 0 && kNumWriteSplits == 1);
      const uint32_t lane = threadIdx.x % 32;
      const uint32_t warp = threadIdx.x / 32;
      const uint32_t smem_base = offsetof(SharedStorage, reduce) / 128 % 8;
      auto *locks = reinterpret_cast<int *>(const_cast<void *>(ctx.params.as)) + ctx.params.shape_m;
      auto *error = locks + (ctx.params.shape_m / 9) * 16;
      auto *result = reinterpret_cast<volatile float *>(const_cast<void *>(ctx.params.c));
      // One warp owns a complete 256-column row while holding its lock.
      for (uint32_t row = warp; row < BlockShape::M; row += kNumMathThreads / 32) {
        const uint32_t routed_row = ctx.smem.wr_row_index[row];
        if (routed_row >= output_shape_m) continue;
        const uint32_t lock_idx = (routed_row / 9) * 16 + col_offset / 256;
        bool acquired = false;
        if (lane == 0) {
          for (uint32_t attempt = 0; attempt < 262144; ++attempt) {
            cuda::atomic_ref<int, cuda::thread_scope_device> lock(locks[lock_idx]);
            int expected = 0;
            if (lock.compare_exchange_strong(expected, 1, cuda::memory_order_acquire,
                                             cuda::memory_order_relaxed)) { acquired = true; break; }
            __nanosleep(32);
          }
          if (!acquired) atomicAdd(error, 1);
        }
        acquired = __shfl_sync(0xffffffff, acquired, 0);
        if (!acquired) continue;  // A diagnostic failure, never an unbounded spin.
        __syncwarp();
        const uint32_t sr = (lane / 8) * BlockShape::M + row;
        const uint32_t sc = (lane % 8) ^ ((sr + smem_base) % 8);
        const int4 val = ctx.smem.reduce[sr * 8 + sc];
        const auto *rounded = reinterpret_cast<const __nv_bfloat16 *>(&val);
        const float weight = reinterpret_cast<const float *>(ctx.params.as)[routed_row] * 2.5f;
        const uint32_t offset = (routed_row / 9) * 4096 + col_offset + lane * 8;
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 8; ++j) {
          result[offset + j] = result[offset + j] + __bfloat162float(rounded[j]) * weight;
        }
        __threadfence();
        __syncwarp();
        if (lane == 0) {
          cuda::atomic_ref<int, cuda::thread_scope_device> lock(locks[lock_idx]);
          lock.store(0, cuda::memory_order_release);
        }
      }
      return;
    }
    constexpr uint32_t total_write_int4s = BlockShape::M * BlockShape::N * 2 / 16 / kNumWriteSplits;''')
    needle='''  CUDA_INLINE\n  void write_tma(uint32_t slice_id, uint32_t slice_count) {'''
    assert w.count(needle)==1
    w=w.replace(needle,'''  CUDA_INLINE
  void combine_last_writer() {
    static_assert(ProblemShape::N == 4096 && ProblemShape::K == 256 && BlockShape::N == 256);
    static_assert(!kUseStreamK && PadShape::N == 0 && kNumWriteSplits == 1);
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warp = threadIdx.x / 32;
    auto *counters = reinterpret_cast<int *>(const_cast<void *>(ctx.params.as)) + ctx.params.shape_m;
    // A dedicated error word follows the counters; never alias it with output.
    auto *combined = reinterpret_cast<__nv_bfloat16 *>(counters + (ctx.params.shape_m / 9) * 16 + 1);
    const auto *routed = reinterpret_cast<const volatile __nv_bfloat16 *>(ctx.params.c);
    // Publish every lane's vector store before signalling that this row is ready.
    __threadfence();
    ctx.sync_math_threads();
    for (uint32_t row = warp; row < BlockShape::M; row += kNumMathThreads / 32) {
      const uint32_t routed_row = ctx.smem.wr_row_index[row];
      if (routed_row >= output_shape_m) continue;
      const uint32_t token = routed_row / 9;
      int completed = 0;
      if (lane == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> counter(counters[token * 16 + col_offset / 256]);
        completed = counter.fetch_add(1, cuda::memory_order_acq_rel);
      }
      completed = __shfl_sync(0xffffffff, completed, 0);
      if (completed != 8) continue;
      // Transfer lane 0's acquire ordering before other lanes read prior writers.
      __syncwarp();
      const uint32_t column = col_offset + lane * 8;
      float sums[8] = {};
      PRAGMA_UNROLL
      for (uint32_t expert = 0; expert < 9; ++expert) {
        const float weight = reinterpret_cast<const float *>(ctx.params.as)[token * 9 + expert] * 2.5f;
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 8; ++j) {
          // Volatile reads cannot use a stale private L1 copy.
          const auto *bits = reinterpret_cast<const volatile unsigned short *>(routed);
          __nv_bfloat16 value = __ushort_as_bfloat16(bits[(token * 9 + expert) * 4096 + column + j]);
          sums[j] = __fmaf_rn(__bfloat162float(value), weight, sums[j]);
        }
      }
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < 8; ++j) combined[token * 4096 + column + j] = __float2bfloat16_rn(sums[j]);
    }
  }

  CUDA_INLINE
  void write_tma(uint32_t slice_id, uint32_t slice_count) {''')
    needle='''      write_legacy(slice_id, slice_count, split_idx);'''
    assert w.count(needle)==1
    w=w.replace(needle,needle+'''
      if constexpr (Ctx::kIsIndexedGemm && (Ctx::kRasterGroupM == 121 || Ctx::kRasterGroupM == 122)) combine_last_writer();''')
    needle='''    if constexpr (kRasterGroupM <= 1 || !kIsDenseGemm) {'''
    assert s.count(needle)==1
    s=s.replace(needle,'''    if constexpr (kIsIndexedGemm && (kRasterGroupM == 118 || kRasterGroupM == 120 || kRasterGroupM == 122)) {
      // A 256-column FP32 output slab is 16 MiB at M=16k, below A100 L2.
      m_id = mn_index % m_blocks;
      n_id = mn_index / m_blocks;
    } else if constexpr (kRasterGroupM <= 1 || !kIsDenseGemm) {''')
    writer.write_text(w);scheduler.write_text(s)
    compiler.Compiler.humming_include_dir=staticmethod(lambda:str(target))
    return dict(source=str(source),target=str(target),
                writer_sha256=hashlib.sha256(writer.read_bytes()).hexdigest(),
                scheduler_sha256=hashlib.sha256(scheduler.read_bytes()).hexdigest())


if __name__=='__main__':
    print(json.dumps(prepare(Path(__file__).resolve().parents[2])))
