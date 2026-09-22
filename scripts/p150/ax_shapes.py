"""Startup-only request warmup and strict cache verification (150).

Finite shape sampling, NOT a proof that arbitrary serving shapes cannot JIT.
No model, kernel policy, user sampling parameters or token accounting is changed.
"""
import asyncio
import logging
import os
import uuid

from sglang.srt.managers.io_struct import GenerateReqInput

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 1800.0
FLUSH_TIMEOUT = 60.0


def shape_plan():
    """Deterministic token IDs; related chain requests share exact prefixes."""
    def tokens(n, seed):
        # Ordinary IDs well inside GLM's vocabulary; distinct first tokens keep
        # unrelated cases cold without relying on cache_salt support.
        return [1000 + (seed * 103 + i * 17) % 30000 for i in range(n)]

    cold = tokens(600, 1)
    chain = tokens(600 + 1000 + 7000 + 12000, 2)
    yield 'cold-600', [cold], [2], False
    yield 'chain-seed', [chain[:600]], [2], False
    for length in (1600, 8600, 20600):
        yield f'chain-{length}', [chain[:length]], [2], True
    yield 'cold-20000', [tokens(20000, 3)], [2], False
    # Small/large grids, page/chunk tails, 113 fallback vs predecode branch.
    for i, length in enumerate((1, 63, 64, 65, 128, 2048, 2049, 4096, 8192, 8193)):
        yield f'edge-{length}', [tokens(length, 10 + i)], [2], False
    for i, marker in enumerate((154827, 154829)):
        role = tokens(2400, 30 + i)
        role[1024] = marker
        yield f'role-{marker}', [role], [2], False
        # Branch strictly before the marker; exercise a saved role checkpoint.
        branch = role[:1024] + [marker] + tokens(1301, 40 + i)
        yield f'role-hit-{marker}', [branch], [2], True
    # 31*257=7967 tokens but 31*ceil(257/64)=155 chunks: NT_BUCKET=2
    # if admitted together at chunk8192. Scheduler may split it; report limits.
    yield 'ragged-31', [tokens(257, 60+i) for i in range(31)], [2]*31, False
    for width in range(6, 33):
        ids = [tokens(64 + (i % 4) * 17, 100 + width * 32 + i)
               for i in range(width)]
        yield f'decode-{width}', ids, [8 + i % 5 for i in range(width)], False


def assert_empty(scheduler):
    """Run on EACH scheduler after the ordinary flush; never mutate pools.

    This verifier intentionally supports the plain FULL+MAMBA non-HiCache
    profile only. Missing/incompatible interfaces fail the verified flush.
    Explicit raises survive python -O.
    """
    def equal(name, actual, expected):
        if actual != expected:
            raise RuntimeError(f'ax_shapes {name}: {actual!r} != {expected!r}')

    equal('idle', scheduler.is_fully_idle(), True)
    req = scheduler.req_to_token_pool
    kv = scheduler.token_to_kv_pool_allocator
    equal('request pool', req.available_size(), req.size)
    equal('KV pool', kv.available_size(), kv.size // kv.page_size * kv.page_size)
    equal('Mamba pool', req.mamba_allocator.available_size(), req.mamba_pool.size)
    if getattr(req, 'mamba_ckpt_pool', None) is not None:
        pool = req.mamba_ckpt_pool
        equal('checkpoint pool', pool.available_size(), pool.num_slots)
    tree = scheduler.tree_cache
    equal('tree tokens/states', tree.total_size(), (0, 0))
    for name in ('full_evictable_size', 'full_protected_size',
                 'mamba_evictable_size', 'mamba_protected_size'):
        equal(name, getattr(tree, name)(), 0)
    logger.info('ax_shapes verified empty request/KV/Mamba pools and radix tree')


def verify_empty(scheduler, flushed):
    """TP peers don't all emit IPC replies: agree on failure before rank0 replies."""
    import torch
    import torch.distributed as dist

    error = None
    try:
        if not flushed:
            raise RuntimeError('ax_shapes scheduler refused flush')
        assert_empty(scheduler)
    except Exception as exc:
        error = str(exc)
        logger.exception('ax_shapes local pool verification failed')
    failed = torch.tensor([int(error is not None)], dtype=torch.int32, device='cpu')
    # Ordinary TP only; all schedulers process the same control request.
    # All ranks participate even if one rank's flush/verification failed.
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=scheduler.tp_cpu_group)
    if failed.item():
        raise RuntimeError(error or 'ax_shapes pool verification failed on another TP rank')


async def run(disaggregation_mode, manager):
    args = manager.server_args
    if getattr(args, 'speculative_algorithm', None):
        logger.warning('ax_shapes skipped: speculative/MTP needs a separate shape plan')
        return
    if disaggregation_mode != 'null':
        raise ValueError('ax_shapes requires an ordinary TP server (no PD)')
    if (getattr(args, 'enable_hierarchical_cache', False)
            or getattr(args, 'enable_dp_attention', False)
            or getattr(args, 'dp_size', 1) != 1
            or getattr(args, 'pp_size', 1) != 1
            or getattr(args, 'tokenizer_worker_num', 1) != 1):
        raise ValueError('ax_shapes requires single-tokenizer plain TP, no HiCache/DP/PP')
    prefix = 'ax-warmup-' + uuid.uuid4().hex
    active = []

    async def flush():
        ret = await asyncio.wait_for(
            manager.flush_cache(timeout_s=FLUSH_TIMEOUT, verify_empty=True),
            timeout=FLUSH_TIMEOUT + 10,
        )
        if not ret.success:
            raise RuntimeError(f'ax_shapes verified flush failed: {ret.message}')

    async def request(label, ids, lengths, expect_hit=False, expect_cold=False):
        active[:] = [f'{prefix}-{label}-{i}' for i in range(len(ids))]
        params = [{'max_new_tokens': n, 'temperature': 0.0, 'ignore_eos': True}
                  for n in lengths]
        batch = len(ids) > 1
        obj = GenerateReqInput(input_ids=ids if batch else ids[0],
                               rid=list(active) if batch else active[0],
                               sampling_params=params if batch else params[0],
                               stream=False, log_metrics=False)
        outputs = []

        async def drain():
            async for out in manager.generate_request(obj, None):
                outputs.extend(out if isinstance(out, list) else [out])
        await asyncio.wait_for(drain(), timeout=REQUEST_TIMEOUT)
        if len(outputs) != len(ids):
            raise RuntimeError(f'ax_shapes {label}: missing/extra final responses')
        cached = []
        for out, seq, count in zip(outputs, ids, lengths):
            meta = out['meta_info']
            if meta['prompt_tokens'] != len(seq) or meta['completion_tokens'] != count:
                raise RuntimeError(f'ax_shapes {label}: token count mismatch')
            reason = meta.get('finish_reason')
            if not isinstance(reason, dict) or reason.get('type') != 'length':
                raise RuntimeError(f'ax_shapes {label}: unexpected finish {reason!r}')
            cached.append(meta['cached_tokens'])
        if expect_hit and not all(c > 0 for c in cached):
            raise RuntimeError(f'ax_shapes {label}: prefix reuse was not exercised')
        if expect_cold and any(c != 0 for c in cached):
            raise RuntimeError(f'ax_shapes {label}: cache survived flush')
        logger.info('ax_shapes case=%s batch=%d prompt_lengths=%s cached_tokens=%s',
                    label, len(ids), [len(x) for x in ids], cached)
        active.clear()

    try:
        # Also verifies supported pool interfaces before allocating any request.
        await flush()
        role_enabled = (os.environ.get('SGLANG_AX_KDA_DUAL_SNAPSHOT') == '1'
                        or all(x in os.environ.get('SGLANG_ARENA_ROLE_BOUNDARY_TOKEN_IDS', '').split(',')
                               for x in ('154827', '154829')))
        if not role_enabled:
            logger.warning('ax_shapes role checkpoint feature is off; role snapshot coverage unavailable')
        for label, ids, lengths, hit in shape_plan():
            if label.startswith('role-hit-') and not role_enabled:
                hit = False
            await request(label, ids, lengths, expect_hit=hit)
        await flush()
        # Reuse an EXACT warmup prompt: honest metadata must now report zero.
        _, ids, lengths, _ = next(shape_plan())
        await request('post-flush-probe', ids, lengths, expect_cold=True)
    except BaseException:
        for rid in active:
            manager.abort_request(rid)
        logger.exception('ax_shapes failed; startup will fail after cleanup')
        raise
    finally:
        # The probe creates real cache too. Always flush it (and failed cases).
        # Failure here must propagate: never announce readiness with dirty pools.
        await flush()
    logger.info('ax_shapes complete; finite shape coverage only, caches verified empty')
