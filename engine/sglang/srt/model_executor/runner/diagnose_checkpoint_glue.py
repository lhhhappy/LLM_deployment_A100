"""Independent debug engine only: paired eager/glue waves and checkpoint reads.

CPU copies/synchronization are intentional. Checks actual all-layer conv/SSM
destination states after the model graph, without changing the track mask or
state. Metadata's synthetic mutation checks are installed separately and restore
inputs before this hook. This file has not run on TP8.
"""
import json
from pathlib import Path

import torch


def install(output_dir):
    from sglang.srt.model_executor.runner.diagnose_metadata_glue import install as install_metadata
    from sglang.srt.model_executor.runner.metadata_glue_graph import MetadataGlueGraph
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
    from sglang.srt.runtime_context import mamba_track_grid

    if getattr(MetadataGlueGraph, '_checkpoint_diagnostic_installed', False):
        return
    MetadataGlueGraph._checkpoint_diagnostic_installed = True
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rank = get_tensor_model_parallel_rank()
    config = dict(rank=rank, track_grid=mamba_track_grid(64),
                  scope='Diagnostic eager/glue output waves and actual model-graph checkpoint copies; no performance claim')
    (out/f'checkpoint-config-rank{rank}.json').write_text(json.dumps(config)+'\n')
    install_metadata(output_dir)
    metadata_run = MetadataGlueGraph.run
    load_batch = DecodeCudaGraphRunner.load_batch
    execute = DecodeCudaGraphRunner.execute
    phases = {}

    def phase():
        return json.loads((out/'phase.json').read_text())['mode']

    def phased_run(glue, backend, view, key):
        if phase() == 'eager':
            backend.init_forward_metadata_out_graph(view)
        else:
            metadata_run(glue, backend, view, key)

    def checked_load(runner, *args, **kwargs):
        result = load_batch(runner, *args, **kwargs)
        runner._checkpoint_diagnostic_pending = None
        if runner.raw_bs != 32 or runner.bs != 32:
            return result
        mode = phase()
        state = phases.setdefault(mode, dict(rank=rank, mode=mode, stage=0, checks=[], steps_bs32=0))
        state['steps_bs32'] += 1
        if state['stage'] >= 3:
            return result
        linear = runner.attn_backend.linear_attn_backend
        pools = linear._track_pools()
        if pools is None or runner.buffers.mamba_track_mask is None:
            raise RuntimeError('checkpoint diagnostic requires actual all-layer track pools/mask')
        conv, ssm, _ = pools
        metadata = linear.forward_metadata
        mask = runner.buffers.mamba_track_mask[:32].cpu().tolist()
        sources = metadata.mamba_cache_indices[:32].cpu().tolist()
        destinations = metadata.mamba_track_indices[:32].cpu().tolist()
        active = [i for i in range(32) if mask[i]]
        for index in active:
            src, dst = sources[index], destinations[index]
            if not (0 <= src < conv.shape[1] and 0 <= dst < conv.shape[1] and src != dst):
                raise RuntimeError(f'invalid/aliased active checkpoint slots: row={index} src={src} dst={dst}')
        if state['stage'] == 1:
            if not active:
                return result
            pending = dict(kind='true', rows=active, sources=sources, destinations=destinations, pools=(conv, ssm), state=state,
                           seq_lens=runner.buffers.seq_lens[:32].cpu().tolist())
        else:
            active_destinations = {destinations[i] for i in active}
            false_rows = [i for i in range(32) if not mask[i] and 0 <= destinations[i] < conv.shape[1]
                          and destinations[i] not in sources and destinations[i] not in active_destinations]
            if not false_rows:
                return result
            row = false_rows[0]
            dst = destinations[row]
            # One whole destination across every actual layer; bounded host-only
            # copies. No persistent GPU scratch or full-pool clone is introduced.
            before = (conv[:, dst].cpu().clone(), ssm[:, dst].cpu().clone())
            pending = dict(kind='false_before' if state['stage'] == 0 else 'false_after', row=row,
                           source=sources[row], destination=dst, before=before, pools=(conv, ssm), state=state,
                           seq_len=int(runner.buffers.seq_lens[row].item()))
        runner._checkpoint_diagnostic_pending = pending
        return result

    def checked_execute(runner, *args, **kwargs):
        output = execute(runner, *args, **kwargs)
        pending = getattr(runner, '_checkpoint_diagnostic_pending', None)
        if pending is None:
            return output
        conv, ssm = pending['pools']
        state = pending['state']
        entry = dict(kind=pending['kind'], step=state['steps_bs32'], conv_layers=conv.shape[0], ssm_layers=ssm.shape[0])
        try:
            if pending['kind'] == 'true':
                entries = []
                for row in pending['rows']:
                    src, dst = pending['sources'][row], pending['destinations'][row]
                    conv_equal = torch.equal(conv[:, src].cpu(), conv[:, dst].cpu())
                    ssm_equal = torch.equal(ssm[:, src].cpu(), ssm[:, dst].cpu())
                    entries.append(dict(row=row, source=src, destination=dst, seq_len=pending['seq_lens'][row],
                                        conv_exact=conv_equal, ssm_exact=ssm_equal))
                entry['rows'] = entries
                passed = all(item['conv_exact'] and item['ssm_exact'] for item in entries)
            else:
                dst = pending['destination']
                entry.update(row=pending['row'], source=pending['source'], destination=dst, seq_len=pending['seq_len'],
                             conv_unchanged=torch.equal(pending['before'][0], conv[:, dst].cpu()),
                             ssm_unchanged=torch.equal(pending['before'][1], ssm[:, dst].cpu()))
                passed = entry['conv_unchanged'] and entry['ssm_unchanged']
            entry['passed'] = passed
            state['checks'].append(entry)
            if not passed:
                state['status'] = 'FAIL_CHECKPOINT'
                raise RuntimeError(f'checkpoint comparison failed: {entry}')
            state['stage'] += 1
            state['status'] = 'PASS_FALSE_TRUE_FALSE_CHECKPOINT' if state['stage'] == 3 else 'INCOMPLETE'
            return output
        finally:
            runner._checkpoint_diagnostic_pending = None
            (out/f'checkpoint-{state["mode"]}-rank{rank}.json').write_text(json.dumps(state, indent=2)+'\n')

    MetadataGlueGraph.run = phased_run
    DecodeCudaGraphRunner.load_batch = checked_load
    DecodeCudaGraphRunner.execute = checked_execute
