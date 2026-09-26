"""Diagnostic-only capture of one real pooled-indexer row; no serving imports.

Capture happens after native top-k has run. No score, key, selection or route is
overwritten. Copies still perturb execution, so use the same capture on A/A/B.
The full small-fixture pooled K table is retained to resolve physical locations
without assuming that virtual locations are consecutive.
"""
import torch


def install(stage, records, row=3950):
    from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as ik

    cls = ik.IndexerKPool
    original_plan = cls._get_topk_ragged_kpool_plan
    original_topk = cls._topk_from_kpool_logits
    active = [None]

    def topk(self, logits, pool_lens, *args, **kwargs):
        result = original_topk(self, logits, pool_lens, *args, **kwargs)
        ctx = active[0]
        if ctx is not None and ctx["r0"] <= row < ctx["r1"]:
            i = row - ctx["r0"]
            record = ctx["record"]
            record["computed_here"] = True
            record["logits"] = logits[i].detach().clone()
            record["pool_length"] = pool_lens[i].detach().clone()
            for key in ("row_starts", "seq_lens", "topk_offsets"):
                value = kwargs.get(key)
                if value is not None:
                    record[key] = value[i].detach().clone()
            table = kwargs.get("page_table")
            row_index = kwargs.get("page_table_row_index")
            if table is not None:
                table_row = row_index[i:i+1].long() if row_index is not None else slice(i, i+1)
                record["page_table"] = table[table_row, :ctx["seq_len"]].detach().clone().flatten()
        return result

    def plan(self, forward_batch, layer_id, q_fp8, weights, metadata):
        if stage[0] != "ext" or q_fp8.shape[0] <= row:
            return original_plan(self, forward_batch, layer_id, q_fp8, weights, metadata)
        layout = metadata.attn_metadata.kpool_extend_plan
        n_real = layout.seq_lens_expanded.shape[0]
        shard = None if self.dsa_enable_prefill_cp else ik._ax_indexer_row_shard(n_real)
        r0 = min(shard[2] * shard[3], n_real) if shard else 0
        r1 = min(r0 + shard[3], n_real) if shard else n_real
        q_start = 0
        for i, q_len in enumerate(forward_batch.extend_seq_lens_cpu):
            if q_start <= row < q_start + q_len:
                seq_len = int(forward_batch.extend_prefix_lens_cpu[i] + row - q_start + 1)
                break
            q_start += q_len
        else:
            raise ValueError("indexer capture row is padding")
        record = dict(layer=layer_id, row=row, r0=r0, r1=r1, computed_here=False)
        previous = active[0]
        active[0] = dict(r0=r0, r1=r1, record=record, seq_len=seq_len)
        try:
            result = original_plan(self, forward_batch, layer_id, q_fp8, weights, metadata)
        finally:
            active[0] = previous
        record.update(
            selected=result[row].detach().clone(),
            query_bytes=q_fp8[row].detach().view(torch.uint8).clone(),
            weights=weights[row].detach().clone(),
            pooled_key_bytes=layout.ragged_k_u8.detach().clone(),
            pooled_key_scales=layout.ragged_k_scale.detach().clone(),
            pooled_pages=layout.ragged_concat_page_table.detach().clone(),
            key_start=layout.ragged_q_ks[row].detach().clone(),
            key_end=layout.ragged_q_ke[row].detach().clone(),
        )
        records.append(record)
        return result

    cls._topk_from_kpool_logits = topk
    cls._get_topk_ragged_kpool_plan = plan


def cpu_records(records):
    return [{k: v.cpu() if torch.is_tensor(v) else v for k, v in record.items()} for record in records]
