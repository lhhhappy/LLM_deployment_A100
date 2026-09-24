"""Loaded by every Python process of the dev-box numeric check (PYTHONPATH=this dir).

HC180_WELL_SCALED=1  re-initialise dummy weights so attention matters (same recipe
                     as scripts/analysis/dcp_check.py: stock +-1e-3 dummy init
                     collapses the DSA o_proj input to cancellation noise).
HC180_DUMP_DIR=DIR   dump the input of every DSA layer's o_proj for every EAGER
                     extend pass to DIR/r{tp_rank}_p{pass:05d}.pt (a dict
                     {layer_name: fp32 CPU tensor [tokens, dim]}). The client sends
                     one request at a time, so pass numbers map to requests.
Nothing here changes what the server computes.
"""

import os

if os.environ.get("HC180_WELL_SCALED") == "1" or os.environ.get("HC180_DUMP_DIR"):
    import builtins

    _real_import = builtins.__import__
    _state = {"patched": False}

    def _patch():
        import zlib

        import torch

        import sglang.srt.model_loader.loader as ld
        from sglang.srt.model_executor.model_runner import ModelRunner

        if os.environ.get("HC180_WELL_SCALED") == "1":
            orig_init = ld.initialize_dummy_weights

            def init(model, *a, **k):
                from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

                orig_init(model, *a, **k)
                attn = [
                    n + "."
                    for n, m in model.named_modules()
                    if isinstance(m, DeepseekV2AttentionMLA)
                ]
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if not torch.is_floating_point(p) or torch.finfo(p.dtype).bits < 16:
                            continue
                        g = torch.Generator(device=p.device)
                        g.manual_seed(zlib.crc32(name.encode()))
                        leaf = name.rsplit(".", 1)[-1]
                        in_attn = any(name.startswith(x) for x in attn)
                        if "embed_tokens" in name and p.dim() == 2:
                            p.copy_(torch.randn(p.shape, generator=g, device=p.device).to(p.dtype))
                        elif p.dim() == 1 and "norm" in name and leaf == "weight":
                            p.fill_(1.0)
                        elif in_attn and leaf == "bias":
                            p.zero_()
                        elif in_attn and p.dim() == 2 and leaf == "weight":
                            w = torch.randn(p.shape, generator=g, device=p.device) / p.shape[1] ** 0.5
                            p.copy_(w.to(p.dtype))

            ld.initialize_dummy_weights = init

        dump_dir = os.environ.get("HC180_DUMP_DIR")
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            orig_icg = ModelRunner.init_cuda_graphs

            def icg(self, *a, **k):
                _install(self, dump_dir)
                return orig_icg(self, *a, **k)

            ModelRunner.init_cuda_graphs = icg

    def _install(runner, dump_dir):
        import torch

        from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

        rank = getattr(runner, "tp_rank", 0)
        cur = {"pass": 0, "rows": {}}
        is_draft = bool(getattr(runner, "is_draft_worker", False))
        tag = "d" if is_draft else "t"

        def pre(mod, args, _name):
            if torch.cuda.is_current_stream_capturing():
                return
            cur["rows"][_name] = args[0].detach().float().cpu()

        for name, m in runner.model.named_modules():
            if isinstance(m, DeepseekV2AttentionMLA):
                m.o_proj.register_forward_pre_hook(lambda mod, args, _n=name: pre(mod, args, _n))

        orig_forward = runner.forward

        def forward(forward_batch, *a, **k):
            out = orig_forward(forward_batch, *a, **k)
            if forward_batch.forward_mode.is_extend() and cur["rows"]:
                torch.save(
                    {"extend_seq_lens": forward_batch.extend_seq_lens_cpu, **cur["rows"]},
                    os.path.join(dump_dir, f"{tag}{rank}_p{cur['pass']:05d}.pt"),
                )
                cur["pass"] += 1
            cur["rows"] = {}
            return out

        runner.forward = forward

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        mod = _real_import(name, globals, locals, fromlist, level)
        if (
            not _state["patched"]
            and name.startswith("sglang.srt.model_executor.model_runner")
        ):
            _state["patched"] = True
            _patch()
        return mod

    builtins.__import__ = _import
