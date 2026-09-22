# L2 queue (managed by `scripts/l2.py`; see ../L2.md)

`NN-slug/` = one approved item (spec.json, frozen profile hashes, APPROVED). Items run in number order
on the resident 8-GPU service. `STOP` here pauses the queue (`l2.py pause` / `l2.py resume`).
Do not hand-edit items; enqueue with `scripts/l2.py add CONFIG`.
