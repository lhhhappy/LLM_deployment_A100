"""[ax] Small CPU decisions shared by schedulers with the same request queue."""


def rank0_decide(cpu_group, compute):
    """Run compute() only on group rank 0 and return its value on every rank.

    Callers must enter on every group rank under identical control flow. The
    callback returns a small serializable decision; changes to local state are
    not broadcast. Apply the returned decision separately on every rank.
    Use the scheduler's request-plane group, not the global process group.
    """
    import torch.distributed as dist

    if dist.get_world_size(cpu_group) == 1:
        return compute()
    value = [compute() if dist.get_rank(cpu_group) == 0 else None]
    dist.broadcast_object_list(
        value, src=dist.get_global_rank(cpu_group, 0), group=cpu_group
    )
    return value[0]


def same_decision(cpu_group, value):
    """READY's post-COW transaction: compare only CPU admission metadata."""
    import torch.distributed as dist

    size = dist.get_world_size(cpu_group)
    if size == 1:
        return True
    values = [None] * size
    dist.all_gather_object(values, value, group=cpu_group)
    return all(v == values[0] for v in values[1:])
