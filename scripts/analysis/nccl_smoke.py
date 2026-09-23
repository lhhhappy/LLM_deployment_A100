# Minimal 2-GPU NCCL all_reduce smoke test (dev-box environment check).
import os, torch, torch.distributed as dist, torch.multiprocessing as mp
def run(rank, world):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29611")
    torch.cuda.set_device(rank); dist.init_process_group("nccl", rank=rank, world_size=world)
    x = torch.ones(1 << 20, device="cuda") * (rank + 1); dist.all_reduce(x); torch.cuda.synchronize()
    if rank == 0: print("ALLREDUCE_OK", x[0].item(), flush=True)
    dist.destroy_process_group()
if __name__ == "__main__":
    mp.spawn(run, args=(2,), nprocs=2)
