import os
import torch.distributed as dist

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # 適当な数字で設定すればいいらしいがよくわかっていない

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
