from contextlib import contextmanager
import torch
# from src.dataset import MyCollate, Seq2EditDataset
from torch.utils.data import DataLoader
import json
import torch.multiprocessing as mp

def read_config(path):
    with open(path, "r", encoding="utf8") as fr:
        config = json.load(fr)
    return config


# def init_sampler(dataset, shuffle: bool, is_distributed: bool):
#     if is_distributed:
#         sampler = torch.utils.data.DistributedSampler(dataset=dataset,
#                                      shuffle=shuffle,
#                                      drop_last=True)
#     else:
#         sampler = None
#     return sampler


def init_dataloader(dataset=None,
                    batch_size=64,
                    shuffle=True,
                    num_workers=10,
                    collate_fn=None):

    is_distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1

    if is_distributed:
        # sampler option is mutually exclusive with shuffle
        shuffle = None
        batch_size = int(batch_size / torch.distributed.get_world_size()) \
            if torch.distributed.is_initialized() and (torch.distributed.get_world_size() > 1) else batch_size

    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    kwargs = dict()
    if (is_distributed and num_workers > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'
        
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        **kwargs
    )
    return data_loader




@contextmanager
def torch_distributed_master_process_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()
