import torch
from torch.utils.data import DataLoader
import os

def create_dataloader(dataset, batch_size, shuffle, config):
    """Create DataLoader with configured parallelism and sane defaults."""
    loader_cfg = (config or {}).get('loader', {})

    num_workers = loader_cfg.get('num_workers')
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(cpu_count, 8)
        print(f'Auto-detected num_workers: {num_workers} (CPU cores: {cpu_count})')

    pin_memory = loader_cfg.get('pin_memory')
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    prefetch_factor = loader_cfg.get('prefetch_factor', 2)
    persistent_workers = loader_cfg.get('persistent_workers', False)

    kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'prefetch_factor': prefetch_factor,
        'persistent_workers': persistent_workers,
    }
    if num_workers == 0:
        kwargs.pop('prefetch_factor', None)
        kwargs['persistent_workers'] = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs,
    )
