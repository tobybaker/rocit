import os
import torch
import lightning as L

from torch.utils.data import Dataset, DataLoader


def get_optimal_num_workers() -> int:
    # SLURM-managed environments
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        return max(int(slurm_cpus) - 1, 0)

    # CPU affinity (Linux; respects cgroups / Docker --cpuset-cpus)
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        # macOS / Windows lack sched_getaffinity
        available = os.cpu_count() or 1

    return max(available - 1, 0)

class ROCITDataModule(L.LightningDataModule):

    
    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.pos_weight = None
        self.num_workers = get_optimal_num_workers()

    def setup(self, stage=None):
        
        labels = torch.tensor(
            [self.train_dataset[i]["tumor_read"] for i in range(len(self.train_dataset))]
        ).float()

        pos = labels.sum()
        neg = labels.numel() - pos

        # Avoid division by zero
        self.pos_weight = neg / pos.clamp(min=1.0)
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
        )