import torch
import pytorch_lightning as pl


import torchmetrics
from torch.utils.data import Dataset, DataLoader
class ROCITDataModule(pl.LightningDataModule):

    NUM_WORKERS =10
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
            num_workers=self.NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.NUM_WORKERS,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.NUM_WORKERS,
            drop_last=False,
        )