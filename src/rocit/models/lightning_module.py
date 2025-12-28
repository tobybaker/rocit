import torch
import pytorch_lightning as pl

from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW

import torchmetrics
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    BinaryMatthewsCorrCoef
)
from torch.utils.data import Dataset, DataLoader

class ROCITModel(pl.LightningModule):
    def __init__(
        self,
        model,
        lr: float,
        warmup_steps: int,
        threshold: float = 0.5,
    ):
        super().__init__()

        # ---- Core components ----
        self.model = model
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.threshold = threshold
        self.pos_weight = None 

        self.loss_fn = BCEWithLogitsLoss()

      
        metric_fns = {
            "acc": BinaryAccuracy(threshold=threshold),
            "precision": BinaryPrecision(threshold=threshold),
            "recall": BinaryRecall(threshold=threshold),
            "f1": BinaryF1Score(threshold=threshold),
            "mcc":BinaryMatthewsCorrCoef(threshold=threshold),
            "auroc": BinaryAUROC(),
        }

        self.train_metrics = torchmetrics.MetricCollection(
            metric_fns, prefix="train_"
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        # Enables checkpoint re-loading
        self.save_hyperparameters(ignore=["model"])

    def setup(self, stage=None):
        # datamodule is attached by the Trainer
        if self.trainer.datamodule is None:
            return
        pos_weight = self.trainer.datamodule.pos_weight

        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(self.device)
        )


    def forward(self, **inputs) -> torch.Tensor:
        """
        Returns logits of shape (B,)
        """
        
        logits = self.model(**inputs)
        
        return logits

    def _shared_step(self, batch):
        labels = batch["tumor_read"].float()
        logits = self(**batch)

        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)

        return loss, probs, labels.int()


    def training_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch)

        self.train_metrics.update(probs, labels)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch)

        self.val_metrics.update(probs, labels)
        self.log("val_loss",loss)

    def test_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch)

        self.test_metrics.update(probs, labels)
        self.log("test_loss", loss)

    
    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

   
    def predict_step(self, batch, batch_idx):
        logits = self(**batch)
        probs = torch.sigmoid(logits)

        return_dict= {
            "Sample_ID": batch['sample_id'],
            "Read_Index": batch['read_index'],
            "Chromosome": batch['chromosome'],
            "Tumor_Probability": probs.numpy(force=True)
        }
        if 'tumor_read' in batch:
            return_dict['Tumor_Read'] = batch['tumor_read'].bool().numpy(force=True)
        return return_dict

    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i /(self.warmup_steps), 1.0))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
