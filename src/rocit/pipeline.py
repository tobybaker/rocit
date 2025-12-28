import torch
import polars

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from rocit.models.lightning_module import ROCITModel
from rocit.datamodule import ROCITDataModule,ReadDataset,EmbeddingStore

from classifier import ROCITClassifier

@dataclass
class TrainingParams:
    model_dim:int = 384
    model_heads:int=6
    model_layers:int = 3
    max_epochs:int = 100
    warmup_steps:int = 100
    learning_rate:float = 1e-4
    probability_threshold:float =0.5
    gradient_clip_val:float = 1.0
    n_log_steps:int=50
    early_stopping_patience:int = 5
    batch_size: int = 256


@dataclass
class ROCITTrainStore():
    train_dataset:ReadDataset
    val_dataset:ReadDataset
    test_dataset:ReadDataset
    embedding_sources:dict

def train(rocit_dataset,log_dir,experiment_name,training_params=None):
    if training_params is None:
        training_params = TrainingParams()
    torch.set_float32_matmul_precision('high') 
    
    logger = CSVLogger(
    save_dir=log_dir,
    name=experiment_name,  
    )

    early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=training_params.early_stopping_patience,
    mode="min",
    )

    checkpointing = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-checkpoint",
    )

    rocit_model = ROCITClassifier(training_params.model_dim,training_params.model_heads,training_params.model_layers)
    rocit_model.set_embedding_context(rocit_dataset.embedding_sources)

    data_module = ROCITDataModule(rocit_dataset.train_dataset,rocit_dataset.test_dataset,rocit_dataset.val_dataset,training_params.batch_size)
    
    EPOCHS = training_params.max_epochs

    warmup_steps = training_params.warmup_steps

    model = ROCITModel(
    model=rocit_model,
    lr=training_params.learning_rate,
    warmup_steps=warmup_steps,
    threshold=training_params.probability_threshold,
    )

    trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="auto",
    devices="auto",
    gradient_clip_val=training_params.gradient_clip_val,
    callbacks=[early_stopping, checkpointing],
    log_every_n_steps=training_params.n_log_steps,
    logger=logger
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)