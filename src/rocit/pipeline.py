import torch
import polars

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


from rocit.models import ROCITModel,ROCITClassifier
from rocit.data import ROCITDataModule,ReadDataset,EmbeddingStore
from pathlib import Path

@dataclass(frozen=True)
class TrainingParams:
    model_dim:int = 384
    model_heads:int = 6
    model_layers:int = 3
    max_epochs:int = 100
    warmup_steps:int = 100
    learning_rate:float = 1e-4
    probability_threshold:float = 0.5
    gradient_clip_val:float = 1.0
    n_log_steps:int=  50
    early_stopping_patience:int = 5
    batch_size:int = 256
    cell_map_dim:int=84
    sample_distribution_dim:int=19

@dataclass
class ROCITTrainStore():
    train_dataset:ReadDataset
    val_dataset:ReadDataset
    test_dataset:ReadDataset
    embedding_sources:dict

@dataclass
class ROCITInferenceStore():
    inference_dataset:ReadDataset
    embedding_sources:dict

@dataclass(frozen=True)
class ROCITTrainResult():
    best_checkpoint_path: Path
    log_dir: Path

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

    data_module = ROCITDataModule(rocit_dataset.train_dataset,rocit_dataset.test_dataset,rocit_dataset.val_dataset,training_params.batch_size)

    warmup_steps = training_params.warmup_steps

    model = ROCITModel(
    model_dim=training_params.model_dim,
    model_heads=training_params.model_heads,
    model_layers=training_params.model_layers,
    lr=training_params.learning_rate,
    warmup_steps=warmup_steps,
    threshold=training_params.probability_threshold,
    sample_distribution_dim=training_params.sample_distribution_dim,
    cell_map_dim=training_params.cell_map_dim
    )
    model.model.set_embedding_context(rocit_dataset.embedding_sources)

    trainer = pl.Trainer(
    max_epochs=training_params.max_epochs,
    accelerator="auto",
    devices="auto",
    gradient_clip_val=training_params.gradient_clip_val,
    callbacks=[early_stopping, checkpointing],
    log_every_n_steps=training_params.n_log_steps,
    logger=logger
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    
    best_checkpoint_path = Path(checkpointing.best_model_path)
    log_dir = Path(logger.log_dir)
    
    return ROCITTrainResult(best_checkpoint_path,log_dir)


def predict(inference_datastore,training_result,inference_batch_size:int=1024):
    
    model = ROCITModel.load_from_checkpoint(training_result.best_checkpoint_path)
    model.model.set_embedding_context(inference_datastore.embedding_sources)
    trainer =pl.Trainer(accelerator="auto", devices=1)

    predict_loader =  DataLoader(
            inference_datastore.inference_dataset,
            batch_size=inference_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=10,
        )

    predictions = trainer.predict(model, dataloaders=predict_loader)
    predictions = polars.concat([polars.from_dict(batch) for batch in predictions])
    return predictions

