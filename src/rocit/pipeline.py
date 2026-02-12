import torch
import polars as pl

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from rocit.data import ReadDataset,EmbeddingStore,ReadDatasetBuilder,ROCITDataModule
from rocit.models import ROCITModel
from pathlib import Path

from typing import List

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
    early_stopping_patience:int = 10
    batch_size:int = 256
    cell_map_dim:int=84
    sample_distribution_dim:int=19
    noise_level:float=0.01

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


def get_sample_train_dataset(read_data,sample_distribution,cell_atlas,val_chromosomes,test_chromosomes):
    
    # Filter the list
    all_chromosomes = read_data['chromosome'].unique()
    non_train_chromosomes = val_chromosomes+test_chromosomes
    train_chromosomes = [chrom for chrom in all_chromosomes if chrom not in non_train_chromosomes]
    
    sample_source = EmbeddingStore('sample_distribution',sample_distribution,['chromosome','position'])
    cell_map_source = EmbeddingStore('cell_map',cell_atlas,['chromosome','position'])
    
    embedding_sources = {sample_source.name:sample_source,cell_map_source.name:cell_map_source}

    label_cols = ['sample_id','read_index','chromosome','tumor_read']
    key_cols = ['read_index']
    
    train_read_data = read_data.filter(pl.col("chromosome").is_in(train_chromosomes))
    test_read_data = read_data.filter(pl.col("chromosome").is_in(test_chromosomes))
    val_read_data = read_data.filter(pl.col("chromosome").is_in(val_chromosomes))


    train_dataset_builder = ReadDatasetBuilder(train_read_data,label_cols,key_cols,embedding_sources)
    test_dataset_builder = ReadDatasetBuilder(test_read_data,label_cols,key_cols,embedding_sources)
    val_dataset_builder = ReadDatasetBuilder(val_read_data,label_cols,key_cols,embedding_sources)
    
    return ROCITTrainStore(train_dataset_builder.build(),val_dataset_builder.build(),test_dataset_builder.build(),embedding_sources)

def training_wrapper(sample_id:str,
        labelled_data:pl.DataFrame,
        sample_distribution:pl.DataFrame,
        cell_atlas:pl.DataFrame,
        val_chromosomes:List[str],
        test_chromosomes:List[str],
        output_dir:Path):
    rocit_dataset = get_sample_train_dataset(labelled_data,sample_distribution,cell_atlas,val_chromosomes,test_chromosomes)
    return train(rocit_dataset,output_dir,sample_id)
    
def train(rocit_dataset,log_dir,experiment_name,training_params=None):
    if training_params is None:
        training_params = TrainingParams()
    torch.set_float32_matmul_precision('medium') 
    
    logger = CSVLogger(
    save_dir=log_dir,
    name=experiment_name,  
    )

    early_stopping = EarlyStopping(
    monitor="val_auroc",
    patience=training_params.early_stopping_patience,
    mode="max",
    )

    checkpointing = ModelCheckpoint(
        monitor="val_auroc",
        mode="max",
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
    cell_map_dim=training_params.cell_map_dim,
    noise_level=training_params.noise_level
    )
    model.model.set_embedding_context(rocit_dataset.embedding_sources)

    trainer = L.Trainer(
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
    torch.set_float32_matmul_precision('medium') 
    model = ROCITModel.load_from_checkpoint(training_result.best_checkpoint_path)
    model.model.set_embedding_context(inference_datastore.embedding_sources)
    trainer =L.Trainer(accelerator="auto", devices=1)

    predict_loader =  DataLoader(
            inference_datastore.inference_dataset,
            batch_size=inference_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=10,
        )

    predictions = trainer.predict(model, dataloaders=predict_loader)
    predictions = pl.concat([pl.from_dict(batch) for batch in predictions])
    
    return predictions

