import torch
import pandas as pd
import numpy as np
import polars as pl

import torch.nn.functional as F

from pathlib import Path
from dataclasses import dataclass

from datasets import Dataset as HFDataset, Features, Sequence, Value

from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from typing import Optional
class EmbeddingStore:
    def __init__(self,name,embedding_df,key_cols):
        self.name = name
        self.index_col = f"{self.name}_Index"
        self.key_cols = key_cols
        self.embedding_df = self._load_embedding_df(embedding_df)
        self.index_df = self._create_index_df()
        
    def __str__(self):
        return self.name
    
    def _validate_embedding_df(self,embedding_df):
        for col in self.key_cols:
            if not col in embedding_df.columns:
                raise ValueError(f'{col} should be an key column for {self.name} but it is not present!')
        n_data_cols = 0
        for col in embedding_df.columns:
            if col in self.key_cols:
                continue
            n_data_cols +=1
            if not embedding_df[col].dtype.is_numeric():
                raise ValueError(f'{col} is a data column for {self.name} but it is not numeric')
        if n_data_cols ==0:
            raise ValueError(f'{self.name} has no data columns')
        if embedding_df.select(self.key_cols).is_duplicated().any():
            raise ValueError(f'{self.name} should be unique up to {"-".join(self.key_cols)}')


    def _load_embedding_df(self,embedding_df):
        embedding_df = embedding_df.drop([c for c in embedding_df.columns if c.startswith("__index_level_")])
        
        self._validate_embedding_df(embedding_df)
        embedding_df = embedding_df.sort(self.key_cols)
        embedding_df = embedding_df.with_columns(
            pl.arange(1, len(embedding_df) + 1, dtype=pl.Int32).alias(self.index_col)
        )
        return embedding_df
    def _create_index_df(self):
        return self.embedding_df.select(self.key_cols+[self.index_col])
    def _validate_read_df(self,read_df):
        for col in self.key_cols:
            if not col in read_df.columns:
                raise ValueError(f'Key column {col} for {self.name} is not in read df')
        if self.index_col in read_df.columns:
            raise ValueError(f'Read df already has index col {self.index_col} for {self.name}')
    def merge_with_read_df(self,read_df):
        self._validate_read_df(read_df)
        read_df = read_df.join(self.index_df, on=self.key_cols, how="left")
        read_df = read_df.with_columns(
        pl.col(self.index_col).fill_null(0).cast(pl.Int32)
        )
        return read_df
    def get_embedding_vector(self):
        drop_cols = self.key_cols +[self.index_col]
        embedding_vector = self.embedding_df.drop(drop_cols).to_numpy()
        embedding_vector = torch.from_numpy(embedding_vector).float()
        return embedding_vector

class ReadDatasetBuilder(TorchDataset):

    def __init__(self, read_df_store,label_cols,key_cols,embedding_sources,max_len=511):

        
        self.label_cols = label_cols
        self.max_len = max_len
        self.key_cols = key_cols
        self.embedding_sources = embedding_sources
        self.embedding_index_cols = [embedding_source.index_col for embedding_source in embedding_sources.values()]
        self.read_df_store = read_df_store
        

    def _validate_read_df(self, read_df: pl.DataFrame):
        required_cols = ['Chromosome', 'Position', 'Read_Position', 'Methylation', 'Read_Index']
        

        for col in required_cols:
            if col not in read_df.columns:
                raise ValueError(f'{col} needs to be in read data')

        non_nullable_columns = ['Read_Index', 'Chromosome', 'Methylation', 'Read_Position']

        for col in non_nullable_columns:
            if read_df[col].is_null().any():
                raise ValueError(f'{col} should not have any NA values')


        if read_df['Methylation'].dtype not in pl.INTEGER_DTYPES:
            raise ValueError('Methylation column should be integer')


        if not read_df['Methylation'].is_between(0, 255).all():
            raise ValueError('Methylation column should only have values between 0 and 255')


        key_cols = ['Read_Index', 'Read_Position']
        if 'Sample_ID' in read_df.columns:
            key_cols.append('Sample_ID')
        
        if read_df.select(key_cols).is_duplicated().any():
            raise ValueError(f'Read data should be unique up to {"-".join(key_cols)}')
        
    def _get_processed_read_index_data(self, read_index_df: pl.DataFrame) -> dict:
        processed = {}
        
        # Access first element of a column
        for label_col in self.label_cols:
            processed[label_col.lower()] = read_index_df.get_column(label_col)[0]
        
        
        processed['n_cpgs'] = read_index_df.height
        processed['methylation'] = read_index_df.get_column('Methylation').cast(pl.UInt16).to_numpy()
        
        # Direct cast and numpy conversion - missing reference positions are mapped to -1
        processed['position'] =  read_index_df.get_column('Position').fill_null(-1).cast(pl.Int32).to_numpy()
       
        processed['read_position'] = read_index_df.get_column('Read_Position').cast(pl.Int32).to_numpy()
        
        for embedding_index_col in self.embedding_index_cols:
            processed[embedding_index_col.lower()] = read_index_df.get_column(embedding_index_col).cast(pl.UInt32).to_numpy()
            
            
        return processed
    
        
    def _process_read_df(self,read_df:pl.DataFrame):
        read_df = read_df.collect() if isinstance(read_df, pl.LazyFrame) else read_df
        self._validate_read_df(read_df)

        drop_cols = ['Supplementary_Alignment','Read_Count','Strand']
        read_df = read_df.drop(drop_cols,strict=False)
        
        for source_name,embedding_source in self.embedding_sources.items():
            read_df = embedding_source.merge_with_read_df(read_df)
        
        
        n_indexes =read_df.select( pl.struct(self.key_cols).n_unique()).item()
        read_groups = read_df.partition_by(self.key_cols, maintain_order=False)
    
        
        for i, read_index_df in enumerate(read_groups):
            yield self._get_processed_read_index_data(read_index_df)
  

    def _read_generator(self):
        read_df_store = self.read_df_store if isinstance(self.read_df_store, list) else [self.read_df_store]
        read_data = {}
        for read_df in read_df_store:
            for read_index_data in self._process_read_df(read_df):
                yield read_index_data
        
    def build(
        self,
        cache_dir: Optional[str] = '/scratch/',
        num_proc: Optional[int] = None,
        fingerprint: Optional[str] = None,
     ):
        
        hf_dataset =  HFDataset.from_generator(
            self._read_generator,
            cache_dir=cache_dir,
            num_proc=num_proc,
            fingerprint=fingerprint,
            split='reads'
        )
        return ReadDataset(
            hf_dataset=hf_dataset,
            label_cols=self.label_cols,
            embedding_index_cols=self.embedding_index_cols
        )
    
    

class ReadDataset(TorchDataset):

    METHYLATION_SCALE = 256.0
    BASE_READ_LENGTH = 20000.0

    def __init__(self, hf_dataset,label_cols,embedding_index_cols,max_len=511):

        self.hf_dataset = hf_dataset
        self.label_cols = label_cols
        self.max_len = max_len
        self.embedding_index_cols =embedding_index_cols
        
        
    def __len__(self) -> int:
        return len(self.hf_dataset)
    

    def get_downsample_indices(self,seq_len: int):
        
        if seq_len > self.max_len:
            perm = torch.randperm(seq_len)
            downsample_indices = perm[:self.max_len]
            downsample_indices, _ = torch.sort(downsample_indices) # Maintain order
            return downsample_indices
        return None

    def get_attention_mask(self,seq_len: int,downsample_indices):
        
        #seq length + class token
        mask = torch.zeros(self.max_len+1, dtype=torch.bool)
        
        if downsample_indices is not None:
      
            valid_len = self.max_len
        else:
            
            valid_len = min(seq_len, self.max_len)
            
        # 3. Fill the masked area with True
        mask[(valid_len+1):] = True
        
        return mask
    
    def apply_tensor_subsample_and_pad(self,tensor: torch.Tensor, downsample_indices, pad_value=0):
        if downsample_indices is not None:
            tensor = tensor[downsample_indices]

        current_len = tensor.size(0)
        if current_len < self.max_len:
            pad_size = self.max_len - current_len
            tensor = F.pad(tensor, (0, pad_size), value=pad_value)
            
        return tensor

    def __getitem__(self, idx):
        processed_read_data = self.hf_dataset[idx]
        downsample_indices = self.get_downsample_indices(processed_read_data['n_cpgs'])
        attention_mask = self.get_attention_mask(processed_read_data['n_cpgs'],downsample_indices)

        item_data = {}

        for col in self.label_cols:
            item_data[col.lower()] = processed_read_data[col.lower()]
        if 'tumor_read' in item_data:
            item_data['tumor_read'] = torch.tensor(item_data['tumor_read']).float()
        tensor_cols = ['methylation','read_position','position'] + [e.lower() for e in self.embedding_index_cols]
        for col in tensor_cols:
            
            item_data[col.lower()] = self.apply_tensor_subsample_and_pad(torch.tensor(processed_read_data[col]).long(),downsample_indices)
        item_data['methylation']  = (item_data['methylation'] / self.METHYLATION_SCALE + 0.5 / self.METHYLATION_SCALE).float()
        item_data['read_position']  = ((item_data['read_position'].float() - (item_data['read_position'][0].float())) / self.BASE_READ_LENGTH - 0.5)
        
        item_data['attention_mask'] = attention_mask
        
        return item_data