import torch
import pandas as pd
import numpy as np
import polars as pl

import torch.nn.functional as F

from pathlib import Path
from dataclasses import dataclass


from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

class ReadDataset(Dataset):

    METHYLATION_SCALE = 256.0

    def __init__(self, read_df,label_cols,key_cols,embedding_sources,max_len=511):

        
        self.label_cols = label_cols
        self.max_len = max_len
        self.key_cols = ['Sample_ID','Read_Index']
        self.embedding_index_cols = [embedding_source.index_col for embedding_source in embedding_sources.values()]
        self.read_data = self._process_read_df(read_df,embedding_sources)
        

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
        
    def _process_read_df(self,read_df:pd.DataFrame,embedding_sources):
        self._validate_read_df(read_df)

        drop_cols = ['Supplementary_Alignment','Read_Count','Strand']
        read_df = read_df.drop(drop_cols,strict=False)
        
        for source_name,embedding_source in embedding_sources.items():
            read_df = embedding_source.merge_with_read_df(read_df)
        
        
        n_indexes =read_df.select( pl.struct(self.key_cols).n_unique()).item()
        read_groups = read_df.partition_by(self.key_cols, maintain_order=False)
    
        read_data = {}
        for i, read_index_df in tqdm(enumerate(read_groups),desc='Loading reads'):
            read_data[i] = self._get_processed_read_index_data(read_index_df)
            
        
        return read_data



    def _get_processed_read_index_data(self, read_index_df: pl.DataFrame) -> dict:
        processed = {}
        
        # Access first element of a column
        for label_col in self.label_cols:
            processed[label_col.lower()] = read_index_df.get_column(label_col)[0]
        
        # Height is the polars equivalent of len() for row count
        processed['n_cpgs'] = read_index_df.height
        
        # Column arithmetic using expressions, then extract to numpy
        read_pos_col = read_index_df.get_column('Read_Position')
        read_positions = ((read_pos_col - read_pos_col.min()) / 20000.0 - 0.5).cast(pl.Float32)
        processed['read_position'] = torch.from_numpy(read_positions.to_numpy())
        
        # Direct cast and numpy conversion
        processed['position'] = torch.from_numpy(
            read_index_df.get_column('Position').cast(pl.Float64).to_numpy()
        )
        
        meth_col = read_index_df.get_column('Methylation')
        read_methylation = (meth_col / self.METHYLATION_SCALE + 0.5 / self.METHYLATION_SCALE).cast(pl.Float32)
        processed['methylation'] = torch.from_numpy(read_methylation.to_numpy())
        
        for embedding_index_col in self.embedding_index_cols:
            processed[embedding_index_col.lower()] = torch.from_numpy(
                read_index_df.get_column(embedding_index_col).to_numpy()
            )
            
        return processed
        def __len__(self):
            return len(self.read_data)

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
        processed_read_data = self.read_data[idx]
        downsample_indices = self.get_downsample_indices(processed_read_data['n_cpgs'])
        attention_mask = self.get_attention_mask(processed_read_data['n_cpgs'],downsample_indices)

        item_data = {}

        for col in self.label_cols:
            item_data[col.lower()] = processed_read_data[col.lower()]
        
        tensor_cols = ['methylation','read_position','position'] + [e.lower() for e in self.embedding_index_cols]
        for col in tensor_cols:
            item_data[col.lower()] = self.apply_tensor_subsample_and_pad(processed_read_data[col],downsample_indices)

        item_data['attention_mask'] = attention_mask
        
        return item_data
    def __len__(self):
        return len(self.read_data)
    