import pickle
import polars as pl
import numpy as np
from rocit.preprocessing import snv_data_labeller,loh_data_labeller,prepare_somatic_data
from pathlib import Path
from dataclasses import dataclass
from rocit.constants import HUMAN_CHROMOSOME_ENUM
@dataclass
class ROCITSomaticData:
    sample_id:str
    sample_bam_path:Path
    sample_methylation_dir:Path
    sample_copy_number:pl.DataFrame
    sample_variants:pl.DataFrame
    sample_haplotags:pl.DataFrame
    sample_haploblocks:pl.DataFrame
    snv_clusters:pl.DataFrame
    snv_cluster_assignments: pl.DataFrame| None = None

    def save(self, path: str):
        """Saves the entire object to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'ROCITSomaticData':
        """Reloads the object from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    def __post_init__(self):
        prepare_somatic_data.prepare_somatic_data(self)
    


def make_read_labels(somatic_data):

    snv_labelled_reads = snv_data_labeller.get_tumor_labelled_reads(somatic_data)
    
    loh_labelled_reads = loh_data_labeller.get_tumor_labelled_reads(somatic_data)
    
    read_labels = pl.concat([snv_labelled_reads,loh_labelled_reads])
    read_labels = read_labels.unique(subset=['read_index'], keep='none', maintain_order=False)
    return read_labels


def load_methylation_df(in_path):
    in_df = pl.scan_parquet(in_path)
    in_df = in_df.filter(~pl.col('supplementary_alignment'))
    in_df = in_df.with_columns(pl.col('methylation').cast(pl.UInt8),pl.col('chromosome').cast(pl.Categorical))
    in_df = in_df.drop(['strand','read_count','supplementary_alignment'])
    return in_df

def get_subsampled_methylation_data(sample_methylation_dir,subsample_rate=0.05,seed=123456):
    subsampled_store =[]
    for in_path in sample_methylation_dir.glob('*cpg_methylation.parquet'):
        in_df = load_methylation_df(in_path)
        read_indices = in_df.select(['chromosome','read_index']).unique().collect()
        sampled_read_indices = read_indices.sample(fraction=subsample_rate,seed=seed)
        in_df = in_df.join(sampled_read_indices.lazy(),how='semi',on=['chromosome','read_index'])
        in_df = in_df.with_columns(pl.lit(False).alias('tumor_read'))
        subsampled_store.append(in_df)
    return pl.concat(subsampled_store)
    
def get_labelled_methylation_data(sample_methylation_dir,read_labels):
    labelled_data_store = []
    read_labels = read_labels.lazy().with_columns(pl.col('chromosome').cast(HUMAN_CHROMOSOME_ENUM),pl.col('read_index').cast(pl.Categorical))
    for in_path in sample_methylation_dir.glob('*_cpg_methylation.parquet'):
        in_df = load_methylation_df(in_path).with_columns(pl.col('chromosome').cast(HUMAN_CHROMOSOME_ENUM))
       
        in_df = in_df.join(read_labels,on=['chromosome','read_index'],how='inner')
        labelled_data_store.append(in_df)
    labelled_data = pl.concat(labelled_data_store)
    
    return labelled_data