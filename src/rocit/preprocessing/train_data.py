import pickle
import polars as pl
import numpy as np
from rocit.preprocessing import snv_data_labeller,loh_data_labeller
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ROCITPreTrainData:
    sample_id:str
    sample_bam_path:Path
    sample_copy_number:pl.DataFrame
    sample_variants:pl.DataFrame
    sample_haplotags:pl.DataFrame
    sample_haploblocks:pl.DataFrame
    
    cluster_labels:pl.DataFrame
    snv_cluster_assignments: pl.DataFrame| None=None

    def save(self, path: str):
        """Saves the entire object to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'ROCITPreTrainData':
        """Reloads the object from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)



def make_read_labels(pretrain_data):

    snv_labelled_reads = snv_data_labeller.get_tumor_labelled_reads(pretrain_data)
    print(snv_labelled_reads)
    loh_labelled_reads = loh_data_labeller.get_tumor_labelled_reads(pretrain_data)
    print(loh_labelled_reads)
    exit()

    return read_labels