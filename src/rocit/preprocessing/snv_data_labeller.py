import polars as pl
import numpy as np
from pathlib import Path

from rocit.preprocessing import variant_processing,bam_tools
from scipy.stats import binom,binomtest

def run_general_variant_qc(variant_table: pl.DataFrame, min_variant_reads: int = 3) -> bool:
    n_variant_haplotypes = variant_table.filter(pl.col('contains_snv'))['haplotag'].n_unique()
    
    if n_variant_haplotypes > 1:
        return False

    if variant_table['contains_snv'].sum() < min_variant_reads:
        return False

    if variant_table['haplotag'].n_unique() != 2:
        return False

    return True


def run_phasing_qc(variant_table,min_p_value:float=0.1):
    
    purity = variant_table['purity'][0]
    minor_cn = variant_table['minor_cn'][0]
    total_cn = variant_table['total_cn'][0]
    normal_total_cn = variant_table['normal_total_cn'][0]
    
    expected_minor_share = (minor_cn*purity + (1-purity))/(total_cn*purity + normal_total_cn*(1-purity))
    observed_minor_counts = variant_table['haplotag'].value_counts(normalize=False)['count'].min()
    
    p_value = binomtest(observed_minor_counts,variant_table.height,p=expected_minor_share).pvalue
    
    if p_value<min_p_value:
        return False
    return True


def run_all_copies_qc(variant_table,min_p_value:float=0.1):
  
    purity = variant_table['purity'][0]
    
    variant_haplotype = variant_table.filter(pl.col('contains_snv'))['haplotag'][0]
    
    other_haplotype = 1 if variant_haplotype ==2 else 2
    
    variant_allele = 'major' if (variant_table['haplotag']==variant_haplotype).mean() >=0.5 else 'minor'

    allele_cn = variant_table[f'{variant_allele}_cn'][0]
   
    if allele_cn != variant_table['n_copies'][0]:
        return False
    
    #this is a safe assumption that either of the mutated alleles will have one copy in the normal
    expected_variant_share = allele_cn*purity/(allele_cn*purity+(1-purity))
    variant_haplotype_table = variant_table.filter(pl.col('haplotag')==variant_haplotype)
    observed_variant_share = variant_haplotype_table['contains_snv'].mean()

    p_value = binomtest(variant_haplotype_table['contains_snv'].sum(),variant_haplotype_table.height,p=expected_variant_share).pvalue
    
    if p_value<min_p_value:
        return False

    return True
def get_tumor_reads_with_snv_labels(pretrain_data,min_block_size=5e5):

    labelled_variants = variant_processing.load_labelled_variants(pretrain_data)
    labelled_variants = labelled_variants.filter(pl.col('cluster_label')!='fail')

    valid_haploblocks = pretrain_data.sample_haploblocks.filter(pl.col('block_size')>=min_block_size)
    count = 0

    read_store = []
    for snv_row in labelled_variants.iter_rows(named=True):
        
        snv_labelled_reads = bam_tools.get_variant_reads(snv_row,pretrain_data.sample_bam_path)
        
        snv_labelled_reads = snv_labelled_reads.join(pretrain_data.sample_haplotags,on=['chromosome','read_index'],how='inner')
        snv_labelled_reads = snv_labelled_reads.filter(pl.col('block_id').is_in(valid_haploblocks['block_id']))
        
        if not run_general_variant_qc(snv_labelled_reads):
            continue
        if not run_phasing_qc(snv_labelled_reads):
            continue
        #if variant is generally passes qc, add to tumor read store
        snv_containing_reads = snv_labelled_reads.filter(pl.col('contains_snv'))
        tumor_reads = snv_containing_reads.select('read_index').with_columns(pl.lit(True).alias('tumor_read'))
        
        read_store.append(tumor_reads)
        all_copies_qc = run_all_copies_qc(snv_labelled_reads)
        #if clonal and on all copies, get reads with same haplotag
        if all_copies_qc:
            variant_haplotype = snv_containing_reads['haplotag'][0]
            non_tumor_reads =  snv_labelled_reads.filter((~pl.col('contains_snv')) & (pl.col('haplotag')==variant_haplotype))
            
            non_tumor_reads = non_tumor_reads.select('read_index').with_columns(pl.lit(False).alias('tumor_read'))
            
            read_store.append(non_tumor_reads)
        if len(read_store)>200:
            break
    return pl.concat(read_store)