import polars as pl
import numpy as np
from pathlib import Path
from rocit.preprocessing import bam_tools
import time
def join_with_max_overlap(left_df,right_df,left_start_col:str,left_end_col:str,right_start_col:str,right_end_col:str,suffix:str='_right'):
    right_start_col_r = f"{right_start_col}{suffix}"
    right_end_col_r = f"{right_end_col}{suffix}"
    
    right_df = right_df.rename({right_start_col:right_start_col_r,right_end_col:right_end_col_r})
   
    join_conditions = [
        pl.col('chromosome')==pl.col('chromosome_right'),
        pl.col(left_start_col) < pl.col(right_end_col_r),
        pl.col(right_start_col_r) < pl.col(left_end_col),
    ]
    joined_df = left_df.join_where(right_df, *join_conditions)
    
    overlap_expression = (
        pl.min_horizontal(pl.col(left_end_col), pl.col(right_end_col_r))
        - pl.max_horizontal(pl.col(left_start_col), pl.col(right_start_col_r))
    ).alias("_overlap")

    joined_df = joined_df.with_columns(overlap_expression)
    joined_df = (
        joined_df
        .sort("_overlap", descending=True)
        .group_by('read_index', maintain_order=True)
        .first()
        .drop(["_overlap","chromosome_right"])
    )
    joined_df = joined_df.rename({right_start_col_r:right_start_col,right_end_col_r:right_end_col})

    return joined_df
def get_loh_table(somatic_data,max_major_cn:int=4,min_segment_length=1e6):
    
    loh_table =  somatic_data.sample_copy_number.filter((pl.col('minor_cn')==0))
    loh_table = loh_table.filter(pl.col('major_cn').is_between(1,max_major_cn))
    
    valid_chromosomes = [f'chr{x}' for x in range(1,23)]
    loh_table = loh_table.filter(pl.col('chromosome').is_in(valid_chromosomes))
    
    loh_table = loh_table.filter(pl.col('segment_length')>min_segment_length)
    return loh_table

def get_haploblocks(somatic_data,min_block_size=5e5):
    haploblocks = somatic_data.sample_haploblocks
    haploblocks = haploblocks.filter(pl.col('block_size')>=min_block_size)
    haploblocks = haploblocks.drop(['n_variants'])
    return haploblocks

def get_subblocks(haploblocks,subblock_size=int(1e5)):
    subblock_store = []
    for block_row in haploblocks.iter_rows(named=True):
        n_breaks = block_row['block_size']//subblock_size
        
        block_spacing = np.round(np.linspace(block_row['block_start'],block_row['block_end'],n_breaks+1)).astype(int)
        for i in range(block_spacing.size-1):
            row = {'chromosome':block_row['chromosome']}
            row['subblock_id'] = f"{block_row['block_id']}_{i}"
            row['subblock_start'] = block_spacing[i]
            row['subblock_end'] = block_spacing[i+1]
            subblock_store.append(row)
    subblock_df =  pl.DataFrame(subblock_store)
    subblock_df = subblock_df.with_columns(
        pl.col('subblock_id').cast(pl.Categorical),
        pl.col('subblock_start').cast(pl.Int32),
        pl.col('subblock_end').cast(pl.Int32)
    )
    return subblock_df
def get_minor_cn_share(cn_row):
    total_share = cn_row['total_cn']*cn_row['purity'] +cn_row['normal_total_cn']*(1-cn_row['purity'])
    minor_share = cn_row['normal_minor_cn']*(1-cn_row['purity'])+ cn_row['purity']*cn_row['minor_cn']
    return minor_share/total_share
def get_pass_blocks(read_table,block_cols,minor_cn_share,min_coverage:int=20,max_freq_diff:float=0.05):
    
    block_df = read_table.pivot(
        on='haplotag', index=block_cols, values='haplotag', aggregate_function='len'
    ).fill_null(0)

    #both haplotags should be present
    if not '1' in block_df.columns or not '2' in block_df.columns:
        return block_df.select(block_cols).clear()
    
    block_df = block_df.with_columns(
            total = pl.col("1") + pl.col("2"),
            min_count = pl.min_horizontal("1", "2")
        )
    block_df =  block_df.filter(pl.col("total") >= min_coverage)

    block_df = block_df.with_columns(observed_min_share = pl.col("min_count") / pl.col("total"))
    
    block_df = block_df.filter((pl.col("observed_min_share") - minor_cn_share).abs() < max_freq_diff)
    return block_df.select(block_cols)

def get_min_haplotags(read_table,block_cols):
    
    block_df = read_table.pivot(
        on='haplotag', index=block_cols, values='haplotag', aggregate_function='len'
    ).fill_null(0)
    block_df = block_df.with_columns(
            min_haplotag = pl.when(pl.col("1") < pl.col("2")).then(1).otherwise(2)
        )
    return block_df.select(['block_id','min_haplotag'])


def get_tumor_labelled_reads(somatic_data):
    loh_table = get_loh_table(somatic_data)
    sample_haploblocks = get_haploblocks(somatic_data)
    
    read_store = []
    for cn_row in loh_table.iter_rows(named=True):

        minor_cn_share = get_minor_cn_share(cn_row)
        read_table = bam_tools.get_reads_from_cn_row(cn_row,somatic_data.sample_bam_path)
        
        read_table = read_table.join(somatic_data.sample_haplotags.drop('block_id'),how='inner',on=['chromosome','read_index'])
        
        
        cn_haploblocks = sample_haploblocks.filter(pl.col('chromosome')==cn_row['chromosome'])
        cn_subblocks = get_subblocks(cn_haploblocks)
        
       
        read_table = join_with_max_overlap(read_table,cn_haploblocks,'read_start','read_end','block_start','block_end')
        read_table = join_with_max_overlap(read_table,cn_subblocks,'read_start','read_end','subblock_start','subblock_end')
        
        subblock_cols = ['subblock_id','subblock_start','subblock_end']
        pass_subblocks = get_pass_blocks(read_table,subblock_cols,minor_cn_share)

        block_cols = ['block_id','block_start','block_end']
        pass_blocks = get_pass_blocks(read_table,block_cols,minor_cn_share)

        read_table = read_table.join(pass_subblocks,how='inner',on=subblock_cols)
        read_table = read_table.join(pass_blocks,how='inner',on=block_cols)

        if read_table.height ==0:
            continue

        min_haplotags = get_min_haplotags(read_table,block_cols)

        
        read_table = read_table.join(min_haplotags,how='inner',left_on=['block_id','haplotag'],right_on=['block_id','min_haplotag'])

        labelled_reads = read_table.select(['chromosome','read_index'])
        labelled_reads = labelled_reads.with_columns(tumor_read=pl.lit(False))
        read_store.append(labelled_reads)
    return pl.concat(read_store)
        
