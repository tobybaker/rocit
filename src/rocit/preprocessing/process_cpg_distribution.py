import polars as pl
from pathlib import Path
METHYLATION_SCALE = 256.0
def get_aggregate_methylation_distribution(methylation_df:pl.DataFrame,min_n_cpgs:int=10):
    methylation_df = methylation_df.drop_nulls("position")

    methylation_dtype = methylation_df.collect_schema()['methylation']
    if methylation_dtype != pl.UInt8:
        raise ValueError(
            f"Column methylation has dtype {methylation_dtype}, expected UInt8"
        )
    

    percentiles = [x/100.0 for x in range(5,100,5)]
    methylation_df = methylation_df.with_columns(
        pl.col("methylation").cast(pl.Float32)/METHYLATION_SCALE + 0.5/METHYLATION_SCALE
    )
    aggregate_methylation_df = methylation_df.group_by(['chromosome','position']).agg(
    pl.when(pl.col("methylation").count() >= min_n_cpgs)
      .then(pl.col("methylation").quantile(p).cast(pl.Float32))
      .otherwise(None)
      .alias(f"methylation_percentile_{int(p * 100)}")
    for p in percentiles
    ).sort(['chromosome','position'])

    return aggregate_methylation_df

def get_aggregate_methylation_distribution_from_dir(methylation_dir:Path,output_dir:Path,sample_id:str):
    df_store = []
    for filepath in methylation_dir.glob('*_cpg_methylation.parquet'):
        in_df = pl.scan_parquet(filepath)
        df_store.append(in_df)
    df_store = pl.concat(df_store)
    aggregate_distribution = get_aggregate_methylation_distribution(df_store)
    output_path = output_dir/f'{sample_id}_methylation_distribution.parquet'
    aggregate_distribution.sink_parquet(output_path)

