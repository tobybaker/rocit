import polars as pl

METHYLATION_SCALE = 256.0
def get_aggregate_methylation_distribution(methylation_df,min_n_cpgs:int=10):
    methylation_df = methylation_df.drop_nulls("position")

    methylation_dtype = methylation_df.schema['methylation']
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

    return aggregate_methylation_df.collect()

