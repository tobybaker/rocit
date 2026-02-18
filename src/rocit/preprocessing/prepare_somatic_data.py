"""Preparation utilities for ROCITSomaticData.

Contains per-dataframe validation and preparation logic. Each prepare
function validates its input and returns a (potentially cast) DataFrame.
These are intended for use in __post_init__ to ensure dataclass fields
are schema-conformant and consistently typed.
"""

import polars as pl

from rocit.constants import HUMAN_CHROMOSOME_ENUM
from rocit.validation import (
    FLOAT_TYPES,
    INTEGER_TYPES,
    ID_TYPES,
    STRING_TYPES,
    validate_bam,
    validate_chromosome_values,
    validate_columns,
    validate_directory,
)


# ---------------------------------------------------------------------------
# Chromosome preparation
# ---------------------------------------------------------------------------

def cast_chromosome_enum(
    df: pl.DataFrame,
    col: str = "chromosome",
) -> pl.DataFrame:
    """Cast a chromosome column to the hg38 Enum dtype."""
    return df.with_columns(pl.col(col).cast(HUMAN_CHROMOSOME_ENUM))


def prepare_chromosome_column(
    df: pl.DataFrame,
    df_name: str,
    col: str = "chromosome",
) -> pl.DataFrame:
    """Validate chromosome values and cast to Enum in one step."""
    validate_chromosome_values(df, df_name, col)
    return cast_chromosome_enum(df, col)


# ---------------------------------------------------------------------------
# Per-dataframe validation and preparation
# ---------------------------------------------------------------------------

def prepare_copy_number(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and prepare sample_copy_number."""
    validate_columns(df, "sample_copy_number", {
        "chromosome": STRING_TYPES,
        "segment_start": INTEGER_TYPES,
        "segment_end": INTEGER_TYPES,
        "major_cn": INTEGER_TYPES,
        "minor_cn": INTEGER_TYPES,
        "purity": FLOAT_TYPES,
        "normal_total_cn": INTEGER_TYPES,
        "normal_minor_cn": INTEGER_TYPES,
    })
    df = prepare_chromosome_column(df, "sample_copy_number")
    df = df.with_columns((pl.col('segment_end')-pl.col('segment_start')).alias('segment_length'))
    df = df.with_columns((pl.col('major_cn')+pl.col('minor_cn')).alias('total_cn'))
    return df


def prepare_variants(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and prepare sample_variants."""
    validate_columns(df, "sample_copy_number", {
        "chromosome": STRING_TYPES,
        "position": INTEGER_TYPES,
        "ref": STRING_TYPES,
        "alt": STRING_TYPES,
        "tumor_ref_count": INTEGER_TYPES,
        "tumor_alt_count": INTEGER_TYPES,
        
    })
    df = prepare_chromosome_column(df, "sample_copy_number")
    df = df.with_columns((pl.col('tumor_alt_count')/(pl.col('tumor_alt_count')+pl.col('tumor_ref_count'))).alias('vaf'))
    return df


def prepare_haplotags(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and prepare sample_haplotags."""
    validate_columns(df, "sample_haplotags", {
        "chromosome": STRING_TYPES,
        "block_id": ID_TYPES,
        "read_index": ID_TYPES,
        "haplotag": ID_TYPES
    })

    # Validate haplotag values are only 1 or 2
    unique_haplotags = df["haplotag"].unique().sort()
    invalid_haplotags = unique_haplotags.filter(~unique_haplotags.is_in([1, 2]))
    if len(invalid_haplotags) > 0:
        raise ValueError(
            f"sample_haplotags: 'haplotag' column must only contain values 1 or 2. "
            f"Found invalid values: {invalid_haplotags.to_list()}"
        )

    df = prepare_chromosome_column(df, "haplotags")
    return df


def prepare_haploblocks(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and prepare sample_haploblocks."""
    validate_columns(df, "sample_haploblocks", {
        "chromosome": STRING_TYPES,
        "block_id": ID_TYPES,
        "block_start": INTEGER_TYPES,
        "block_end": INTEGER_TYPES
    })
    df = prepare_chromosome_column(df, "haploblocks")
    df = df.with_columns((pl.col('block_end')-pl.col('block_start')).alias('block_size'))
    return df


def prepare_snv_clusters(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and prepare snv_clusters."""
    validate_columns(df, "snv_clusters", {
        "cluster_id": ID_TYPES,
        "cluster_ccf": FLOAT_TYPES,
        "cluster_fraction": FLOAT_TYPES,
    })
    return df


def prepare_snv_cluster_assignments(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and prepare snv_cluster_assignments."""
    validate_columns(df, "snv_cluster_assignments", {
        "chromosome": STRING_TYPES,
        "position": INTEGER_TYPES,
        "cluster_id": ID_TYPES,
        "n_copies": INTEGER_TYPES,
    })
    df = prepare_chromosome_column(df, "snv_cluster_assignments")
    return df


def prepare_somatic_data(obj) -> None:
    """Run all validations and preparations on a ROCITSomaticData instance."""
    validate_bam(obj.sample_bam_path)
    validate_directory(obj.sample_methylation_dir, "methylation directory")

    obj.sample_copy_number = prepare_copy_number(obj.sample_copy_number)
    obj.sample_variants = prepare_variants(obj.sample_variants)
    obj.sample_haplotags = prepare_haplotags(obj.sample_haplotags)
    obj.sample_haploblocks = prepare_haploblocks(obj.sample_haploblocks)
    obj.snv_clusters = prepare_snv_clusters(obj.snv_clusters)

    if obj.snv_cluster_assignments is not None:
        obj.snv_cluster_assignments = prepare_snv_cluster_assignments(
            obj.snv_cluster_assignments
        )