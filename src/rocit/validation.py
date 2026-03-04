from pathlib import Path
from typing import Sequence

import polars as pl
import pysam

from rocit.constants import HUMAN_CHROMOSOMES


INTEGER_TYPES: list[type[pl.DataType]] = [
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
]

FLOAT_TYPES: list[type[pl.DataType]] = [
    pl.Float32, pl.Float64,
]

NUMERIC_TYPES: list[type[pl.DataType]] = INTEGER_TYPES + FLOAT_TYPES

STRING_TYPES: list[type[pl.DataType]] = [
    pl.Utf8, pl.String,pl.Categorical,pl.Enum
]

ID_TYPES: list[type[pl.DataType]] = INTEGER_TYPES + STRING_TYPES
def validate_columns(
    df: pl.DataFrame,
    df_name: str,
    required_columns: dict[str, type[pl.DataType] | Sequence[type[pl.DataType]]],
) -> None:
    """Check that *df* contains every column in *required_columns* with an
    acceptable dtype.

    Parameters
    ----------
    df : pl.DataFrame
        The dataframe to validate.
    df_name : str
        Human-readable name used in error messages.
    required_columns : dict
        Mapping of ``column_name`` to either a single Polars dtype class or
        a sequence of dtype classes (e.g. ``NUMERIC_TYPES``).

    Raises
    ------
    ValueError
        If a required column is missing or has an unexpected dtype.
    """
    schema = df.schema
    errors: list[str] = []

    for col, expected in required_columns.items():
        if col not in schema:
            errors.append(f"  - missing column '{col}'")
            continue

        acceptable = expected if isinstance(expected, Sequence) else [expected]
        actual = schema[col]

        if not any(isinstance(actual, t) for t in acceptable):
            type_names = ", ".join(t.__name__ for t in acceptable)
            errors.append(
                f"  - column '{col}': expected one of ({type_names}), "
                f"got {type(actual).__name__}"
            )

    if errors:
        raise ValueError(
            f"Validation failed for '{df_name}':\n" + "\n".join(errors)
        )


def validate_chromosome_values(
    df: pl.DataFrame,
    df_name: str,
    col: str = "chromosome",
) -> None:
    """Ensure a chromosome column contains only valid hg38 contigs.

    Raises
    ------
    ValueError
        If any value falls outside chr1-chr22, chrX, chrY.
    """
    allowed = set(HUMAN_CHROMOSOMES)
    observed = set(df[col].unique().to_list())
    invalid = observed - allowed

    if invalid:
        raise ValueError(
            f"Validation failed for '{df_name}': column '{col}' contains "
            f"invalid chromosome values: {sorted(invalid)}"
        )


def validate_bam(bam_path: Path) -> None:
    """Perform basic sanity checks on a BAM file using pysam.

    Checks that the file exists, is a valid BAM, is coordinate-sorted,
    has an associated index, and defines at least one reference sequence.
    """
    if not bam_path.exists():
        raise FileNotFoundError(f"BAM file not found: {bam_path}")

    try:
        af = pysam.AlignmentFile(str(bam_path), "rb")
    except Exception as exc:
        raise ValueError(f"Cannot open BAM file '{bam_path}': {exc}") from exc

    with af:
        header = af.header.to_dict()
        sort_order = header.get("HD", {}).get("SO", "unknown")
        if sort_order != "coordinate":
            raise ValueError(
                f"BAM '{bam_path}' sort order is '{sort_order}'; "
                "expected 'coordinate'."
            )

        if not af.has_index():
            raise ValueError(
                f"BAM '{bam_path}' has no associated index (.bai/.csi). "
                "Run `samtools index` first."
            )

        if af.nreferences == 0:
            raise ValueError(
                f"BAM '{bam_path}' contains no reference sequences in its header."
            )


def validate_directory(dirpath: Path, name: str = "directory") -> None:
    """Validate that a path exists and is a directory.

    Parameters
    ----------
    dirpath : Path
        The path to check.
    name : str
        Human-readable name used in error messages.
    """
    if not dirpath.exists():
        raise FileNotFoundError(f"{name} not found: {dirpath}")
    if not dirpath.is_dir():
        raise ValueError(
            f"Expected a directory for {name}, got file: {dirpath}"
        )