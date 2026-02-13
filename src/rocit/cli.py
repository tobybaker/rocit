import click
import polars as pl
from pathlib import Path
from rocit.preprocessing.extract_pacbio_cpg_info import process_bam
from rocit.preprocessing import tumor_data_labeller,get_aggregate_methylation_distribution_from_dir
from rocit.pipeline import training_wrapper,predict_wrapper
from typing import Any,Optional, List

_IO_READERS: dict[str, tuple] = {
    ".csv": (pl.read_csv, pl.scan_csv),
    ".tsv": (
        lambda p, **kw: pl.read_csv(p, separator="\t", **kw),
        lambda p, **kw: pl.scan_csv(p, separator="\t", **kw),
    ),
    ".parquet": (pl.read_parquet, pl.scan_parquet),
    ".pqt": (pl.read_parquet, pl.scan_parquet),
    ".feather": (pl.read_ipc, pl.scan_ipc),
    ".arrow": (pl.read_ipc, pl.scan_ipc),
    ".ipc": (pl.read_ipc, pl.scan_ipc),
    ".ndjson": (pl.read_ndjson, pl.scan_ndjson),
    # ".json" is dropped as it does not support lazy scanning
}

SUPPORTED_EXTENSIONS = sorted(_IO_READERS.keys())

# Define your standard schema expectations here
STANDARD_CASTS = {
    "chromosome": pl.Categorical,
}



class ValidationError(Exception):
    """Raised when an input DataFrame fails schema validation."""
class DataFramePath(click.Path):
    """Click path type that rejects unsupported tabular-file extensions."""

    name = "DATAFRAME_PATH"

    def __init__(self) -> None:
        super().__init__(exists=True, dir_okay=False, readable=True, path_type=Path)

    def convert(self, value: Any, param: Any, ctx: Any) -> Path:
        path: Path = super().convert(value, param, ctx)
        if path.suffix.lower() not in _IO_READERS:
            self.fail(
                f"Unsupported format '{path.suffix}'. "
                f"Accepted: {', '.join(SUPPORTED_EXTENSIONS)}",
                param, ctx,
            )
        return path
def _enforce_standard_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    Casts columns in *df* to standard types if they exist in STANDARD_CASTS.
    """
    # 1. Identify which standard columns are actually present in this DF
    cast_targets = {
        col: dtype 
        for col, dtype in STANDARD_CASTS.items() 
        if col in df.columns
    }
    
    # 2. Apply casts only if there's work to do
    if cast_targets:
        return df.cast(cast_targets)
    
    return df
def read_dataframe(path: Path,scan:bool=False) -> pl.DataFrame:
    """Read a DataFrame from *path*, dispatching on file extension.

    Raises
    ------
    ValidationError
        If the extension is unsupported or the file cannot be parsed.
    """
    suffix = path.suffix.lower()
    # Get the dispatch pair (eager, lazy)
    dispatch_pair = _IO_READERS.get(suffix)
    
    if dispatch_pair is None:
        raise ValidationError(
            f"Unsupported file format '{suffix}' for {path}. "
            f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Select the correct function: index 0 for eager, index 1 for lazy
    reader = dispatch_pair[1] if scan else dispatch_pair[0]
    try:
        df = reader(path)
        
    except Exception as exc:
        raise ValidationError(
            f"Failed to read '{path}' as '{suffix}' DataFrame: {exc}"
        ) from exc
    return _enforce_standard_schema(df)

def parse_chromosomes(ctx, param, value):
    """
    Parses a space-separated string into a list of chromosomes.
    Validates that the list is not empty and items follow basic conventions.
    """
    if value is None:
        return []
    chromosomes = value.strip().split()

    if not chromosomes:
        raise click.BadParameter("No chromosomes found in input string.")

    for chrom in chromosomes:
        if not chrom.isalnum() and not chrom.startswith("chr"):
             click.echo(f"Warning: '{chrom}' does not look like standard chromosome notation.", err=True)


    return list(set(chromosomes))

@click.group()
@click.version_option() # Optional: adds --version flag automatically
def main():
    """ROCIT: A machine learning classifier that separates tumor and non-tumor reads obtained from long read bulk tumor sequencing."""
    pass

@main.command()
@click.option(
    '--sample-id', 
    required=True, 
    type=str, 
    help='Unique identifier for the sample.'
)
@click.option(
    '--labelled-data', 
    required=True, 
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help='Path to the labelled data file.'
)
@click.option(
    '--sample-distribution', 
    required=True, 
    type=click.Path(exists=True, path_type=Path),
    help='Path to the sample distribution file.'
)
@click.option(
    '--cell-atlas', 
    required=True, 
    type=click.Path(exists=True, path_type=Path),
    help='Path to the cell atlas directory or file.'
)
@click.option(
    '--val-chromosomes', 
    required=True, 
    type=str,
    callback=parse_chromosomes, # <--- Hooking the parser here
    help='Space-separated string of validation chromosomes (e.g., "chr1 chr2").'
)
@click.option(
    '--test-chromosomes', 
    required=True, 
    type=str, 
    callback=parse_chromosomes, # <--- Hooking the parser here
    help='Space-separated string of test chromosomes (e.g., "chr3 chr4").'
)
@click.option(
    '--output-dir',
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    help='Directory where training artifacts will be saved.'
)
@click.option(
    '--cache-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default='/scratch',
    help="Temporary directory for training data."
)
def train(
        sample_id:str,
        labelled_data:Path,
        sample_distribution:Path,
        cell_atlas:Path,
        val_chromosomes:str,
        test_chromosomes:str,
        output_dir:Path,
        cache_dir:Path):
    """
    CLI interface to trigger the training process.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if set(val_chromosomes) & set(test_chromosomes):
        raise ValidationError(f'Overlap between validation chromosomes {val_chromosomes} and test chromosomes {test_chromosomes}.')

    labelled_data = read_dataframe(labelled_data)
    sample_distribution = read_dataframe(sample_distribution)
    cell_atlas = read_dataframe(cell_atlas)
    
    training_wrapper(
        sample_id=sample_id,
        labelled_data=labelled_data,
        sample_distribution=sample_distribution,
        cell_atlas=cell_atlas,
        val_chromosomes=val_chromosomes,
        test_chromosomes=test_chromosomes,
        output_dir=output_dir ,
        cache_dir=cache_dir
    )

    
@main.command()
@click.option(
    "--sample-id", required=True, type=str,
    help="Unique sample identifier.",
)
@click.option(
    "--bam", "bam", required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Path to the sample BAM file.",
)
@click.option(
    "--methylation-dir", "methylation_dir", required=True,
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
    help="Directory containing per-sample methylation data.",
)
@click.option(
    "--copy-number", "copy_number", required=True,
    type=DataFramePath(),
    help=f"Path to copy-number DataFrame.",
)
@click.option(
    "--variants", "variants", required=True,
    type=DataFramePath(),
    help=f"Path to variant calls DataFrame.",
)
@click.option(
    "--haplotags", "haplotags", required=True,
    type=DataFramePath(),
    help=f"Path to haplotag assignments DataFrame.",
)
@click.option(
    "--haploblocks", "haploblocks", required=True,
    type=DataFramePath(),
    help=f"Path to haploblock regions DataFrame.",
)
@click.option(
    "--cluster-labels", "cluster_labels", required=True,
    type=DataFramePath(),
    help=f"Path to cluster label DataFrame.",
)
@click.option(
    "--snv-clusters", "snv_clusters", required=False, default=None,
    type=DataFramePath(),
    help=f"(Optional) Path to SNV cluster assignments DataFrame.",
)
@click.option(
    "--output-dir", "output_dir", required=True,
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    help="Output directory for preprocessed dataframes. Created if absent.",
)
def preprocess(
    sample_id: str,
    bam: Path,
    methylation_dir: Path,
    copy_number: Path,
    variants: Path,
    haplotags: Path,
    haploblocks: Path,
    cluster_labels: Path,
    snv_clusters: Path | None,
    output_dir: Path,
) -> None:
    """Preprocess a sample for ROCIT training.

    Reads, validates, and bundles all required inputs into a
    ROCITPreTrainData object, then runs the preprocessing pipeline.
    """
    click.echo(f"Loading inputs for sample '{sample_id}' ...")

    sample_copy_number = read_dataframe(copy_number)
    sample_variants = read_dataframe(variants)
    sample_haplotags = read_dataframe(haplotags)
    sample_haploblocks = read_dataframe(haploblocks)
    cluster_labels = read_dataframe(cluster_labels)

    snv_cluster_assignments: pl.DataFrame | None = None
    if snv_clusters is not None:
        snv_cluster_assignments = read_dataframe(snv_clusters)
        

    somatic_data = tumor_data_labeller.ROCITSomaticData(
        sample_id=sample_id,
        sample_bam_path=bam,
        sample_methylation_dir=methylation_dir,
        sample_copy_number=sample_copy_number,
        sample_variants=sample_variants,
        sample_haplotags=sample_haplotags,
        sample_haploblocks=sample_haploblocks,
        cluster_labels=cluster_labels,
        snv_cluster_assignments=snv_cluster_assignments,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    labelled_reads = tumor_data_labeller.make_read_labels(somatic_data)
    labelled_methylation_data = tumor_data_labeller.get_labelled_methylation_data(methylation_dir,labelled_reads)

    labelled_reads_out_path = output_dir/'labelled_reads.parquet'
    labelled_methylation_data_out_path = output_dir/'labelled_methylation_data.parquet'

    labelled_reads.to_parquet(labelled_reads_out_path)
    labelled_methylation_data.to_parquet(labelled_methylation_data_out_path)

@main.command()
@click.option(
    '--sample-id', 
    required=True, 
    type=str, 
    help='Unique identifier for the sample.'
)
@click.option(
    '--train-result', 
    required=True, 
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help='Path to labelled checkpoint file.'
)

@click.option(
    '--sample-distribution', 
    required=True, 
    type=click.Path(exists=True, path_type=Path),
    help='Path to the sample distribution file.'
)
@click.option(
    '--cell-atlas', 
    required=True, 
    type=click.Path(exists=True, path_type=Path),
    help='Path to the cell atlas directory or file.'
)

@click.option(
    '--output-dir',
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    help='Directory where training artifacts will be saved.'
)
@click.option(
    '--cache-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default='/scratch',
    help="Temporary directory for training data."
)
@click.option(
    '--read-store', 
    required=False, 
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help='Path to dataframe of read methylation data.'
)
@click.option(
    '--read-store-dir', 
    required=False, 
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help='Path to the directory containing read methylation data.'
)
def predict(
        sample_id:str,
        train_result:Path,
        sample_distribution:Path,
        cell_atlas:Path,
        output_dir:Path,
        read_store:Path,
        read_store_dir:Path,
        cache_dir:Path):
    """
    CLI interface to trigger the training process.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not (read_store or read_store_dir):
        raise click.UsageError("You must provide either --read-store or --read-store-dir.")
    
    if read_store and read_store_dir:
        raise click.UsageError("Arguments --read-store and --read-store-dir are mutually exclusive.")

    if read_store:
        read_store = [read_dataframe(read_store,scan=True).filter(~pl.col('supplementary_alignment'))]
    if read_store_dir:
    
        read_store = [read_dataframe(filepath,scan=True).filter(~pl.col('supplementary_alignment')) for filepath in read_store_dir.iterdir()]
    sample_distribution = read_dataframe(sample_distribution)
    cell_atlas = read_dataframe(cell_atlas)
    
    predict_wrapper(
        sample_id=sample_id,
        train_result=train_result,
        read_store=read_store,
        sample_distribution=sample_distribution,
        cell_atlas=cell_atlas,
        output_dir=output_dir ,
        cache_dir=cache_dir
    )

@main.command()
@click.option(
    '--sample-id',
    type=str, 
    required=True, 
    help="The Sample ID used to name the output file."
)
@click.option(
    '--sample-bam',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input directory containing *cpg_methylation_data.parquet files."
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory where the output file will be saved."
)
@click.option(
    "--index", 
    type=click.Path(exists=True, path_type=Path), 
    help="BAM index file path"
)
@click.option(
    "--min-mapq", 
    default=0, 
    show_default=True, 
    type=int, 
    help="Minimum mapping quality"
)
@click.option(
    "--workers", 
    default=1, 
    show_default=True, 
    type=int, 
    help="Number of parallel workers"
)
@click.option(
    "--chromosomes", 
    callback=parse_chromosomes,
    show_default=False,
    default=None,
    help="Space separated chromosomes to process (default: chr1-chrY)."
)
def extract_bam_methylation(
    bam: Path, 
    output_dir: Path, 
    sample_id: str, 
    index: Optional[Path], 
    min_mapq: int, 
    workers: int, 
    chromosomes: Optional[str]
) -> None:
    """Extract CpG methylation from PacBio BAM files"""
    
    
    chroms_arg = chromosomes if chromosomes else None

    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = process_bam(
        bam_path=bam,
        output_dir=output_dir,
        sample_id=sample_id,
        chromosomes=chroms_arg,
        index_path=index,
        min_mapq=min_mapq,
        n_workers=workers
    )

@main.command()
@click.option(
    '--sample-id',
    type=str, 
    required=True, 
    help="The Sample ID used to name the output file."
)
@click.option(
    '--methylation-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Input directory containing *cpg_methylation_data.parquet files."
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory where the output file will be saved."
)

def extract_cpg_distribution(methylation_dir:Path, output_dir:Path, sample_id:str):
    """Aggregates methylation distribution from a directory of parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    get_aggregate_methylation_distribution_from_dir(methylation_dir,output_dir,sample_id)

if __name__ == "__main__":
    main()