import click
import polars as pl
from pathlib import Path
from rocit.preprocessing.extract_pacbio_cpg_info import process_bam
from rocit.preprocessing import tumor_data_labeller
from rocit.pipeline import training_wrapper
from typing import Any,Optional, List

_READERS: dict[str, Any] = {
    ".csv": pl.read_csv,
    ".tsv": lambda p, **kw: pl.read_csv(p, separator="\t", **kw),
    ".parquet": pl.read_parquet,
    ".pqt": pl.read_parquet,
    ".feather": pl.read_ipc,
    ".arrow": pl.read_ipc,
    ".ipc": pl.read_ipc,
    ".json": pl.read_json,
    ".ndjson": pl.read_ndjson,
}

SUPPORTED_EXTENSIONS = sorted(_READERS.keys())

class ValidationError(Exception):
    """Raised when an input DataFrame fails schema validation."""
class DataFramePath(click.Path):
    """Click path type that rejects unsupported tabular-file extensions."""

    name = "DATAFRAME_PATH"

    def __init__(self) -> None:
        super().__init__(exists=True, dir_okay=False, readable=True, path_type=Path)

    def convert(self, value: Any, param: Any, ctx: Any) -> Path:
        path: Path = super().convert(value, param, ctx)
        if path.suffix.lower() not in _READERS:
            self.fail(
                f"Unsupported format '{path.suffix}'. "
                f"Accepted: {', '.join(SUPPORTED_EXTENSIONS)}",
                param, ctx,
            )
        return path

def read_dataframe(path: Path) -> pl.DataFrame:
    """Read a DataFrame from *path*, dispatching on file extension.

    Raises
    ------
    ValidationError
        If the extension is unsupported or the file cannot be parsed.
    """
    suffix = path.suffix.lower()
    reader = _READERS.get(suffix)
    if reader is None:
        raise ValidationError(
            f"Unsupported file format '{suffix}' for {path}. "
            f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    try:
        return reader(path)
    except Exception as exc:
        raise ValidationError(
            f"Failed to read '{path}' as '{suffix}' DataFrame: {exc}"
        ) from exc

def parse_chromosomes(ctx, param, value):
    """
    Parses a space-separated string into a list of chromosomes.
    Validates that the list is not empty and items follow basic conventions.
    """
    if not value:
        raise click.BadParameter("Chromosome list cannot be empty.")
    

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
    '--labelled-data-path', 
    required=True, 
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help='Path to the labelled data file.'
)
@click.option(
    '--sample-distribution-path', 
    required=True, 
    type=click.Path(exists=True, path_type=Path),
    help='Path to the sample distribution file.'
)
@click.option(
    '--cell-atlas-path', 
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
def train(
        sample_id:str,
        labelled_data_path:Path,
        sample_distribution_path:Path,
        cell_atlas_path:Path,
        val_chromosomes:str,
        test_chromosomes:str,
        output_dir:Path):
    """
    CLI interface to trigger the training process.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if set(val_chromosomes) & set(test_chromosomes):
        raise ValidationError(f'Overlap between validation chromosomes {val_chromosomes} and test chromosomes {test_chromosomes}.')

    labelled_data = read_dataframe(labelled_data)
    sample_distribution = read_dataframe(sample_distribution_path)
    cell_atlas = read_dataframe(cell_atlas_path)
    
    training_wrapper(
        sample_id=sample_id,
        labelled_data=labelled_data,
        sample_distribution=sample_distribution,
        cell_atlas=cell_atlas,
        val_chromosomes=val_chromosomes,
        test_chromosomes=test_chromosomes,
        output_dir=output_dir 
    )

    
@main.command()
@click.option(
    "--sample-id", required=True, type=str,
    help="Unique sample identifier.",
)
@click.option(
    "--bam", "sample_bam_path", required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Path to the sample BAM file.",
)
@click.option(
    "--methylation-dir", "sample_methylation_dir", required=True,
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
    help="Directory containing per-sample methylation data.",
)
@click.option(
    "--copy-number", "copy_number_path", required=True,
    type=DataFramePath(),
    help=f"Path to copy-number DataFrame ({SUPPORTED_EXTENSIONS}).",
)
@click.option(
    "--variants", "variants_path", required=True,
    type=DataFramePath(),
    help=f"Path to variant calls DataFrame ({SUPPORTED_EXTENSIONS}).",
)
@click.option(
    "--haplotags", "haplotags_path", required=True,
    type=DataFramePath(),
    help=f"Path to haplotag assignments DataFrame ({SUPPORTED_EXTENSIONS}).",
)
@click.option(
    "--haploblocks", "haploblocks_path", required=True,
    type=DataFramePath(),
    help=f"Path to haploblock regions DataFrame ({SUPPORTED_EXTENSIONS}).",
)
@click.option(
    "--cluster-labels", "cluster_labels_path", required=True,
    type=DataFramePath(),
    help=f"Path to cluster label DataFrame ({SUPPORTED_EXTENSIONS}).",
)
@click.option(
    "--snv-clusters", "snv_clusters_path", required=False, default=None,
    type=DataFramePath(),
    help=f"(Optional) Path to SNV cluster assignments DataFrame ({SUPPORTED_EXTENSIONS}).",
)
@click.option(
    "--output-dir", "output_dir", required=True,
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    help="Output directory for preprocessed artefacts. Created if absent.",
)
def preprocess(
    sample_id: str,
    sample_bam_path: Path,
    sample_methylation_dir: Path,
    copy_number_path: Path,
    variants_path: Path,
    haplotags_path: Path,
    haploblocks_path: Path,
    cluster_labels_path: Path,
    snv_clusters_path: Path | None,
    output_dir: Path,
) -> None:
    """Preprocess a sample for ROCIT training.

    Reads, validates, and bundles all required inputs into a
    ROCITPreTrainData object, then runs the preprocessing pipeline.
    """
    click.echo(f"Loading inputs for sample '{sample_id}' ...")

    sample_copy_number = read_dataframe(copy_number_path)
    sample_variants = read_dataframe(variants_path)
    sample_haplotags = read_dataframe(haplotags_path)
    sample_haploblocks = read_dataframe(haploblocks_path)
    cluster_labels = read_dataframe(cluster_labels_path)

    snv_cluster_assignments: pl.DataFrame | None = None
    if snv_clusters_path is not None:
        snv_cluster_assignments = read_dataframe(snv_clusters_path)
        

    somatic_data = tumor_data_labeller.ROCITPreTrainData(
        sample_id=sample_id,
        sample_bam_path=sample_bam_path,
        sample_methylation_dir=sample_methylation_dir,
        sample_copy_number=sample_copy_number,
        sample_variants=sample_variants,
        sample_haplotags=sample_haplotags,
        sample_haploblocks=sample_haploblocks,
        cluster_labels=cluster_labels,
        snv_cluster_assignments=snv_cluster_assignments,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    labelled_reads = tumor_data_labeller.make_read_labels(somatic_data)
    labelled_methylation_data = tumor_data_labeller.get_labelled_methylation_data(sample_methylation_dir,labelled_reads)

    labelled_reads_out_path = output_dir/'labelled_reads.parquet'
    labelled_methylation_data_out_path = output_dir/'labelled_methylation_data.parquet'

    labelled_reads.to_parquet(labelled_reads_out_path)
    labelled_methylation_data.to_parquet(labelled_methylation_data_out_path)

@main.command()
def predict():
    """Predict tumor origin for sample."""
    click.echo("Evaluating...")

@main.command()
@click.argument("bam", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, writable=True, path_type=Path))
@click.argument("sample_id", type=str)
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
    help="Space separated chromosomes to process (default: chr1-chrY)."
)
def extract_bam_methylation_info(
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


if __name__ == "__main__":
    main()