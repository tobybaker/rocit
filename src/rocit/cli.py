import click
import polars as pl
from pathlib import Path
from rocit.preprocessing.extract_pacbio_cpg_info import process_bam
from rocit.preprocessing import tumor_data_labeller,get_aggregate_methylation_distribution_from_dir
from rocit.pipeline import training_wrapper,predict_wrapper
from rocit.config import (
    TrainConfig, PredictConfig, PreprocessConfig, RunConfig,
    load_config, ConfigError,
    resolve_file, resolve_dir,
    validate_train_config, validate_predict_config, validate_run_config,
)
from typing import Any, Optional
from rocit.constants import HUMAN_CHROMOSOME_ENUM
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
    "chromosome": HUMAN_CHROMOSOME_ENUM,
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
        if col in df.collect_schema().names()
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
    "--config", "config_path", required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to YAML configuration file for training.",
)
def train(config_path: Path):
    """Train a ROCIT model from a YAML configuration file."""
    cfg = load_config(TrainConfig, config_path)
    validate_train_config(cfg)

    labelled_data_path = resolve_file(cfg.labelled_data)
    sample_distribution_path = resolve_file(cfg.sample_distribution)
    cell_atlas_path = resolve_file(cfg.cell_atlas)
    output_dir = resolve_dir(cfg.output_dir, must_exist=False)
    cache_dir = resolve_dir(cfg.cache_dir, must_exist=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    labelled_data = read_dataframe(labelled_data_path)
    sample_distribution = read_dataframe(sample_distribution_path)
    cell_atlas = read_dataframe(cell_atlas_path)

    training_wrapper(
        sample_id=cfg.sample_id,
        labelled_data=labelled_data,
        sample_distribution=sample_distribution,
        cell_atlas=cell_atlas,
        val_chromosomes=list(set(cfg.val_chromosomes)),
        test_chromosomes=list(set(cfg.test_chromosomes)),
        output_dir=output_dir,
        cache_dir=cache_dir,
    )

    
@main.command()
@click.option(
    "--config", "config_path", required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to YAML configuration file for preprocessing.",
)
def preprocess(config_path: Path) -> None:
    """Preprocess a sample for ROCIT training from a YAML configuration file."""
    cfg = load_config(PreprocessConfig, config_path)

    bam_path = resolve_file(cfg.bam)
    methylation_dir = resolve_dir(cfg.methylation_dir)
    copy_number_path = resolve_file(cfg.copy_number)
    variants_path = resolve_file(cfg.variants)
    haplotags_path = resolve_file(cfg.haplotags)
    haploblocks_path = resolve_file(cfg.haploblocks)
    snv_clusters_path = resolve_file(cfg.snv_clusters)
    output_dir = resolve_dir(cfg.output_dir, must_exist=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading inputs for sample '{cfg.sample_id}' ...")

    sample_copy_number = read_dataframe(copy_number_path)
    sample_variants = read_dataframe(variants_path)
    sample_haplotags = read_dataframe(haplotags_path)
    sample_haploblocks = read_dataframe(haploblocks_path)
    
    snv_clusters = read_dataframe(snv_clusters_path)

    snv_cluster_assignments: pl.DataFrame | None = None
    if cfg.snv_cluster_assignments is not None:
        snv_cluster_assignments_path = resolve_file(cfg.snv_cluster_assignments)
        snv_cluster_assignments = read_dataframe(snv_cluster_assignments_path)

    somatic_data = tumor_data_labeller.ROCITSomaticData(
        sample_id=cfg.sample_id,
        sample_bam_path=bam_path,
        sample_methylation_dir=methylation_dir,
        sample_copy_number=sample_copy_number,
        sample_variants=sample_variants,
        sample_haplotags=sample_haplotags,
        sample_haploblocks=sample_haploblocks,
        snv_clusters=snv_clusters,
        snv_cluster_assignments=snv_cluster_assignments,
    )

    labelled_reads = tumor_data_labeller.make_read_labels(somatic_data)
    labelled_methylation_data = tumor_data_labeller.get_labelled_methylation_data(
        methylation_dir, labelled_reads
    )

    labelled_reads.write_parquet(output_dir / 'labelled_reads.parquet')
    labelled_methylation_data.write_parquet(output_dir / 'labelled_methylation_data.parquet')

@main.command()
@click.option(
    "--config", "config_path", required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to YAML configuration file for prediction.",
)
def predict(config_path: Path):
    """Run ROCIT prediction from a YAML configuration file."""
    cfg = load_config(PredictConfig, config_path)
    validate_predict_config(cfg)

    best_checkpoint_path = resolve_file(cfg.best_checkpoint_path)
    sample_distribution_path = resolve_file(cfg.sample_distribution)
    cell_atlas_path = resolve_file(cfg.cell_atlas)
    output_dir = resolve_dir(cfg.output_dir, must_exist=False)
    cache_dir = resolve_dir(cfg.cache_dir, must_exist=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.read_store:
        rs_path = resolve_file(cfg.read_store)
        read_store = [read_dataframe(rs_path, scan=True).filter(~pl.col('supplementary_alignment'))]
    else:
        rs_dir = resolve_dir(cfg.read_store_dir)
        read_store = [
            read_dataframe(fp, scan=True).filter(~pl.col('supplementary_alignment'))
            for fp in rs_dir.iterdir()
            if fp.is_file() and fp.suffix.lower() in _IO_READERS
        ]

    sample_distribution = read_dataframe(sample_distribution_path)
    cell_atlas = read_dataframe(cell_atlas_path)



    predict_wrapper(
        sample_id=cfg.sample_id,
        best_checkpoint_path=best_checkpoint_path,
        read_store=read_store,
        sample_distribution=sample_distribution,
        cell_atlas=cell_atlas,
        output_dir=output_dir,
        cache_dir=cache_dir,
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
    help="Path to a PacBio BAM file."
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
    sample_bam: Path, 
    output_dir: Path, 
    sample_id: str, 
    index: Optional[Path], 
    min_mapq: int, 
    workers: int, 
    chromosomes: Optional[list[str]]
) -> None:
    """Extract CpG methylation from PacBio BAM files"""
    
    
    chroms_arg = chromosomes if chromosomes else None

    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = process_bam(
        bam_path=sample_bam,
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

@main.command()
@click.option(
    "--config", "config_path", required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to YAML configuration file for the full pipeline.",
)
def run(config_path: Path):
    """Run the full ROCIT pipeline: extract methylation, label reads, train, and predict."""
    cfg = load_config(RunConfig, config_path)
    validate_run_config(cfg)

    # Resolve external input paths
    bam_path = resolve_file(cfg.bam)
    copy_number_path = resolve_file(cfg.copy_number)
    variants_path = resolve_file(cfg.variants)
    haplotags_path = resolve_file(cfg.haplotags)
    haploblocks_path = resolve_file(cfg.haploblocks)
    snv_clusters_path = resolve_file(cfg.snv_clusters)
    cell_atlas_path = resolve_file(cfg.cell_atlas)
    output_dir = resolve_dir(cfg.output_dir, must_exist=False)
    cache_dir = resolve_dir(cfg.cache_dir, must_exist=False)

    bam_index_path = resolve_file(cfg.bam_index) if cfg.bam_index else None

    snv_cluster_assignments: pl.DataFrame | None = None
    if cfg.snv_cluster_assignments is not None:
        snv_cluster_assignments = read_dataframe(resolve_file(cfg.snv_cluster_assignments))

    # Create output subdirectories
    methylation_dir = output_dir / "methylation"
    distribution_dir = output_dir / "distribution"
    preprocess_dir = output_dir / "preprocessing"
    training_dir = output_dir / "training"
    prediction_dir = output_dir / "predictions"
    for d in [methylation_dir, distribution_dir, preprocess_dir, training_dir, prediction_dir, cache_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract BAM methylation
    click.echo(f"[1/5] Extracting BAM methylation for '{cfg.sample_id}' ...")
    process_bam(
        bam_path=bam_path,
        output_dir=methylation_dir,
        sample_id=cfg.sample_id,
        chromosomes=cfg.chromosomes,
        index_path=bam_index_path,
        min_mapq=cfg.min_mapq,
        n_workers=cfg.workers,
    )

    # Step 2: Compute CpG distribution
    click.echo("[2/5] Computing CpG distribution ...")
    get_aggregate_methylation_distribution_from_dir(methylation_dir, distribution_dir, cfg.sample_id)
    sample_distribution_path = distribution_dir / f"{cfg.sample_id}_methylation_distribution.parquet"

    # Step 3: Label reads
    click.echo("[3/5] Labelling reads ...")
    somatic_data = tumor_data_labeller.ROCITSomaticData(
        sample_id=cfg.sample_id,
        sample_bam_path=bam_path,
        sample_methylation_dir=methylation_dir,
        sample_copy_number=read_dataframe(copy_number_path),
        sample_variants=read_dataframe(variants_path),
        sample_haplotags=read_dataframe(haplotags_path),
        sample_haploblocks=read_dataframe(haploblocks_path),
        snv_clusters=read_dataframe(snv_clusters_path),
        snv_cluster_assignments=snv_cluster_assignments,
    )
    labelled_reads = tumor_data_labeller.make_read_labels(somatic_data)
    labelled_methylation_data = tumor_data_labeller.get_labelled_methylation_data(
        methylation_dir, labelled_reads
    )
    labelled_reads.write_parquet(preprocess_dir / "labelled_reads.parquet")
    labelled_methylation_data.write_parquet(preprocess_dir / "labelled_methylation_data.parquet")

    # Step 4: Train
    click.echo("[4/5] Training model ...")
    sample_distribution = pl.read_parquet(sample_distribution_path)
    cell_atlas = read_dataframe(cell_atlas_path)
    train_result = training_wrapper(
        sample_id=cfg.sample_id,
        labelled_data=labelled_methylation_data,
        sample_distribution=sample_distribution,
        cell_atlas=cell_atlas,
        val_chromosomes=list(set(cfg.val_chromosomes)),
        test_chromosomes=list(set(cfg.test_chromosomes)),
        output_dir=training_dir,
        cache_dir=cache_dir,
    )

    # Step 5: Predict
    click.echo("[5/5] Running predictions ...")
    read_store = [
        pl.scan_parquet(f).filter(~pl.col("supplementary_alignment"))
        for f in methylation_dir.glob("*_cpg_methylation.parquet")
    ]
    predict_wrapper(
        sample_id=cfg.sample_id,
        best_checkpoint_path=train_result.best_checkpoint_path,
        read_store=read_store,
        sample_distribution=sample_distribution,
        cell_atlas=cell_atlas,
        output_dir=prediction_dir,
        cache_dir=cache_dir,
    )

    click.echo(f"Pipeline complete. Results in {prediction_dir}")

if __name__ == "__main__":
    main()