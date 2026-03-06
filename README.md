# ROCIT

<div align="center">
<img src="https://res.cloudinary.com/dhco9jghq/image/upload/v1771031245/rocitlogo_e2huhi.png" alt="ROCIT Logo" width="200"/>
</div>

**ROCIT** (Read Origin Classifier In Tumors) is a deep learning tool that classifies individual sequencing reads from long-read bulk tumor sequencing as tumor-derived or normal-derived. By leveraging CpG methylation patterns, ROCIT enables read-level resolution of tumor heterogeneity from PacBio sequencing data.

ROCIT currently supports training and prediction on PacBio HiFi Tumor BAMs with CpG methylation probabilities produced by [Jasmine](https://github.com/PacificBiosciences/jasmine). Oxford Nanopore support is planned for future releases.


### How It Works

ROCIT uses a multi-step approach:

1. **Data Preprocessing**: Extracts CpG methylation from PacBio BAM files and labels the origin of a subset of reads based on somatic variants (SNVs) and loss of heterozygosity (LOH) events
2. **Input Features**: Combines read-level methylation patterns with cell-type reference atlases and bulk sample methylation distributions
3. **Model Training**: Trains a transformer-based neural network to classify the labelled read subset using chromosomal cross-validation
4. **Prediction**: Applies the trained model to classify all reads in the sample

## Installation

### Via pip (Recommended)

```bash
pip install rocit
```

### From Source

```bash
git clone https://github.com/tobybaker/rocit.git
cd rocit
pip install -e .
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.9.1
- PyTorch Lightning ≥ 2.6.0
- Polars ≥ 1.36.1
- pysam == 0.22.1
- Additional dependencies listed in [pyproject.toml](pyproject.toml)

## Quick Start

### Cell Atlas Requirement

ROCIT requires a reference cell-type methylation atlas derived from whole-genome bisulfite sequencing data (GSE186458). Download the pre-computed atlas:

```bash
mkdir -p reference
wget https://zenodo.org/records/18859554/files/cell_map_reference_atlas.parquet -O reference/cell_map_reference_atlas.parquet
```

Alternatively, you can [generate the atlas from source](#generating-the-cell-atlas-from-source) using the provided scripts.

> **Citation:** Loyfer, N., et al. (2023). [A DNA methylation atlas of normal human cell types](https://doi.org/10.1038/s41586-022-05580-6). *Nature*.

### Running ROCIT

ROCIT provides a complete end-to-end pipeline via the `rocit run` command:

```bash
rocit run --config config.yaml
```

For more control, you can run individual steps:

```bash
# 1. Extract methylation from BAM
rocit extract-bam-methylation --sample-id SAMPLE_ID \
    --sample-bam aligned.bam \
    --output-dir methylation/

# 2. Compute methylation distribution
rocit extract-cpg-distribution --sample-id SAMPLE_ID \
    --methylation-dir methylation/ \
    --output-dir distribution/

# 3. Label reads for training
rocit preprocess --config preprocess_config.yaml

# 4. Train the model
rocit train --config train_config.yaml

# 5. Run predictions
rocit predict --config predict_config.yaml
```


## Example Data

The [`examples/`](examples/) directory contains a worked example using the HG008 cancer cell line, including sample config files and instructions for downloading the example dataset from Zenodo. This is the best starting point if you want to see the expected input data formats and run ROCIT end-to-end on real data.

## Configuration Files

ROCIT uses YAML configuration files for reproducibility and ease of use. Below are templates for each command. Tabular input files (variants, copy number, haplotags, etc.) can be provided in any [supported format](#data-format-specifications) — configuration examples use `.parquet` but `.csv`, `.tsv`, `.arrow`, and `.ndjson` are equally valid.

### Training Configuration (`train_config.yaml`)

```yaml
sample_id: "SAMPLE_ID"
labelled_data: "preprocessing/labelled_methylation_data.parquet"
sample_distribution: "distribution/SAMPLE_ID_methylation_distribution.parquet"
cell_atlas: "reference/cell_atlas.parquet"
val_chromosomes: ["chr20", "chr21"]
test_chromosomes: ["chr22"]
output_dir: "training/"
cache_dir: "/scratch/"
```

**Required fields:**
- `sample_id`: Unique identifier for the sample
- `labelled_data`: Path to labelled methylation data from preprocessing
- `sample_distribution`: Path to sample methylation distribution
- `cell_atlas`: Path to cell-type methylation reference
- `val_chromosomes`: Chromosomes reserved for validation (must not overlap with test)
- `test_chromosomes`: Chromosomes reserved for testing (must not overlap with validation)
- `output_dir`: Directory for training outputs
- `cache_dir`: Cache directory for dataset processing (default: `/scratch`)

### Prediction Configuration (`predict_config.yaml`)

```yaml
sample_id: "SAMPLE_ID"
best_checkpoint_path: "training/SAMPLE_ID/version_0/checkpoints/best-checkpoint.ckpt"
sample_distribution: "distribution/SAMPLE_ID_methylation_distribution.parquet"
cell_atlas: "reference/cell_atlas.parquet"
read_store_dir: "methylation/"  # OR use read_store for single file
output_dir: "predictions/"
cache_dir: "/scratch/"
```

**Required fields:**
- `sample_id`: Unique identifier for the sample
- `best_checkpoint_path`: Path to trained model checkpoint
- `sample_distribution`: Path to sample methylation distribution
- `cell_atlas`: Path to cell-type methylation reference
- `read_store` OR `read_store_dir`: Single file or directory of methylation parquet files
- `output_dir`: Directory for prediction outputs
- `cache_dir`: Cache directory (default: `/scratch`)

### Preprocessing Configuration (`preprocess_config.yaml`)

```yaml
sample_id: "SAMPLE_ID"
bam: "data/aligned.bam"
methylation_dir: "methylation/"
copy_number: "data/copy_number_segments.parquet"
variants: "data/somatic_variants.parquet"
haplotags: "data/haplotags.parquet"
haploblocks: "data/haploblocks.parquet"
snv_clusters: "data/snv_clusters.parquet"
snv_cluster_assignments: "data/snv_cluster_assignments.parquet"
output_dir: "preprocessing/"
```

**Required fields:**
- `sample_id`: Unique identifier for the sample
- `bam`: Path to aligned BAM file with methylation tags
- `methylation_dir`: Directory containing extracted methylation data
- `copy_number`: Path to copy number segments file
- `variants`: Path to somatic variants file
- `haplotags`: Path to read haplotype assignments
- `haploblocks`: Path to phased haplotype blocks
- `snv_clusters`: Path to SNV cluster assignments
- `output_dir`: Directory for preprocessing outputs

**Optional fields:**
- `snv_cluster_assignments`: Path to SNV cluster assignments (if not provided, will be inferred)

### Full Pipeline Configuration (`run_config.yaml`)

```yaml
sample_id: "SAMPLE_ID"
bam: "data/aligned.bam"
bam_index: "data/aligned.bam.bai"
copy_number: "data/copy_number_segments.parquet"
variants: "data/somatic_variants.parquet"
haplotags: "data/haplotags.parquet"
haploblocks: "data/haploblocks.parquet"
snv_clusters: "data/snv_clusters.parquet"
snv_cluster_assignments: "data/snv_cluster_assignments.parquet"
cell_atlas: "reference/cell_atlas.parquet"
val_chromosomes: ["chr20", "chr21"]
test_chromosomes: ["chr22"]
min_mapq: 0
workers: 8
output_dir: "output/"
cache_dir: "/scratch/"
```

**Required fields:**
- `sample_id`: Unique identifier for the sample
- `bam`: Path to aligned BAM file with methylation tags
- `copy_number`: Path to copy number segments file
- `variants`: Path to somatic variants file
- `haplotags`: Path to read haplotype assignments
- `haploblocks`: Path to phased haplotype blocks
- `snv_clusters`: Path to SNV cluster assignments
- `cell_atlas`: Path to cell-type methylation reference
- `val_chromosomes`: Chromosomes reserved for validation (must not overlap with test)
- `test_chromosomes`: Chromosomes reserved for testing (must not overlap with validation)
- `output_dir`: Directory for all pipeline outputs
- `cache_dir`: Cache directory for dataset processing

**Optional fields:**
- `bam_index`: BAM index file (auto-detected if not provided)
- `snv_cluster_assignments`: Path to SNV cluster assignments (if not provided, will be inferred)
- `chromosomes`: Specific chromosomes to process (defaults to chr1-chrY)
- `min_mapq`: Minimum mapping quality for reads (default: 0)
- `workers`: Number of parallel workers for BAM processing (default: 1)

## Command Reference

### `rocit run`

Run the complete ROCIT pipeline from BAM to predictions.

```bash
rocit run --config run_config.yaml
```

**Pipeline steps:**
1. Extract BAM methylation
2. Compute CpG distribution
3. Label reads using somatic variants
4. Train classification model
5. Generate predictions

**Outputs:**
- `output/methylation/`: Per-chromosome methylation data
- `output/distribution/`: Sample methylation distribution
- `output/preprocessing/`: Labelled reads and methylation data
- `output/training/`: Model checkpoints and training logs
- `output/predictions/`: Final tumor/normal predictions

### `rocit train`

Train a ROCIT classification model.

```bash
rocit train --config train_config.yaml
```

**Outputs:**
- `{output_dir}/{sample_id}/version_X/checkpoints/best-checkpoint.ckpt`: Best model
- `{output_dir}/{sample_id}/version_X/metrics.csv`: Training metrics (loss, AUROC, etc.)

**Training parameters** (modifiable in code via `TrainingParams` in the python API):
- Model architecture: 384-dim, 6 heads, 3 layers
- Max epochs: 100 with early stopping (patience=10)
- Learning rate: 1e-4 with warmup
- Batch size: 256
- Early Stopping Metric: Validation AUROC

### `rocit predict`

Generate predictions using a trained model.

```bash
rocit predict --config predict_config.yaml
```

**Outputs:**
- `{output_dir}/{sample_id}_tumor_origin_predictions.parquet`: Read-level predictions with columns:
  - `read_index`: Unique read identifier
  - `chromosome`: Chromosome name
  - `tumor_probability`: Predicted probability of tumor origin (0-1)

### `rocit preprocess`

Label reads for training using somatic variant information.

```bash
rocit preprocess --config preprocess_config.yaml
```

**Outputs:**
- `{output_dir}/labelled_reads.parquet`: Read labels (tumor/normal)
- `{output_dir}/labelled_methylation_data.parquet`: Methylation data with labels


### `rocit extract-bam-methylation`

Extract CpG methylation from PacBio BAM files.

```bash
rocit extract-bam-methylation \
    --sample-id SAMPLE_ID \
    --sample-bam aligned.bam \
    --output-dir methylation/ \
    --workers 8 \
    --min-mapq 0 \
    --chromosomes "chr1 chr2 chr3"
```

**Options:**
- `--sample-id`: Sample identifier for output naming
- `--sample-bam`: Input BAM file with MM/ML tags
- `--output-dir`: Output directory
- `--index`: BAM index file (optional, auto-detected)
- `--min-mapq`: Minimum mapping quality (default: 0)
- `--workers`: Number of parallel workers (default: 1)
- `--chromosomes`: Space-separated chromosomes to process (default: chr1-chrY)

**Outputs:**
- `{output_dir}/{chromosome}_cpg_methylation_data.parquet` for each chromosome

### `rocit extract-cpg-distribution`

Aggregate methylation distribution from extracted data.

```bash
rocit extract-cpg-distribution \
    --sample-id SAMPLE_ID \
    --methylation-dir methylation/ \
    --output-dir distribution/
```

**Outputs:**
- `{output_dir}/{sample_id}_methylation_distribution.parquet`: 
An aggregated distribution of methylation values across the sample, used for model context.

## Output Files

### Prediction Output

The primary output from ROCIT is a parquet file with read-level predictions:

```python
import polars as pl

predictions = pl.read_parquet("predictions/SAMPLE_ID_tumor_origin_predictions.parquet")
print(predictions.head())

# Example output:
# ┌────────────┬────────────┬───────────────────┐
# │ read_index │ chromosome │ tumor_probability │
# ├────────────┼────────────┼───────────────────┤
# │ 1001       │ chr1       │ 0.87              │
# │ 1002       │ chr1       │ 0.12              │
# │ 1003       │ chr1       │ 0.94              │
# └────────────┴────────────┴───────────────────┘
```

### Training Metrics

Training progress is logged to CSV:

```python
metrics = pl.read_csv("training/SAMPLE_ID/version_0/metrics.csv")
# Contains: epoch, train_loss, train_auroc, val_loss, val_auroc, etc.
```

## Model Architecture

ROCIT uses a transformer-based architecture designed for long-read methylation data:

- Input: CpG methylation patterns, cell atlas features, sample distribution features
## Python API

ROCIT can also be used programmatically:

```python
import rocit
from pathlib import Path

# Training
train_result = rocit.train(
    sample_id="SAMPLE_ID",
    labelled_data=labelled_df,
    sample_distribution=distribution_df,
    cell_atlas=atlas_df,
    val_chromosomes=["chr20", "chr21"],
    test_chromosomes=["chr22"],
    output_dir=Path("training/"),
    cache_dir=Path("/scratch/")
)

# Prediction
predictions = rocit.predict(
    sample_id="SAMPLE_ID",
    best_checkpoint_path=Path("training/best-checkpoint.ckpt"),
    read_store=[methylation_lazy_df],  # List of polars DataFrames or LazyFrames
    sample_distribution=distribution_df,
    cell_atlas=atlas_df,
    output_dir=Path("predictions/"),
    cache_dir=Path("/scratch/")
)
```

## Generating the Cell Atlas from Source

If you prefer to build the cell-type methylation atlas yourself rather than using the pre-computed version, two scripts are provided. This process downloads and processes whole-genome bisulfite sequencing data from GEO accession **GSE186458**, which contains methylation profiles across diverse normal human cell types.

### Requirements

```bash
pip install pyBigWig polars tqdm requests
```

### Usage

**Step 1: Download the bigwig files**

This will download ~60 GB of raw data from GEO NCBI:

```bash
python setup_scripts/download_bigwig_files.py \
    --outdir /path/to/download_directory/
```

**Options:**
- `--outdir`: Output directory for downloaded files (default: `GSE186458_hg38_bigwigs`)
- `--max-concurrent`: Number of concurrent downloads (default: 4; do not exceed ~5)
- `--dry-run`: Print URLs without downloading

**Step 2: Build the atlas from downloaded files**

```bash
python setup_scripts/generate_cell_map_df.py \
    --data-dir /path/to/download_directory/ \
    --output reference/cell_atlas.parquet
```

### What the Scripts Do

1. **Download** (`download_bigwig_files.py`): Fetches all `*.hg38.bigwig` files from the GSE186458 GEO series via NCBI FTP, with resume support and retry logic
2. **Process** (`generate_cell_map_df.py`): For each cell type, aggregates methylation values across biological replicates
3. **Combine**: Joins all cell types into a single reference atlas
4. **Output**: Saves a Parquet file with columns:
   - `chromosome`: chr1-chr22, chrX
   - `position`: CpG genomic position
   - `average_methylation_{cell_type}`: Mean methylation value (0-1) for each cell type

The resulting atlas enables ROCIT to contextualize read-level methylation patterns using cell-type-specific reference signatures.

### Dataset Information

**GSE186458** contains whole-genome bisulfite sequencing (WGBS) data from normal human tissues and cell types. Each cell type typically has multiple biological replicates, which the script averages to produce robust methylation estimates.

> **Citation:** Loyfer, N., et al. (2023). [A DNA methylation atlas of normal human cell types](https://doi.org/10.1038/s41586-022-05580-6). *Nature*.

---

## Data Format Specifications

ROCIT accepts the following tabular file formats for all non-BAM inputs (copy number, variants, haplotags, haploblocks, SNV clusters, cell atlas, etc.):

| Format | Extensions |
|--------|-----------|
| Parquet | `.parquet`, `.pqt` |
| CSV | `.csv` |
| TSV | `.tsv` |
| Arrow IPC / Feather | `.arrow`, `.feather`, `.ipc` |
| Newline-delimited JSON | `.ndjson` |

All outputs from ROCIT are written as Parquet.

### Copy Number Segments

Required columns:
- `chromosome`: Chromosome name (e.g., "chr1")
- `start`: Segment start position
- `end`: Segment end position
- `minor_cn`: Minor allele copy number
- `major_cn`: Major allele copy number
- `total_cn`: Total copy number
- `purity`: Tumor purity estimate
- `normal_total_cn`: Normal total copy number (typically 2 except for chrX/chrY in XY subjects)

### Somatic Variants

Required columns:
- `chromosome`: Chromosome name
- `position`: Variant position
- `ref`: Reference allele
- `alt`: Alternate allele
- Additional variant metadata as needed

### Haplotags

Required columns:
- `read_index`: Unique read identifier
- `chromosome`: Chromosome name
- `haplotag`: Haplotype assignment (1 or 2)
- `start`: Read start position
- `end`: Read end position

### Haploblocks

Required columns:
- `chromosome`: Chromosome name
- `block_start`: Block start position
- `block_end`: Block end position
- `block_size`: Size of phased block
- `haploblock_id`: Unique block identifier

### SNV Clusters

Required columns:
- `cluster_id`: Unique cluster identifier
- `cluster_ccf`: Cancer cell fraction for the cluster (0-1)
- `cluster_fraction`: Fraction of variants assigned to this cluster (0-1)

### SNV Cluster Assignments

This file is optional. If not provided, cluster assignments will be inferred using a binomial model.

Required columns:
- `chromosome`: Chromosome name
- `position`: Variant position
- `cluster_id`: Cluster identifier (must match IDs in SNV Clusters)
- `n_copies`: Number of allelic copies of the variant.


## Citation

If you use ROCIT in your research, please cite:

Baker TM, Matulionis N, Andrasz C, Gerke D, Garcia-Dutton N, Atkinson D, et al. Genome-wide classification of tumor-derived reads from bulk long-read sequencing. *bioRxiv* [Preprint]. 2026 Mar 5. DOI: [10.64898/2026.03.03.709085](https://doi.org/10.64898/2026.03.03.709085)

## License

ROCIT is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

