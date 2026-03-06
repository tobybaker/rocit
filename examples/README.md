# HG008 Example

A working example using the reads sampled from the HG008 cancer cell line. This example demonstrates how to run ROCIT on real data, including data preparation, training, and prediction steps.

## Download

Download the example data and cell atlas from Zenodo:

```bash
# Download example data
wget <ZENODO_LINK>/HG008_example_data.tar.gz

# Download cell atlas
wget <ZENODO_LINK>/cell_map_reference_atlas.parquet

# Extract example data
tar -xzf HG008_example_data.tar.gz
```

## Data

The `HG008_example_data/` directory should contain:

| File | Description |
|------|-------------|
| `HG008.Revio.hifi_reads.subsampled.bam` | Subsampled PacBio HiFi BAM |
| `HG008.Revio.hifi_reads.subsampled.bam.bai` | BAM index |
| `HG008_sample_cn.tsv` | Copy number segments |
| `HG008_variants.tsv` | Somatic variants |
| `HG008_haplotags.tsv` | Read haplotype assignments |
| `HG008_haploblocks.tsv` | Phased haplotype blocks |
| `HG008_clusters.tsv` | SNV clusters |
| `HG008_snv_cluster_assignments.tsv` | SNV cluster assignments |

See the main [README](../README.md#cell-atlas-requirement) for details on file formats and requirements.

## Running Step-by-Step

```bash
# 1. Extract methylation from BAM 
rocit extract-bam-methylation --sample-id HG008 --sample-bam HG008_example_data/HG008.Revio.hifi_reads.subsampled.bam --output-dir HG008_cpg_methylation/ --chromosomes "chr19 chr21 chr22"

# 2. Extract per-CpG methylation distribution
rocit extract-cpg-distribution --sample-id HG008 --methylation-dir HG008_cpg_methylation/ --output-dir HG008_methylation_distribution/

# 3. Label reads for training
rocit preprocess --config configs/HG008_preprocess_config.yaml

# 4. Train
rocit train --config configs/HG008_train_config.yaml

# 5. Predict
rocit predict --config configs/HG008_predict_config.yaml
```

## Running the Full Pipeline

```bash
rocit run --config configs/HG008_run_config.yaml
```

This runs all steps end-to-end. See the main [README](../README.md#rocit-run) for output details.
