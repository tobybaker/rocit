import array

import numpy as np
import polars as pl
import pysam
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from rocit.constants import HUMAN_CHROMOSOMES
from rocit.preprocessing.qc import QCThresholds

# PacBio encodes 5mC at CpG contexts; forward strand maps to C at position 0,
# reverse strand to C at position 1 (the G on the reference becomes C on the read)
FORWARD_CPG_MODIFICATION_INDEX = ('C', 0, 'm')
REVERSE_CPG_MODIFICATION_INDEX = ('C', 1, 'm')

# Genomic coordinates are non-negative, so -1 is a safe sentinel for missing
# reference positions (soft-clipped/inserted bases). Replaced with NULL on write.
_NULL_POSITION = -1
_STRAND_FORWARD = 0
_STRAND_REVERSE = 1

def get_cpg_modification_index(strand: str) -> tuple[str, int, str]:
    """Return the pysam modified_bases key for the given strand."""
    if strand == '+':
        return FORWARD_CPG_MODIFICATION_INDEX
    if strand == '-':
        return REVERSE_CPG_MODIFICATION_INDEX
    raise ValueError(f"Invalid strand: {strand}")


def has_valid_methylation_tags(read: pysam.AlignedSegment) -> bool:
    """
    Validate that the read has properly formatted methylation tags.
    
    PacBio reads should have methylation calls on exactly one strand
    (matching the read's alignment orientation).
    """
    if not (read.has_tag("MM") and read.has_tag("ML")):
        return False

    modified_bases = read.modified_bases
    if not modified_bases:
        return False

    has_forward = FORWARD_CPG_MODIFICATION_INDEX in modified_bases
    has_reverse = REVERSE_CPG_MODIFICATION_INDEX in modified_bases

    # Expect exactly one of forward/reverse, matching alignment orientation
    if has_forward == has_reverse:
        return False
    if has_forward and read.is_reverse:
        return False
    if has_reverse and not read.is_reverse:
        return False

    return True


def passes_qc(read: pysam.AlignedSegment) -> bool:
    """Check whether read passes basic QC filters."""
    if read.is_unmapped or read.is_secondary or read.is_duplicate or read.is_qcfail:
        return False
    if not has_valid_methylation_tags(read):
        return False
    return True


def extract_cpg_methylation(
    read: pysam.AlignedSegment,
    strand: str
) -> tuple[list[int], list[int|None], list[int]]:
    """
    Extract CpG positions and methylation probabilities from a read.
    
    Returns:
        Tuple of (read_positions, reference_positions, methylation_probabilities)
        where positions are 0-based and probabilities are in [0, 255].
    """
    read_len = read.query_length
    mod_index = get_cpg_modification_index(strand)
    methylation_info = read.modified_bases.get(mod_index, [])

    # Reverse strand methylation is called on G (pairs with C on opposite strand),
    # so we adjust to report the C position on the reference
    offset = 1 if strand == '-' else 0

    # Filter CpG sites where the dinucleotide extends beyond read boundaries
    if strand == '+':
        # Need position and position+1 to be valid (C and G)
        valid_sites = [(pos, prob) for pos, prob in methylation_info if pos < read_len - 1]
    else:
        # Need position-1 and position to be valid (reporting the C at pos-1)
        valid_sites = [(pos, prob) for pos, prob in methylation_info if pos > 0]

    if not valid_sites:
        return [], [], []

    reference_positions = read.get_reference_positions(full_length=True)

    read_positions = []
    ref_positions = []
    probabilities = []

    for read_pos, prob in valid_sites:
        ref_pos = reference_positions[read_pos]
        
        read_positions.append(read_pos - offset)
        probabilities.append(prob)
        if ref_pos is not None:
            ref_positions.append(ref_pos - offset)
        else:
            ref_positions.append(None)

    return read_positions, ref_positions, probabilities


def process_chromosome(
    bam_path: str | Path,
    chromosome: str,
    output_dir: str | Path,
    sample_id:str,
    index_path: Optional[str | Path] = None,
    min_mapq: int = 0
) -> Path | None:
    """
    Process a single chromosome and write methylation data to parquet.
    
    Args:
        bam_path: Path to the BAM file.
        chromosome: Chromosome name to process.
        output_dir: Directory for output parquet files.
        sample_id: Sample ID used for file naming
        index_path: Optional path to BAM index file.
        min_mapq: Minimum mapping quality threshold.
    
    Returns:
        Path to the output parquet file.
    """
    bam_path = Path(bam_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_id}_{chromosome}_cpg_methylation.parquet"

    # array.array packs raw machine ints, dropping the ~28-byte per-element
    # overhead Python list[int] carries. Dtype codes match the final cast.
    read_counts = array.array('I')           # uint32
    supplementary_flags = array.array('B')   # uint8 → bool
    positions = array.array('i')             # int32 with -1 sentinel for None
    read_positions = array.array('i')        # int32
    methylation_probs = array.array('B')     # uint8 (0-255)
    strand_ids = array.array('B')            # uint8 (0=+, 1=-)
    # str refs are already cheap (8B/row, all pointing to the same interned
    # query_name per read); polars dedupes on the Categorical cast below.
    read_indexes: list[str] = []

    index_filename = str(index_path) if index_path else None
    alignment_counter = 0

    with pysam.AlignmentFile(str(bam_path), "rb", index_filename=index_filename) as bamfile:
        for read in bamfile.fetch(chromosome):
            if not passes_qc(read):
                continue
            if read.mapping_quality < min_mapq:
                continue

            is_reverse = read.is_reverse
            strand = "-" if is_reverse else "+"
            read_pos, ref_pos, probs = extract_cpg_methylation(read, strand)

            n_sites = len(probs)
            if n_sites == 0:
                continue

            strand_id = _STRAND_REVERSE if is_reverse else _STRAND_FORWARD
            supp_flag = 1 if read.is_supplementary else 0

            read_counts.extend([alignment_counter] * n_sites)
            supplementary_flags.extend([supp_flag] * n_sites)
            strand_ids.extend([strand_id] * n_sites)
            read_positions.extend(read_pos)
            methylation_probs.extend(probs)
            positions.extend(_NULL_POSITION if p is None else p for p in ref_pos)
            read_indexes.extend([read.query_name] * n_sites)

            alignment_counter += 1

    if len(methylation_probs) == 0:
        return None

    # np.frombuffer wraps each array.array zero-copy; the buffers stay alive in
    # this scope while polars copies them into its arrow-backed columns.
    df = pl.DataFrame({
        "read_count": np.frombuffer(read_counts, dtype=np.uint32),
        "read_index": read_indexes,
        "supplementary_alignment": np.frombuffer(supplementary_flags, dtype=np.uint8).astype(bool),
        "position": np.frombuffer(positions, dtype=np.int32),
        "read_position": np.frombuffer(read_positions, dtype=np.int32),
        "methylation": np.frombuffer(methylation_probs, dtype=np.uint8),
        "_strand_id": np.frombuffer(strand_ids, dtype=np.uint8),
    }).with_columns([
        pl.lit(chromosome).cast(pl.Categorical).alias("chromosome"),
        pl.col("read_index").cast(pl.Categorical),
        pl.when(pl.col("_strand_id") == _STRAND_FORWARD)
            .then(pl.lit("+"))
            .otherwise(pl.lit("-"))
            .cast(pl.Categorical)
            .alias("strand"),
        pl.when(pl.col("position") == _NULL_POSITION)
            .then(None)
            .otherwise(pl.col("position"))
            .alias("position"),
    ]).select([
        "read_count",
        "read_index",
        "supplementary_alignment",
        "chromosome",
        "position",
        "read_position",
        "methylation",
        "strand",
    ])

    df.write_parquet(output_path)

    return output_path


def process_bam(
    bam_path: str | Path,
    output_dir: str | Path,
    sample_id:str,
    chromosomes: Optional[list[str]] = None,
    index_path: Optional[str | Path] = None,
    qc: QCThresholds = QCThresholds(),
    n_workers: int = 1
) -> list[Path]:
    """
    Process all specified chromosomes from a BAM file.

    Args:
        bam_path: Path to the BAM file.
        output_dir: Directory for output parquet files.
        sample_id: Sample ID used for file naming
        chromosomes: List of chromosomes to process. Defaults to HG38_STANDARD_CHROMOSOMES.
        index_path: Optional path to BAM index file.
        qc: QC thresholds; ``qc.min_mapq`` sets the minimum mapping quality.
        n_workers: Number of parallel workers. Use 1 for sequential processing.

    Returns:
        List of paths to output parquet files.
    """
    min_mapq = qc.min_mapq
    if chromosomes is None:
        chromosomes = HUMAN_CHROMOSOMES
    bam_path = Path(bam_path)
    output_dir = Path(output_dir)

    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)

    # Validate BAM has required chromosomes
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        available_chroms = set(bam.references)
    
    missing = set(chromosomes) - available_chroms
    if missing:
        print(f"Warning: chromosomes not found in BAM: {sorted(missing)}")
        chromosomes = [c for c in chromosomes if c in available_chroms]

    if n_workers == 1:
        results = [
            process_chromosome(bam_path, chrom, output_dir,sample_id, index_path, min_mapq)
            for chrom in chromosomes
        ]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    process_chromosome, bam_path, chrom, output_dir,sample_id, index_path, min_mapq
                )
                for chrom in chromosomes
            ]
            results = [f.result() for f in futures]

    return [p for p in results if p is not None]

