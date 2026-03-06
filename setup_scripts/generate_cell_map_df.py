import pyBigWig
import polars as pl
import re
import sys
import argparse
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm

CHROMOSOMES = [f"chr{x}" for x in range(1, 23)] + ["chrX"]


def load_bigwig_paths(data_dir: Path) -> dict[str, list[Path]]:
    """Discover bigwig files and group them by cell type."""
    bigwig_paths: dict[str, list[Path]] = {}
    for filepath in data_dir.glob("*.hg38.bigwig"):
        match = re.search(r"_(.*?)-(Z0|11)", filepath.name)
        if match:
            cell_type = match.group(1)
        else:
            raise ValueError(f"No cell type found for {filepath}")
        bigwig_paths.setdefault(cell_type, []).append(filepath)
    return bigwig_paths


def bigwig_to_parquets(
    bigwig_path: Path, out_dir: Path, eps: float = 1e-9
) -> list[Path]:
    """Convert a single bigwig to one parquet per chromosome.

    Each chromosome's intervals are extracted, filtered, written to disk, and
    immediately freed.  This avoids ever holding more than one chromosome's
    worth of data in memory.

    Returns the list of parquet files written.
    """
    bw = pyBigWig.open(str(bigwig_path))
    written: list[Path] = []

    for chromosome in CHROMOSOMES:
        if chromosome not in bw.chroms():
            continue
        intervals = bw.intervals(chromosome)
        if not intervals:
            continue

        # Build column arrays directly to avoid an intermediate list of tuples
        positions = []
        values = []
        for start, _end, value in intervals:
            if value >= -eps:
                positions.append(start)
                values.append(value)
        del intervals

        if not positions:
            continue

        df = pl.DataFrame(
            {
                "chromosome": [chromosome] * len(positions),
                "position": positions,
                "average_methylation": values,
            },
            schema={
                "chromosome": pl.Utf8,
                "position": pl.UInt32,
                "average_methylation": pl.Float32,
            },
        )
        del positions, values

        out_path = out_dir / f"{bigwig_path.stem}_{chromosome}.parquet"
        df.write_parquet(out_path)
        written.append(out_path)
        del df

    bw.close()
    return written


def aggregate_cell_type(parquet_paths: list[Path], out_path: Path) -> None:
    """Lazily scan replicate parquets, average methylation, and sink to parquet."""
    lf = pl.scan_parquet(parquet_paths)
    lf = lf.group_by(["chromosome", "position"]).agg(
        pl.col("average_methylation").mean()
    )
    lf.sink_parquet(out_path)


def join_per_chromosome_chunked(
    celltype_parquets: dict[str, Path],
    output_path: Path,
    tmp_dir: Path,
) -> None:
    """Join cell-type parquets one chromosome at a time, writing per-chrom
    parquets then concatenating via lazy scan at the end.

    Avoids the repeated read-append-write pattern.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chrom_dir = tmp_dir / "per_chrom"
    chrom_dir.mkdir(exist_ok=True)
    chrom_files: list[Path] = []

    for chrom in tqdm(CHROMOSOMES, desc="Joining chromosomes"):
        frames: list[pl.DataFrame] = []
        for cell_type, pq_path in celltype_parquets.items():
            col_name = f"average_methylation_{cell_type.lower()}"
            df = (
                pl.scan_parquet(pq_path)
                .filter(pl.col("chromosome") == chrom)
                .rename({"average_methylation": col_name})
                .drop("chromosome")
                .collect()
            )
            if df.height > 0:
                frames.append(df)

        if not frames:
            continue

        joined = frames[0]
        for right in frames[1:]:
            joined = joined.join(right, on="position", how="full", coalesce=True)
        del frames

        joined = joined.with_columns(pl.lit(chrom).alias("chromosome"))
        joined = joined.sort("position")

        chrom_path = chrom_dir / f"{chrom}.parquet"
        joined.write_parquet(chrom_path)
        chrom_files.append(chrom_path)
        del joined

    # Concatenate all chromosome parquets lazily and sink
    if chrom_files:
        pl.scan_parquet(chrom_files).sink_parquet(output_path)


def build_atlas(
    all_bigwig_paths: dict[str, list[Path]],
    output_path: Path,
) -> None:
    """Build the combined atlas with minimal memory footprint.

    Pipeline:
      1. Each bigwig -> per-chromosome parquets (one chrom in memory at a time).
      2. Per cell type: lazily aggregate replicate parquets -> cell type parquet.
      3. Per chromosome: read relevant slices from cell type parquets, join, write.
      4. Lazily concatenate chromosome parquets into final output.
    """
    with tempfile.TemporaryDirectory(prefix="methylation_atlas_") as tmp_str:
        tmp_dir = Path(tmp_str)
        bigwig_tmp = tmp_dir / "bigwig"
        celltype_tmp = tmp_dir / "celltype"
        bigwig_tmp.mkdir()
        celltype_tmp.mkdir()

        # -- Stage 1: bigwig -> per-chromosome parquets (sequential) --------
        cell_type_replicate_pqs: dict[str, list[Path]] = {}
        total_files = sum(len(v) for v in all_bigwig_paths.values())

        with tqdm(total=total_files, desc="Converting bigwig files") as pbar:
            for cell_type, paths in all_bigwig_paths.items():
                for bw_path in paths:
                    pq_dir = bigwig_tmp / bw_path.stem
                    pq_dir.mkdir()
                    pq_paths = bigwig_to_parquets(bw_path, pq_dir)
                    cell_type_replicate_pqs.setdefault(cell_type, []).extend(pq_paths)
                    pbar.update(1)

        # -- Stage 2: aggregate replicates per cell type (lazy sink) -------
        celltype_parquets: dict[str, Path] = {}
        for cell_type, pq_paths in tqdm(
            cell_type_replicate_pqs.items(), desc="Aggregating cell types"
        ):
            out = celltype_tmp / f"{cell_type.lower()}.parquet"
            aggregate_cell_type(pq_paths, out)
            celltype_parquets[cell_type] = out

        # Free replicate parquets
        shutil.rmtree(bigwig_tmp)

        # -- Stage 3 & 4: per-chromosome join -> final parquet -------------
        join_per_chromosome_chunked(celltype_parquets, output_path, tmp_dir)

    # TemporaryDirectory cleaned up here


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a combined cell-type average methylation atlas from GSE186458 "
            "bigwig files and write the result as a Parquet file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-d", "--data-dir",
        type=Path,
        required=True,
        help=(
            "Path to a directory containing the extracted *.hg38.bigwig files "
            "(e.g. from the GSE186458 tar archive)."
        ),
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output path for the resulting Parquet file.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    all_bigwig_paths = load_bigwig_paths(data_dir)
    if not all_bigwig_paths:
        print(f"Error: no *.hg38.bigwig files found in {data_dir}.", file=sys.stderr)
        sys.exit(1)

    total_files = sum(len(v) for v in all_bigwig_paths.values())
    print(
        f"Found {total_files} bigwig files across {len(all_bigwig_paths)} cell types."
    )

    build_atlas(all_bigwig_paths, args.output)
    print(f"Atlas written to {args.output}.")


if __name__ == "__main__":
    main()
