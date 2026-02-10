import pyBigWig
import polars as pl
import re
import tarfile
import urllib.request
import sys
import argparse
from pathlib import Path
from functools import reduce
from tqdm import tqdm

# GEO accession bulk download URL for GSE186458
GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE186458&format=file"
DOWNLOAD_SIZE_GB = 328

CHROMOSOMES = [f"chr{x}" for x in range(1, 23)] + ["chrX"]


def download_and_extract(dest_dir: Path) -> Path:
    """Download the GSE186458 tar archive from GEO and extract bigwig files.

    Parameters
    ----------
    dest_dir : Path
        Directory in which to place the downloaded tar and extracted contents.

    Returns
    -------
    Path
        The directory containing the extracted bigwig files.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "GSE186458_RAW.tar"

    if tar_path.exists():
        print(f"Tar archive already exists at {tar_path}, skipping download.")
    else:
        print(f"Downloading {DOWNLOAD_SIZE_GB} GB archive from GEO to {tar_path} ...")
        # Simple progress hook for urllib
        def _reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                print(f"\r  {downloaded / 1e9:.2f} / {total_size / 1e9:.2f} GB ({pct:.1f}%)", end="")
            else:
                print(f"\r  {downloaded / 1e9:.2f} GB downloaded", end="")

        urllib.request.urlretrieve(GEO_URL, str(tar_path), reporthook=_reporthook)
        print()  # newline after progress

    # Extract -----------------------------------------------------------
    extract_dir = dest_dir / "bigwig_files"
    if extract_dir.exists() and any(extract_dir.glob("*.bigwig")):
        print(f"Extracted files already present in {extract_dir}, skipping extraction.")
    else:
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {tar_path} into {extract_dir} ...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_dir)
        print("Extraction complete.")

    return extract_dir


def load_bigwig_paths(data_dir: Path) -> dict[str, list[Path]]:
    bigwig_paths: dict[str, list[Path]] = {}
    for filepath in data_dir.glob("*.hg38.bigwig"):
        match = re.search(r"_(.*?)-(Z0|11)", filepath.name)
        if match:
            cell_type = match.group(1)
        else:
            raise ValueError(f"No cell type found for {filepath}")
        bigwig_paths.setdefault(cell_type, []).append(filepath)
    return bigwig_paths


def load_big_wig_df(bigwig_path: Path, eps: float = 1e-9) -> pl.DataFrame:
    bw = pyBigWig.open(str(bigwig_path))
    data = []
    for chromosome in bw.chroms():
        if chromosome not in CHROMOSOMES:
            continue
        intervals = bw.intervals(chromosome)
        if intervals:
            for start, end, value in intervals:
                if value < -eps:
                    continue
                data.append((chromosome, start, value))
    bw.close()

    df = pl.DataFrame(
        data,
        schema={
            "chromosome": pl.Enum(CHROMOSOMES),
            "position": pl.UInt32,
            "average_methylation": pl.Float32,
        },
        orient="row",
    )
    return df


def get_cell_type_df(bigwig_paths: list[Path]) -> pl.DataFrame:
    all_dfs = [load_big_wig_df(p) for p in bigwig_paths]
    all_dfs = pl.concat(all_dfs)
    return all_dfs.group_by(["chromosome", "position"]).agg(
        pl.col("average_methylation").mean()
    )


def join_frames(df_store: list[pl.LazyFrame]) -> pl.LazyFrame:
    return reduce(
        lambda left, right: left.join(
            right, on=["chromosome", "position"], how="full", coalesce=True
        ),
        df_store,
    )


def get_combined_cell_type_df(all_bigwig_paths: dict[str, list[Path]]) -> pl.DataFrame:
    df_store = []
    for cell_type, cell_type_bigwig_paths in tqdm(
        all_bigwig_paths.items(), desc="Loading cell types"
    ):
        cell_type_df = get_cell_type_df(cell_type_bigwig_paths)
        cell_type_df = cell_type_df.rename(
            {"average_methylation": f"average_methylation_{cell_type.lower()}"}
        )
        df_store.append(cell_type_df.lazy())

    combined_df = join_frames(df_store).collect()
    combined_df = combined_df.sort(["chromosome", "position"])
    return combined_df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a combined cell-type average methylation atlas from GSE186458 "
            "bigwig files and write the result as a Parquet file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input: either a local path or a download flag (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-d", "--data-dir",
        type=Path,
        help=(
            "Path to a directory containing the extracted *.hg38.bigwig files "
            "(e.g. from the GSE186458 tar archive)."
        ),
    )
    input_group.add_argument(
        "--download",
        type=Path,
        metavar="DOWNLOAD_DIR",
        help=(
            f"Download the GSE186458 archive (~{DOWNLOAD_SIZE_GB} GB) from GEO into "
            "DOWNLOAD_DIR, extract it, and use the extracted bigwig files as input."
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

    # Resolve input directory ------------------------------------------------
    if args.download is not None:
        print(
            f"WARNING: This will download approximately {DOWNLOAD_SIZE_GB} GB of data "
            f"into {args.download.resolve()}."
        )
        confirmation = input("Proceed? [y/N]: ").strip().lower()
        if confirmation != "y":
            print("Aborted.")
            sys.exit(0)
        data_dir = download_and_extract(args.download)
    else:
        data_dir = args.data_dir
        if not data_dir.is_dir():
            print(f"Error: {data_dir} is not a valid directory.", file=sys.stderr)
            sys.exit(1)

    # Build atlas ------------------------------------------------------------
    all_bigwig_paths = load_bigwig_paths(data_dir)
    if not all_bigwig_paths:
        print(f"Error: no *.hg38.bigwig files found in {data_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {sum(len(v) for v in all_bigwig_paths.values())} bigwig files "
          f"across {len(all_bigwig_paths)} cell types.")

    combined_df = get_combined_cell_type_df(all_bigwig_paths)

    # Write output -----------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined_df.write_parquet(args.output)
    print(f"Atlas written to {args.output} "
          f"({combined_df.shape[0]:,} rows x {combined_df.shape[1]} columns).")


if __name__ == "__main__":
    main()