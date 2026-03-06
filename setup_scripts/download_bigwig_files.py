#!/usr/bin/env python3
"""
Download all .hg38.bigwig files from GEO accession GSE186458.

This script parses the GSE186458 series page to extract sample accessions
and titles, constructs per-sample FTP URLs for the .hg38.bigwig files,
and downloads them with resume support and retry logic.

Usage:
    python download_hg38_bigwigs.py [--outdir OUTPUT_DIR] [--max-concurrent N] [--dry-run]
"""

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    sys.exit(
        "ERROR: 'requests' is required. Install with: pip install requests"
    )


GEO_SERIES_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
GEO_SERIES_ACC = "GSE186458"
FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/samples"

# Retry configuration
MAX_RETRIES = 5
RETRY_BACKOFF = 10  # seconds; doubles each retry
CHUNK_SIZE = 1024 * 1024  # 1 MB


def fetch_sample_list(session: requests.Session) -> list[tuple[str, str]]:
    """
    Parse the GSE186458 GEO page to extract (GSM_accession, sample_title) pairs.

    Returns a list of (accession, title) tuples, e.g.:
        [("GSM5652176", "Adipocytes-Z000000T7"), ...]
    """
    params = {"acc": GEO_SERIES_ACC, "targ": "gsm", "view": "brief", "form": "text"}
    print(f"Fetching sample list from GEO for {GEO_SERIES_ACC}...")
    resp = session.get(GEO_SERIES_URL, params=params, timeout=120)
    resp.raise_for_status()

    # The text format returns blocks like:
    # ^SAMPLE = GSM5652176
    # !Sample_title = Adipocytes-Z000000T7
    samples = []
    current_acc = None
    for line in resp.text.splitlines():
        if line.startswith("^SAMPLE"):
            current_acc = line.split("=")[-1].strip()
        elif line.startswith("!Sample_title") and current_acc:
            title = line.split("=", 1)[-1].strip()
            samples.append((current_acc, title))
            current_acc = None

    if not samples:
        sys.exit(
            "ERROR: Failed to parse any samples from GEO. "
            "The page format may have changed."
        )

    print(f"  Found {len(samples)} samples.")
    return samples


def build_url(accession: str, title: str) -> str:
    """
    Construct the HTTPS FTP URL for a given sample's .hg38.bigwig file.

    GEO FTP directory structure:
        /geo/samples/GSM{prefix}nnn/GSM{id}/suppl/{filename}
    where prefix is the accession digits with the last 3 replaced by 'nnn'.
    """
    # e.g. GSM5652176 -> GSM5652nnn
    prefix = accession[:-3] + "nnn"
    filename = f"{accession}_{title}.hg38.bigwig"
    return f"{FTP_BASE}/{prefix}/{accession}/suppl/{quote(filename)}"


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    max_retries: int = MAX_RETRIES,
) -> bool:
    """
    Download a single file with resume support and exponential backoff.

    Returns True on success, False on failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            # Check existing partial download for resume
            headers = {}
            mode = "wb"
            existing_size = 0
            if dest.exists():
                existing_size = dest.stat().st_size
                headers["Range"] = f"bytes={existing_size}-"
                mode = "ab"

            resp = session.get(url, headers=headers, stream=True, timeout=300)

            # If server responds 416, file is already complete
            if resp.status_code == 416:
                print(f"  Already complete: {dest.name}")
                return True

            # 200 means server ignores Range; restart from scratch
            if resp.status_code == 200 and existing_size > 0:
                mode = "wb"
                existing_size = 0

            resp.raise_for_status()

            # Determine expected total size
            total = None
            if "Content-Length" in resp.headers:
                total = int(resp.headers["Content-Length"]) + existing_size

            # If file already matches expected size, skip
            if total and existing_size == total:
                print(f"  Already complete: {dest.name}")
                return True

            downloaded = existing_size
            with open(dest, mode) as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)

            # Verify size if known
            if total and downloaded != total:
                print(
                    f"  WARNING: Size mismatch for {dest.name} "
                    f"(got {downloaded}, expected {total}). Will retry."
                )
                continue

            return True

        except (requests.RequestException, IOError) as e:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            print(
                f"  Attempt {attempt}/{max_retries} failed for {dest.name}: {e}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    print(f"  FAILED after {max_retries} attempts: {dest.name}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download .hg38.bigwig files from GSE186458."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("GSE186458_hg38_bigwigs"),
        help="Output directory (default: GSE186458_hg38_bigwigs)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Max concurrent downloads (default: 4). "
        "Be respectful to NCBI servers; do not exceed ~5.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs without downloading.",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "GSE186458-bigwig-downloader/1.0 (research use)"}
    )

    # Step 1: Get sample list from GEO
    samples = fetch_sample_list(session)

    # Step 2: Build URLs and filenames
    tasks = []
    for acc, title in samples:
        url = build_url(acc, title)
        filename = f"{acc}_{title}.hg38.bigwig"
        tasks.append((url, filename))

    print(f"\n{len(tasks)} .hg38.bigwig files to download.\n")

    if args.dry_run:
        for url, filename in tasks:
            print(f"  {filename}")
            print(f"    {url}")
        print(f"\nDry run complete. {len(tasks)} files listed.")
        return

    # Step 3: Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Step 4: Download with concurrency
    succeeded = 0
    failed = []

    def _download(url_filename):
        url, filename = url_filename
        dest = args.outdir / filename
        print(f"Downloading: {filename}")
        ok = download_file(session, url, dest)
        return filename, ok

    with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
        futures = {pool.submit(_download, t): t for t in tasks}
        for future in as_completed(futures):
            filename, ok = future.result()
            if ok:
                succeeded += 1
            else:
                failed.append(filename)

    # Step 5: Summary
    print(f"\n{'=' * 60}")
    print(f"Download complete: {succeeded}/{len(tasks)} succeeded.")
    if failed:
        print(f"\nFailed downloads ({len(failed)}):")
        for f in sorted(failed):
            print(f"  {f}")
        print("\nRe-run the script to resume failed/partial downloads.")
    print(f"Files saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
