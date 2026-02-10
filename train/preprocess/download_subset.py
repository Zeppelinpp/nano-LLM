"""
Download subset of OpenWebText dataset (single parquet file)
Usage: uv run train/download_subset.py --output_dir ./data/openwebtext_subset
"""

import argparse
import os
import requests
from tqdm import tqdm


def download_file(url, output_path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with (
        open(output_path, "wb") as f,
        tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                size = f.write(chunk)
                pbar.update(size)


def download_openwebtext_subset(output_dir):
    """Download a single parquet file from OpenWebText (304MB)"""

    print("Downloading a single parquet file from OpenWebText (304MB)...")
    print("This is for testing purposes.\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # OpenWebText dataset URL (first parquet file)
    base_url = "https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main"
    filename = "plain_text/train-00000-of-00080.parquet"
    url = f"{base_url}/{filename}"

    output_path = os.path.join(output_dir, "train.parquet")

    print(f"URL: {url}")
    print(f"Output: {output_path}\n")

    try:
        download_file(url, output_path)
        print(f"\n✓ Downloaded successfully!")
        print(f"File size: {os.path.getsize(output_path) / (1024**2):.1f} MB")
        print(f"Saved to: {output_path}")

        # Save info file
        info_path = os.path.join(output_dir, "dataset_info.txt")
        with open(info_path, "w") as f:
            f.write(f"OpenWebText Subset (Single File)\n")
            f.write(f"==================================\n\n")
            f.write(f"Downloaded: Single parquet file\n")
            f.write(f"File: {filename}\n")
            f.write(f"Path: {output_path}\n")
            f.write(f"Size: {os.path.getsize(output_path) / (1024**2):.1f} MB\n")

        print(f"\nNext steps:")
        print(f"1. Run: uv run train/prepare_subset.py --data_dir {output_dir}")
        print(f"2. Train with: uv run train/pretrain.py --data_dir {output_dir} ...")

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("Please check your internet connection and try again.")
        if os.path.exists(output_path):
            os.remove(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download subset of OpenWebText dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/openwebtext_subset",
        help="Output directory for the dataset",
    )
    args = parser.parse_args()

    download_openwebtext_subset(args.output_dir)


if __name__ == "__main__":
    main()
