"""
Dataset Download Utility
=========================
Helper script for downloading battery cycling datasets.

The datasets used in this study are publicly available:
  - CALCE : https://calce.umd.edu/battery-data
  - HUST  : (Publicly available via academic request)
  - HNEI  : https://www.hnei.hawaii.edu/
  - CALB  : (Industrial partner dataset)
  - ISU   : Iowa State University ILCC vehicular dataset

After downloading, the raw data should be converted to the unified .pkl
format expected by BatteryCycleDataset. See README.md for details.
"""

import os
import argparse


def download_via_huggingface(repo_id, token=None, local_dir='./dataset'):
    """
    Download preprocessed battery datasets from HuggingFace Hub.

    Args:
        repo_id   : HuggingFace repository ID (e.g., 'your-org/battery-datasets').
        token     : HuggingFace access token (optional for public repos).
        local_dir : Local directory to save datasets.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[!] Missing huggingface_hub. Installing...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    print(f"[*] Downloading from HuggingFace: {repo_id}")
    os.makedirs(local_dir, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            token=token,
            resume_download=True,
            max_workers=8,
        )
        print(f"[+] Download complete! Data saved to {local_dir}")
    except Exception as e:
        print(f"[-] Download failed: {e}")
        print("\n[*] You can manually download datasets from:")
        print("  1. CALCE  : https://calce.umd.edu/battery-data")
        print("  2. HUST   : Contact authors")
        print("  3. HNEI   : https://www.hnei.hawaii.edu/")
        print("  4. ISU    : Contact authors")


def main():
    parser = argparse.ArgumentParser(description="Battery Dataset Downloader")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="HuggingFace repository ID for preprocessed datasets")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace access token")
    parser.add_argument("--output_dir", type=str, default="./dataset",
                        help="Output directory for downloaded data")
    args = parser.parse_args()

    print("=" * 50)
    print("  PG-M2TN Dataset Downloader")
    print("=" * 50)

    if args.repo_id:
        download_via_huggingface(args.repo_id, args.hf_token, args.output_dir)
    else:
        print("\n  No --repo_id specified.")
        print("  Please provide a HuggingFace repo ID or download manually.")
        print("\n  Public dataset sources:")
        print("    CALCE : https://calce.umd.edu/battery-data")
        print("    HNEI  : https://www.hnei.hawaii.edu/")
        print("\n  After downloading, convert to .pkl format (see README.md).")


if __name__ == '__main__':
    main()
