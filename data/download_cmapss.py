"""
Download NASA C-MAPSS Turbofan Engine Degradation Dataset
Source: NASA PCoE Prognostics Data Repository
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloading files."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file from URL with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_cmapss_dataset(data_dir='./'):
    """
    Download and extract NASA C-MAPSS dataset.

    Dataset contains:
    - FD001: Train trajectories: 100, Test trajectories: 100, Operating conditions: ONE (Sea Level), Fault Modes: ONE (HPC Degradation)
    - FD002: Train trajectories: 260, Test trajectories: 259, Operating conditions: SIX, Fault Modes: ONE (HPC Degradation)
    - FD003: Train trajectories: 100, Test trajectories: 100, Operating conditions: ONE (Sea Level), Fault Modes: TWO (HPC Degradation, Fan Degradation)
    - FD004: Train trajectories: 248, Test trajectories: 249, Operating conditions: SIX, Fault Modes: TWO (HPC Degradation, Fan Degradation)

    Each file contains:
    - train_FD00X.txt: Training data
    - test_FD00X.txt: Test data (RUL unknown)
    - RUL_FD00X.txt: True RUL values for test data

    Columns:
    1) Unit number
    2) Time (cycles)
    3) Operational setting 1
    4) Operational setting 2
    5) Operational setting 3
    6-26) Sensor measurements 1-21
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # URL for C-MAPSS dataset
    url = "https://ti.arc.nasa.gov/c/6/"
    zip_filename = "CMAPSSData.zip"
    zip_path = data_dir / zip_filename

    print("=" * 80)
    print("NASA C-MAPSS Turbofan Engine Degradation Dataset")
    print("=" * 80)
    print(f"\nDownload directory: {data_dir.absolute()}")

    # Check if already downloaded
    if zip_path.exists():
        print(f"\n✓ Found existing zip file: {zip_path}")
    else:
        print(f"\nDownloading dataset from NASA repository...")
        print(f"URL: {url}")
        try:
            download_url(url, str(zip_path))
            print(f"✓ Download completed: {zip_path}")
        except Exception as e:
            print(f"\n✗ Error downloading from NASA repository: {e}")
            print("\nAlternative: Manual download instructions")
            print("1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
            print("2. Navigate to: Turbofan Engine Degradation Simulation Data Set")
            print("3. Download 'CMAPSSData.zip'")
            print(f"4. Place it in: {data_dir.absolute()}")
            return False

    # Extract the dataset
    extract_dir = data_dir / "CMAPSS"

    if extract_dir.exists():
        print(f"\n✓ Dataset already extracted to: {extract_dir}")
    else:
        print(f"\nExtracting dataset...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"✓ Extraction completed to: {extract_dir}")
        except Exception as e:
            print(f"✗ Error extracting: {e}")
            return False

    # Verify all files are present
    print("\nVerifying dataset files...")
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    all_files_present = True

    for ds in datasets:
        train_file = extract_dir / f"train_{ds}.txt"
        test_file = extract_dir / f"test_{ds}.txt"
        rul_file = extract_dir / f"RUL_{ds}.txt"

        if train_file.exists() and test_file.exists() and rul_file.exists():
            print(f"  ✓ {ds}: train, test, RUL files found")
        else:
            print(f"  ✗ {ds}: Missing files!")
            all_files_present = False

    if all_files_present:
        print("\n" + "=" * 80)
        print("✓ C-MAPSS Dataset ready for use!")
        print("=" * 80)
        print("\nDataset Description:")
        print("  - 4 sub-datasets (FD001-FD004)")
        print("  - 21 sensor measurements per timestep")
        print("  - 3 operational settings")
        print("  - Varying operating conditions and fault modes")
        print("  - Total: ~700+ engine run-to-failure trajectories")
        print("\nNext steps:")
        print("  1. Run data_loader.py to preprocess the data")
        print("  2. Explore with notebooks/01_data_exploration.ipynb")
        return True
    else:
        print("\n✗ Dataset verification failed. Please check the files.")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download NASA C-MAPSS dataset')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Directory to download and extract dataset (default: current directory)')

    args = parser.parse_args()

    success = download_cmapss_dataset(args.data_dir)
    exit(0 if success else 1)
