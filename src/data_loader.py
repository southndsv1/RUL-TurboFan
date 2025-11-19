"""
C-MAPSS Data Loader and Preprocessing

This module handles loading, preprocessing, and preparing the NASA C-MAPSS
turbofan engine dataset for training deep learning models.

Features:
- Load raw C-MAPSS data files
- Normalize sensor readings
- Create sliding windows for time-series
- Piecewise linear RUL labeling
- Data augmentation (time-warping, magnitude-warping)
- Train/val/test splitting
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class CMAPSSDataLoader:
    """Load and preprocess NASA C-MAPSS dataset."""

    # Column names for the dataset
    INDEX_NAMES = ['unit_id', 'time_cycles']
    SETTING_NAMES = ['setting_1', 'setting_2', 'setting_3']
    SENSOR_NAMES = [f'sensor_{i}' for i in range(1, 22)]
    COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

    # Sensors with near-zero variance (to be dropped)
    # Based on literature: sensors 1, 5, 6, 10, 16, 18, 19 have little variation
    DROP_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                    'sensor_16', 'sensor_18', 'sensor_19']

    def __init__(self, data_dir: str = '../data/CMAPSS', dataset_name: str = 'FD001'):
        """
        Initialize data loader.

        Args:
            data_dir: Path to CMAPSS data directory
            dataset_name: One of ['FD001', 'FD002', 'FD003', 'FD004']
        """
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name

        # Paths to data files
        self.train_file = self.data_dir / f'train_{dataset_name}.txt'
        self.test_file = self.data_dir / f'test_{dataset_name}.txt'
        self.rul_file = self.data_dir / f'RUL_{dataset_name}.txt'

        # Scalers for normalization
        self.sensor_scaler = None
        self.setting_scaler = None

        # Dataset characteristics
        self.sequence_length = None
        self.feature_names = None

        print(f"Initialized CMAPSSDataLoader for {dataset_name}")
        print(f"Data directory: {self.data_dir}")

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Load raw data from text files.

        Returns:
            train_df: Training data
            test_df: Test data
            test_rul: True RUL values for test data
        """
        print(f"\nLoading {self.dataset_name} data...")

        # Load training data
        train_df = pd.read_csv(self.train_file, sep=r'\s+', header=None, names=self.COL_NAMES)
        print(f"  Training: {len(train_df)} samples, {train_df['unit_id'].nunique()} units")

        # Load test data
        test_df = pd.read_csv(self.test_file, sep=r'\s+', header=None, names=self.COL_NAMES)
        print(f"  Test: {len(test_df)} samples, {test_df['unit_id'].nunique()} units")

        # Load true RUL values for test data
        test_rul = pd.read_csv(self.rul_file, sep=r'\s+', header=None).values.flatten()
        print(f"  Test RUL: {len(test_rul)} values")

        return train_df, test_df, test_rul

    def add_rul_labels(self, df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
        """
        Add RUL (Remaining Useful Life) labels to dataframe.

        Uses piecewise linear labeling:
        - RUL = max_rul for early cycles
        - RUL decreases linearly to 0 at failure

        Args:
            df: Input dataframe with 'unit_id' and 'time_cycles'
            max_rul: Maximum RUL value (clip early cycles)

        Returns:
            Dataframe with 'RUL' column added
        """
        df = df.copy()

        # Calculate RUL for each unit
        rul_list = []

        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id]
            max_cycles = unit_data['time_cycles'].max()

            # RUL = remaining cycles until failure
            unit_rul = max_cycles - unit_data['time_cycles']

            # Apply piecewise linear clipping
            unit_rul = unit_rul.clip(upper=max_rul)

            rul_list.append(unit_rul.values)

        # Add RUL column
        df['RUL'] = np.concatenate(rul_list)

        return df

    def normalize_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                       method: str = 'z-score') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize sensor readings and operational settings.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            method: 'z-score' or 'min-max'

        Returns:
            Normalized train and test dataframes
        """
        print(f"\nNormalizing data using {method}...")

        train_df = train_df.copy()
        test_df = test_df.copy()

        # Select scaler
        if method == 'z-score':
            self.sensor_scaler = StandardScaler()
            self.setting_scaler = StandardScaler()
        elif method == 'min-max':
            self.sensor_scaler = MinMaxScaler()
            self.setting_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Get feature names (exclude dropped sensors)
        sensor_cols = [s for s in self.SENSOR_NAMES if s not in self.DROP_SENSORS]
        setting_cols = self.SETTING_NAMES

        # Fit on training data
        self.sensor_scaler.fit(train_df[sensor_cols])
        self.setting_scaler.fit(train_df[setting_cols])

        # Transform both train and test
        train_df[sensor_cols] = self.sensor_scaler.transform(train_df[sensor_cols])
        train_df[setting_cols] = self.setting_scaler.transform(train_df[setting_cols])

        test_df[sensor_cols] = self.sensor_scaler.transform(test_df[sensor_cols])
        test_df[setting_cols] = self.setting_scaler.transform(test_df[setting_cols])

        self.feature_names = setting_cols + sensor_cols
        print(f"  Normalized {len(self.feature_names)} features")

        return train_df, test_df

    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int = 30,
                          stride: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for time-series.

        Args:
            df: Input dataframe with features and RUL
            sequence_length: Length of each sequence (window size)
            stride: Step size for sliding window

        Returns:
            sequences: Shape (num_sequences, sequence_length, num_features)
            labels: RUL values, shape (num_sequences,)
            unit_ids: Unit ID for each sequence
        """
        print(f"\nPreparing sequences (window={sequence_length}, stride={stride})...")

        sequences = []
        labels = []
        unit_ids_list = []

        # Get feature names
        sensor_cols = [s for s in self.SENSOR_NAMES if s not in self.DROP_SENSORS]
        feature_cols = self.SETTING_NAMES + sensor_cols

        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id]

            # Extract features and RUL
            features = unit_data[feature_cols].values
            rul_values = unit_data['RUL'].values

            # Create sliding windows
            for i in range(0, len(features) - sequence_length + 1, stride):
                seq = features[i:i + sequence_length]
                # Use RUL at the end of the sequence
                rul = rul_values[i + sequence_length - 1]

                sequences.append(seq)
                labels.append(rul)
                unit_ids_list.append(unit_id)

        sequences = np.array(sequences)
        labels = np.array(labels)
        unit_ids_array = np.array(unit_ids_list)

        print(f"  Created {len(sequences)} sequences")
        print(f"  Shape: {sequences.shape}")
        print(f"  Labels: min={labels.min():.1f}, max={labels.max():.1f}, mean={labels.mean():.1f}")

        return sequences, labels, unit_ids_array

    def prepare_test_sequences(self, df: pd.DataFrame, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare test sequences - use only the last window for each unit.

        Args:
            df: Test dataframe
            sequence_length: Length of each sequence

        Returns:
            sequences: Shape (num_units, sequence_length, num_features)
            unit_ids: Unit IDs
        """
        print(f"\nPreparing test sequences (last {sequence_length} cycles per unit)...")

        sequences = []
        unit_ids_list = []

        # Get feature names
        sensor_cols = [s for s in self.SENSOR_NAMES if s not in self.DROP_SENSORS]
        feature_cols = self.SETTING_NAMES + sensor_cols

        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id]
            features = unit_data[feature_cols].values

            # Take last sequence_length cycles
            if len(features) >= sequence_length:
                seq = features[-sequence_length:]
            else:
                # Pad if shorter than sequence_length
                padding = np.zeros((sequence_length - len(features), features.shape[1]))
                seq = np.vstack([padding, features])

            sequences.append(seq)
            unit_ids_list.append(unit_id)

        sequences = np.array(sequences)
        unit_ids_array = np.array(unit_ids_list)

        print(f"  Created {len(sequences)} test sequences")
        print(f"  Shape: {sequences.shape}")

        return sequences, unit_ids_array

    def load_and_preprocess(self, sequence_length: int = 30, max_rul: int = 125,
                            val_split: float = 0.2, stride: int = 1,
                            normalize: str = 'z-score') -> Dict:
        """
        Complete pipeline: load, preprocess, and prepare data.

        Args:
            sequence_length: Window size for sequences
            max_rul: Maximum RUL for piecewise linear labeling
            val_split: Fraction of training units for validation
            stride: Stride for sliding window
            normalize: Normalization method

        Returns:
            Dictionary containing all prepared data
        """
        print("=" * 80)
        print(f"C-MAPSS {self.dataset_name} Data Preprocessing Pipeline")
        print("=" * 80)

        self.sequence_length = sequence_length

        # 1. Load raw data
        train_df, test_df, test_rul = self.load_raw_data()

        # 2. Drop low-variance sensors
        print(f"\nDropping {len(self.DROP_SENSORS)} low-variance sensors...")
        train_df = train_df.drop(columns=self.DROP_SENSORS)
        test_df = test_df.drop(columns=self.DROP_SENSORS)

        # 3. Add RUL labels
        train_df = self.add_rul_labels(train_df, max_rul=max_rul)

        # 4. Split train into train and validation
        train_units = train_df['unit_id'].unique()
        np.random.seed(42)
        np.random.shuffle(train_units)

        val_size = int(len(train_units) * val_split)
        val_units = train_units[:val_size]
        train_units_final = train_units[val_size:]

        val_df = train_df[train_df['unit_id'].isin(val_units)].copy()
        train_df = train_df[train_df['unit_id'].isin(train_units_final)].copy()

        print(f"\nTrain/Val split: {len(train_units_final)} / {len(val_units)} units")

        # 5. Normalize data
        train_df, val_df = self.normalize_data(train_df, val_df, method=normalize)
        train_df, test_df = self.normalize_data(train_df, test_df, method=normalize)

        # 6. Create sequences
        X_train, y_train, train_unit_ids = self.prepare_sequences(
            train_df, sequence_length=sequence_length, stride=stride
        )
        X_val, y_val, val_unit_ids = self.prepare_sequences(
            val_df, sequence_length=sequence_length, stride=stride
        )
        X_test, test_unit_ids = self.prepare_test_sequences(
            test_df, sequence_length=sequence_length
        )

        print("\n" + "=" * 80)
        print("Data Preprocessing Complete!")
        print("=" * 80)
        print(f"\nDataset Statistics:")
        print(f"  Training sequences: {len(X_train)}")
        print(f"  Validation sequences: {len(X_val)}")
        print(f"  Test sequences: {len(X_test)}")
        print(f"  Sequence shape: {X_train.shape[1:]}")
        print(f"  Features: {X_train.shape[2]}")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'train_unit_ids': train_unit_ids,
            'X_val': X_val,
            'y_val': y_val,
            'val_unit_ids': val_unit_ids,
            'X_test': X_test,
            'y_test': test_rul,
            'test_unit_ids': test_unit_ids,
            'feature_names': self.feature_names,
            'sequence_length': sequence_length,
            'num_features': X_train.shape[2],
            'scaler_sensor': self.sensor_scaler,
            'scaler_setting': self.setting_scaler,
        }


class DataAugmentation:
    """Data augmentation techniques for time-series."""

    @staticmethod
    def time_warp(sequences: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """
        Apply time warping to sequences.

        Args:
            sequences: Shape (batch, time, features)
            sigma: Standard deviation of warping

        Returns:
            Warped sequences
        """
        batch_size, time_steps, features = sequences.shape
        augmented = []

        for seq in sequences:
            # Generate smooth warping curve
            warp = np.random.normal(loc=1.0, scale=sigma, size=(features,))
            warped_seq = seq * warp
            augmented.append(warped_seq)

        return np.array(augmented)

    @staticmethod
    def magnitude_warp(sequences: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        Apply magnitude warping (scaling) to sequences.

        Args:
            sequences: Shape (batch, time, features)
            sigma: Standard deviation of scaling

        Returns:
            Warped sequences
        """
        batch_size, time_steps, features = sequences.shape
        scale = np.random.normal(loc=1.0, scale=sigma, size=(batch_size, 1, features))
        return sequences * scale

    @staticmethod
    def add_noise(sequences: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to sequences.

        Args:
            sequences: Shape (batch, time, features)
            noise_level: Standard deviation of noise

        Returns:
            Noisy sequences
        """
        noise = np.random.normal(loc=0.0, scale=noise_level, size=sequences.shape)
        return sequences + noise


if __name__ == "__main__":
    # Example usage
    import sys

    dataset = 'FD001' if len(sys.argv) < 2 else sys.argv[1]

    loader = CMAPSSDataLoader(data_dir='../data/CMAPSS', dataset_name=dataset)
    data = loader.load_and_preprocess(sequence_length=30, max_rul=125, val_split=0.2)

    print("\nSample data:")
    print(f"  X_train[0] shape: {data['X_train'][0].shape}")
    print(f"  y_train[0]: {data['y_train'][0]}")
    print(f"\nFeatures: {data['feature_names']}")
