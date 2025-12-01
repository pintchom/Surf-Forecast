"""
Data preprocessing and cleaning for NOAA buoy data.

Handles missing values, resampling, and data quality issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Optional


class BuoyDataCleaner:
    def __init__(self):
        self.required_features = ["WVHT", "DPD", "WSPD", "PRES", "WTMP"]
        self.target_variables = ["WVHT", "DPD"]
        self.missing_codes = [9999.0, 999.0, 99.0, 99.00, 999]

    def load_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        print(f"Loaded {file_path}: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        return df

    def replace_missing_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        for col in df_clean.columns:
            if col in self.required_features:
                df_clean[col] = df_clean[col].replace(self.missing_codes, np.nan)

        return df_clean

    def resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        df_hourly = df[numeric_cols].resample("H").mean()

        print(f"Resampled to hourly: {df.shape} -> {df_hourly.shape}")
        return df_hourly

    def interpolate_short_gaps(
        self, df: pd.DataFrame, max_gap_hours: int = 3
    ) -> pd.DataFrame:
        df_interp = df.copy()

        for col in self.required_features:
            if col in df_interp.columns:
                df_interp[col] = df_interp[col].interpolate(
                    method="linear", limit=max_gap_hours, limit_direction="both"
                )

        return df_interp

    def remove_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_valid = df.copy()
        if "WVHT" in df_valid.columns:
            valid_wvht = df_valid["WVHT"].isna() | (
                (df_valid["WVHT"] >= 0) & (df_valid["WVHT"] <= 20)
            )
            df_valid = df_valid[valid_wvht]

        if "DPD" in df_valid.columns:
            valid_dpd = df_valid["DPD"].isna() | (
                (df_valid["DPD"] >= 1) & (df_valid["DPD"] <= 30)
            )
            df_valid = df_valid[valid_dpd]

        if "WSPD" in df_valid.columns:
            valid_wspd = df_valid["WSPD"].isna() | (
                (df_valid["WSPD"] >= 0) & (df_valid["WSPD"] <= 50)
            )
            df_valid = df_valid[valid_wspd]

        if "PRES" in df_valid.columns:
            valid_pres = df_valid["PRES"].isna() | (
                (df_valid["PRES"] >= 950) & (df_valid["PRES"] <= 1050)
            )
            df_valid = df_valid[valid_pres]

        print(f"Removed invalid values: {df.shape} -> {df_valid.shape}")
        return df_valid

    def remove_rows_missing_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        df_targets = df.copy()

        for target in self.target_variables:
            if target in df_targets.columns:
                df_targets = df_targets.dropna(subset=[target])

        print(f"Removed rows with missing targets: {df.shape} -> {df_targets.shape}")
        return df_targets

    def calculate_data_quality_metrics(
        self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame
    ) -> Dict:
        metrics = {}

        for col in self.required_features:
            if col in df_original.columns:
                original_count = df_original[col].notna().sum()
                cleaned_count = (
                    df_cleaned[col].notna().sum() if col in df_cleaned.columns else 0
                )

                metrics[col] = {
                    "original_count": original_count,
                    "cleaned_count": cleaned_count,
                    "missing_rate_original": 1 - (original_count / len(df_original)),
                    "missing_rate_cleaned": 1 - (cleaned_count / len(df_cleaned))
                    if len(df_cleaned) > 0
                    else 1.0,
                    "data_retained": cleaned_count / original_count
                    if original_count > 0
                    else 0,
                }

        return metrics

    def clean_station_data(self, input_path: str, output_path: str) -> Dict:
        print(f"\n=== Cleaning {input_path} ===")

        df_original = self.load_data(input_path)

        df = self.replace_missing_codes(df_original)
        df = self.resample_to_hourly(df)
        df = self.interpolate_short_gaps(df)
        df = self.remove_invalid_values(df)
        df = self.remove_rows_missing_targets(df)

        available_features = [
            col for col in self.required_features if col in df.columns
        ]
        df_final = df[available_features].copy()

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path)

        metrics = self.calculate_data_quality_metrics(df_original, df_final)

        print(f"Saved cleaned data: {output_path}")
        print(f"Final shape: {df_final.shape}")
        print(f"Final date range: {df_final.index.min()} to {df_final.index.max()}")

        return metrics

    def print_quality_report(self, metrics: Dict, station_id: str):
        print(f"\n=== Data Quality Report: Station {station_id} ===")

        for feature, stats in metrics.items():
            print(f"\n{feature}:")
            print(f"  Original data points: {stats['original_count']:,}")
            print(f"  Cleaned data points: {stats['cleaned_count']:,}")
            print(f"  Missing rate (original): {stats['missing_rate_original']:.1%}")
            print(f"  Missing rate (cleaned): {stats['missing_rate_cleaned']:.1%}")
            print(f"  Data retained: {stats['data_retained']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Clean NOAA buoy data")
    parser.add_argument(
        "--input-dir",
        default="data/processed",
        help="Input directory with combined CSV files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for cleaned files",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=["46012", "46221", "46026"],
        help="Station IDs to process",
    )

    args = parser.parse_args()

    cleaner = BuoyDataCleaner()

    for station in args.stations:
        input_file = f"{args.input_dir}/{station}_combined.csv"
        output_file = f"{args.output_dir}/{station}_cleaned.csv"

        try:
            metrics = cleaner.clean_station_data(input_file, output_file)
            cleaner.print_quality_report(metrics, station)
        except FileNotFoundError:
            print(f"File not found: {input_file}")
        except Exception as e:
            print(f"Error processing {station}: {e}")


if __name__ == "__main__":
    main()
