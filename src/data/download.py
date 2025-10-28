"""
NOAA Buoy Data Downloader

Downloads historical buoy data from NOAA National Data Buoy Center.
Supports stations 46012 (Half Moon Bay) and 46221 (Santa Barbara).
"""

import requests
import pandas as pd
import os
from pathlib import Path
import time
from typing import List, Dict
import argparse


class NOAABuoyDownloader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://www.ndbc.noaa.gov/view_text_file.php"

    def download_station_year(self, station_id: str, year: int) -> bool:
        filename = f"{station_id}h{year}.txt"
        url = f"{self.base_url}?filename={filename}.gz&dir=data/historical/stdmet/"

        output_path = self.data_dir / f"{station_id}_{year}.txt"

        if output_path.exists():
            print(f"File already exists: {output_path}")
            return True

        print(f"Downloading {station_id} data for {year}...")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Save the data
            with open(output_path, "w") as f:
                f.write(response.text)

            print(f"Successfully downloaded: {output_path}")
            return True

        except requests.RequestException as e:
            print(f"Error downloading {station_id} {year}: {e}")
            return False

    def download_all_data(
        self,
        stations: List[str] = ["46012", "46221"],
        start_year: int = 2018,
        end_year: int = 2024,
    ) -> Dict[str, List[int]]:
        results = {station: [] for station in stations}

        for station in stations:
            print(f"\n=== Downloading data for station {station} ===")

            for year in range(start_year, end_year + 1):
                success = self.download_station_year(station, year)
                if success:
                    results[station].append(year)
                time.sleep(1)

        return results

    def parse_data_file(self, file_path: Path) -> pd.DataFrame:
        with open(file_path, "r") as f:
            lines = f.readlines()

        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith("#YY"):
                data_start = i + 2 
                break

        header_line = lines[data_start - 2].strip()
        columns = header_line.replace("#", "").split()

        data_lines = lines[data_start:]
        data = []

        for line in data_lines:
            if line.strip():
                row = line.split()
                if len(row) >= len(columns):
                    data.append(row[: len(columns)])

        if not data:
            print(f"No data found in {file_path}")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=columns)

        for col in df.columns:
            if col in ["YY", "MM", "DD", "hh", "mm"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].replace(
                    ["99.00", "999.0", "99.0", "999", "99", "MM"], pd.NA
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        year_col = df["YY"]
        if year_col.max() < 100:
            year_col = year_col + 2000 

        df["datetime"] = pd.to_datetime(
            {
                "year": year_col,
                "month": df["MM"],
                "day": df["DD"],
                "hour": df["hh"],
                "minute": df["mm"] if "mm" in df.columns else 0,
            },
            errors="coerce",
        )

        df = df.set_index("datetime")

        time_cols = ["YY", "MM", "DD", "hh"]
        if "mm" in df.columns:
            time_cols.append("mm")
        df = df.drop(columns=time_cols, errors="ignore")

        return df

    def combine_station_data(self, station_id: str) -> pd.DataFrame:
        station_files = list(self.data_dir.glob(f"{station_id}_*.txt"))

        if not station_files:
            print(f"No files found for station {station_id}")
            return pd.DataFrame()

        all_data = []

        for file_path in sorted(station_files):
            print(f"Processing {file_path}")
            df = self.parse_data_file(file_path)
            if not df.empty:
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=False)
        combined_df = combined_df.sort_index()

        combined_df = combined_df[~combined_df.index.duplicated(keep="first")]

        return combined_df

    def save_combined_data(self, stations: List[str] = ["46012", "46221"]):
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        for station in stations:
            print(f"\nCombining data for station {station}...")
            df = self.combine_station_data(station)

            if not df.empty:
                output_path = processed_dir / f"{station}_combined.csv"
                df.to_csv(output_path)
                print(f"Saved combined data: {output_path}")
                print(f"Data shape: {df.shape}")
                print(f"Date range: {df.index.min()} to {df.index.max()}")
            else:
                print(f"No data to save for station {station}")


def main():
    parser = argparse.ArgumentParser(description="Download NOAA buoy data")
    parser.add_argument(
        "--stations",
        nargs="+",
        default=["46012", "46221"],
        help="Station IDs to download",
    )
    parser.add_argument("--start-year", type=int, default=2018, help="Start year")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument(
        "--data-dir", default="data/raw", help="Directory to save raw data"
    )

    args = parser.parse_args()

    downloader = NOAABuoyDownloader(args.data_dir)
    results = downloader.download_all_data(
        args.stations, args.start_year, args.end_year
    )

    print("\n=== Download Summary ===")
    for station, years in results.items():
        print(f"Station {station}: {len(years)} years downloaded")

    downloader.save_combined_data(args.stations)


if __name__ == "__main__":
    main()
