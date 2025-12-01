"""
Historical Surf Data Lookup Script

Fetches surf conditions for a specific date/time and the following 6 hours.
Usage: python historical_surf_lookup.py --date "2024-11-30" --hour 20 --station 46221
"""

import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Optional, Dict, List

class HistoricalSurfLookup:
    def __init__(self):
        self.base_url = "https://www.ndbc.noaa.gov/view_text_file.php"
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_station_data(self, station_id: str, year: int) -> Optional[pd.DataFrame]:
        """Download and parse NOAA buoy data for a specific station and year."""
        # For recent data (current year), try real-time endpoint first
        current_year = datetime.now().year
        if year == current_year:
            print(f"Attempting to fetch recent data from real-time endpoint...")
            realtime_data = self.fetch_realtime_data(station_id)
            if realtime_data is not None:
                return realtime_data
        
        # Fallback to historical data endpoint
        filename = f"{station_id}h{year}.txt"
        url = f"{self.base_url}?filename={filename}.gz&dir=data/historical/stdmet/"
        
        cache_path = self.data_dir / f"{station_id}_{year}.txt"
        
        # Check if we have cached data
        if cache_path.exists():
            print(f"Using cached data: {cache_path}")
            try:
                return self.parse_noaa_data(cache_path)
            except Exception as e:
                print(f"Error reading cached file, re-downloading: {e}")
        
        print(f"Downloading {station_id} data for {year}...")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save the data
            with open(cache_path, "w") as f:
                f.write(response.text)
            
            print(f"Successfully downloaded: {cache_path}")
            return self.parse_noaa_data(cache_path)
            
        except requests.RequestException as e:
            print(f"Error downloading {station_id} {year}: {e}")
            return None
    
    def fetch_realtime_data(self, station_id: str) -> Optional[pd.DataFrame]:
        """Fetch real-time data (last 45 days) from NOAA API."""
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the real-time data format
            lines = response.text.strip().split('\n')
            
            # Find header line
            data_start = 0
            header_line = None
            for i, line in enumerate(lines):
                if line.startswith("#YY") or (i == 0 and "YY" in line):
                    header_line = line.replace("#", "").strip()
                    data_start = i + 2  # Skip header and units line
                    break
            
            if header_line is None:
                return None
            
            # Column names
            columns = header_line.split()
            
            # Parse data rows
            data_rows = []
            for line in lines[data_start:]:
                if line.strip() and not line.startswith('#'):
                    row = line.split()
                    if len(row) >= len(columns):
                        data_rows.append(row[:len(columns)])
            
            if not data_rows:
                return None
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=columns)
            
            # Convert time columns
            time_cols = ['YY', 'MM', 'DD', 'hh', 'mm']
            for col in time_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert other columns to numeric
            numeric_columns = ['WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 
                             'DEWP', 'VIS', 'PTDY', 'TIDE', 'WSPD', 'GST', 'WDIR']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].replace(['99.00', '999.0', '99.0', '999', '99', 'MM', '9999'], pd.NA)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create datetime
            year_col = df['YY'].copy()
            if year_col.max() < 100:
                year_col = year_col + 2000
            
            datetime_dict = {
                'year': year_col,
                'month': df['MM'],
                'day': df['DD'],
                'hour': df['hh'] if 'hh' in df.columns else 0,
                'minute': df['mm'] if 'mm' in df.columns else 0
            }
            
            df['datetime'] = pd.to_datetime(datetime_dict, errors='coerce')
            df = df.dropna(subset=['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # Drop time columns
            time_cols_to_drop = [col for col in time_cols if col in df.columns]
            df = df.drop(columns=time_cols_to_drop, errors='ignore')
            
            print(f"✅ Fetched real-time data: {len(df)} records")
            return df
            
        except Exception as e:
            print(f"Failed to fetch real-time data: {e}")
            return None
    
    def parse_noaa_data(self, file_path: Path) -> pd.DataFrame:
        """Parse NOAA buoy data file."""
        try:
            # Read the file, skipping the units row (second line)
            df = pd.read_csv(file_path, sep=r'\s+', skiprows=[1], na_values=['99.0', '999.0', '9999.0'])
            
            # Create datetime column
            df['datetime'] = pd.to_datetime(df[['#YY', 'MM', 'DD', 'hh', 'mm']])
            
            # Select relevant columns for surf conditions
            surf_columns = ['datetime', 'WVHT', 'DPD', 'APD', 'MWD', 'WSPD', 'WDIR', 'PRES']
            available_columns = [col for col in surf_columns if col in df.columns]
            
            return df[available_columns].set_index('datetime')
            
        except Exception as e:
            print(f"Error parsing file {file_path}: {e}")
            return None
    
    def get_surf_conditions(self, station_id: str, target_datetime: datetime, hours_forward: int = 6) -> Optional[pd.DataFrame]:
        """Get surf conditions for a specific time and the following hours."""
        year = target_datetime.year
        
        # Download data for the target year
        df = self.download_station_data(station_id, year)
        if df is None:
            return None
        
        # Create time range
        end_datetime = target_datetime + timedelta(hours=hours_forward)
        
        # Filter data for the requested time range
        mask = (df.index >= target_datetime) & (df.index <= end_datetime)
        result = df.loc[mask].copy()
        
        if result.empty:
            print(f"No data found for {station_id} between {target_datetime} and {end_datetime}")
            return None
        
        # Add hours since start column
        result['hours_since_start'] = (result.index - target_datetime).total_seconds() / 3600
        
        return result
    
    def format_surf_report(self, df: pd.DataFrame, station_id: str, start_time: datetime) -> str:
        """Format surf conditions into a readable report."""
        if df is None or df.empty:
            return "No data available"
        
        station_names = {
            '46012': 'Half Moon Bay, CA',
            '46221': 'Santa Monica, CA'
        }
        
        station_name = station_names.get(station_id, f"Station {station_id}")
        
        report = f"""
🏄‍♂️ SURF CONDITIONS REPORT
Station: {station_id} - {station_name}
Start Time: {start_time.strftime('%Y-%m-%d %H:%M UTC')}

{'Time (UTC)':<20} {'Wave Height':<12} {'Wave Period':<12} {'Wind Speed':<12} {'Direction':<12} {'Pressure':<12}
{'='*90}
"""
        
        for idx, row in df.iterrows():
            time_str = idx.strftime('%m-%d %H:%M')
            wave_height = f"{row.get('WVHT', 'N/A'):.1f}m" if pd.notna(row.get('WVHT')) else "N/A"
            wave_period = f"{row.get('DPD', 'N/A'):.1f}s" if pd.notna(row.get('DPD')) else "N/A"
            wind_speed = f"{row.get('WSPD', 'N/A'):.1f}m/s" if pd.notna(row.get('WSPD')) else "N/A"
            wind_dir = f"{row.get('WDIR', 'N/A'):.0f}°" if pd.notna(row.get('WDIR')) else "N/A"
            pressure = f"{row.get('PRES', 'N/A'):.1f}hPa" if pd.notna(row.get('PRES')) else "N/A"
            
            report += f"{time_str:<20} {wave_height:<12} {wave_period:<12} {wind_speed:<12} {wind_dir:<12} {pressure:<12}\n"
        
        # Add summary
        if 'WVHT' in df.columns and df['WVHT'].notna().any():
            avg_height = df['WVHT'].mean()
            max_height = df['WVHT'].max()
            report += f"\n📊 SUMMARY:\n"
            report += f"Average Wave Height: {avg_height:.2f}m\n"
            report += f"Maximum Wave Height: {max_height:.2f}m\n"
            
            if avg_height >= 1.2:
                report += f"🟢 Surf Conditions: GOOD (waves >= 1.2m)\n"
            else:
                report += f"🔴 Surf Conditions: POOR (waves < 1.2m)\n"
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Look up historical surf conditions')
    parser.add_argument('--date', required=True, help='Date in YYYY-MM-DD format')
    parser.add_argument('--hour', type=int, required=True, help='Hour in 24-hour UTC format (0-23)')
    parser.add_argument('--station', default='46221', help='Buoy station ID (default: 46221 - Santa Monica)')
    parser.add_argument('--hours', type=int, default=6, help='Number of hours forward to look (default: 6)')
    
    args = parser.parse_args()
    
    try:
        # Parse the date
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
        target_datetime = target_date.replace(hour=args.hour, minute=0, second=0)
        
        print(f"Looking up surf conditions for {args.station} starting {target_datetime.strftime('%Y-%m-%d %H:%M UTC')}...")
        
        # Create lookup instance and get data
        lookup = HistoricalSurfLookup()
        surf_data = lookup.get_surf_conditions(args.station, target_datetime, args.hours)
        
        # Format and display report
        report = lookup.format_surf_report(surf_data, args.station, target_datetime)
        print(report)
        
        # Optionally save to file
        output_file = f"surf_report_{args.station}_{args.date}_{args.hour:02d}UTC.txt"
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {output_file}")
        
    except ValueError as e:
        print(f"Error: Invalid date format. Use YYYY-MM-DD. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()