import pandas as pd
import numpy as np
from pathlib import Path

def create_temporal_features(df, datetime_col='datetime'):
    df = df.copy()

    df['hour'] = df[datetime_col].dt.hour
    df['day_of_year'] = df[datetime_col].dt.dayofyear
    df['month'] = df[datetime_col].dt.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    df = df.drop(['hour', 'day_of_year', 'month'], axis=1)
    
    return df

def create_wave_features(df):
    df = df.copy()
    
    if 'WVHT' in df.columns and 'DPD' in df.columns:
        wave_height = df['WVHT'].fillna(0)
        wave_period = df['DPD'].fillna(0)
        df['wave_power'] = wave_height ** 2 * wave_period
    
    return df

def create_pressure_features(df):
    df = df.copy()
    
    if 'PRES' in df.columns:
        df['pressure_gradient'] = df['PRES'].diff()
        df['pressure_3h_change'] = df['PRES'].diff(periods=3)
        df['pressure_6h_change'] = df['PRES'].diff(periods=6)
    
    return df

def create_rolling_features(df, windows=[6, 12, 24]):
    df = df.copy()
    rolling_vars = ['WVHT', 'DPD', 'WSPD', 'PRES', 'WTMP']
    
    for var in rolling_vars:
        if var in df.columns:
            for window in windows:
                df[f'{var}_{window}h_mean'] = df[var].rolling(window=window, min_periods=1).mean()                
                df[f'{var}_{window}h_std'] = df[var].rolling(window=window, min_periods=1).std()

                if var == 'WVHT':
                    df[f'{var}_{window}h_min'] = df[var].rolling(window=window, min_periods=1).min()
                    df[f'{var}_{window}h_max'] = df[var].rolling(window=window, min_periods=1).max()
    
    return df

def create_lag_features(df, variables=['WVHT', 'DPD'], lags=[1, 3, 6, 12]):    
    df = df.copy()
    
    for var in variables:
        if var in df.columns:
            for lag in lags:
                df[f'{var}_lag_{lag}h'] = df[var].shift(lag)
    
    return df

def engineer_features(df, datetime_col='datetime'):    
    print(f"Starting feature engineering with {len(df)} records...")
    
    original_cols = df.columns.tolist()
    print(f"Original features: {len(original_cols)} columns")
    
    df = create_temporal_features(df, datetime_col)
    print(f"After temporal features: {len(df.columns)} columns")
    
    df = create_wave_features(df)
    print(f"After wave features: {len(df.columns)} columns")
    
    df = create_pressure_features(df)
    print(f"After pressure features: {len(df.columns)} columns")
    
    df = create_rolling_features(df)
    print(f"After rolling features: {len(df.columns)} columns")
    
    df = create_lag_features(df)
    print(f"After lag features: {len(df.columns)} columns")
    
    new_features = [col for col in df.columns if col not in original_cols]
    print(f"\nNew features created ({len(new_features)}):")
    for feature in new_features:
        print(f"  - {feature}")
    
    return df

def temporal_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, datetime_col='datetime'):    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    df_sorted = df.sort_values(datetime_col).reset_index(drop=True)
    
    n_samples = len(df_sorted)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def process_station(station_id, data_dir="/Users/maxpintchouk/Code/DeepLearning/Surf-Forecast/data"):
    print(f"\n{'='*60}")
    print(f"Processing Station {station_id}")
    print(f"{'='*60}")
    
    input_file = Path(data_dir) / "processed" / f"{station_id}_cleaned.csv"
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"Loaded {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")

    df_features = engineer_features(df)
    
    missing_pct = df_features.select_dtypes(include=[np.number]).isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 20]
    if len(high_missing) > 0:
        print(f"\nWarning: Features with >20% missing values:")
        for col, pct in high_missing.items():
            print(f"  {col}: {pct:.1f}%")
    
    train_df, val_df, test_df = temporal_split(df_features)
    
    print(f"\nData splits:")
    print(f"  Training:   {len(train_df):,} records ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    print(f"  Validation: {len(val_df):,} records ({val_df['datetime'].min()} to {val_df['datetime'].max()})")  
    print(f"  Test:       {len(test_df):,} records ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
    
    output_dir = Path(data_dir) / "splits" / station_id
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    df_features.to_csv(output_dir / "features_full.csv", index=False)
    
    feature_summary = {
        'station_id': station_id,
        'total_records': len(df),
        'total_features': len(df_features.columns),
        'date_range': {
            'start': df['datetime'].min().isoformat(),
            'end': df['datetime'].max().isoformat()
        },
        'splits': {
            'train': {
                'records': len(train_df),
                'start': train_df['datetime'].min().isoformat(),
                'end': train_df['datetime'].max().isoformat()
            },
            'val': {
                'records': len(val_df), 
                'start': val_df['datetime'].min().isoformat(),
                'end': val_df['datetime'].max().isoformat()
            },
            'test': {
                'records': len(test_df),
                'start': test_df['datetime'].min().isoformat(), 
                'end': test_df['datetime'].max().isoformat()
            }
        },
        'feature_columns': df_features.columns.tolist(),
        'original_columns': df.columns.tolist(),
        'engineered_features': [col for col in df_features.columns if col not in df.columns]
    }
    
    import json
    with open(output_dir / "feature_summary.json", 'w') as f:
        json.dump(feature_summary, f, indent=2)
    
    print(f"\nSaved to {output_dir}:")
    print(f"  ✓ train.csv ({len(train_df):,} records)")
    print(f"  ✓ val.csv ({len(val_df):,} records)")
    print(f"  ✓ test.csv ({len(test_df):,} records)")
    print(f"  ✓ features_full.csv ({len(df_features):,} records)")
    print(f"  ✓ feature_summary.json")
    
    return feature_summary

def main():
    stations = ['46012', '46221', '46026']
    
    for station_id in stations:
        try:
            summary = process_station(station_id)
            print(f"\n✓ Station {station_id} complete!")
            print(f"  Created {len(summary['engineered_features'])} new features")
            print(f"  Total features: {summary['total_features']}")
            
        except Exception as e:
            print(f"\n❌ Error processing station {station_id}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Feature engineering and data splitting complete!")
    print("Next steps: Implement baseline models")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()