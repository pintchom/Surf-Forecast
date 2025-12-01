import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

def create_sequences(df, feature_cols, target_cols, lookback_hours=24, forecast_horizons=[1, 3, 6]):
    """
    Transform time series data into sequences for supervised learning.
    
    Args:
        df: DataFrame with time series data (sorted by datetime)
        feature_cols: List of feature column names to use as inputs
        target_cols: List of target column names (e.g., ['WVHT', 'DPD'])
        lookback_hours: Number of past hours to use as input
        forecast_horizons: List of hours ahead to predict (e.g., [1, 3, 6])
    
    Returns:
        X: Input sequences (n_samples, lookback_hours, n_features)
        y: Multi-horizon targets (n_samples, n_horizons * n_targets)
        valid_indices: Original indices of valid samples
    """
    
    # Ensure data is sorted by datetime
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    
    max_horizon = max(forecast_horizons)
    n_samples = len(df_sorted)
    n_features = len(feature_cols)
    n_targets = len(target_cols)
    n_horizons = len(forecast_horizons)
    
    # Calculate valid sample range
    # Need lookback_hours for input + max_horizon for output
    start_idx = lookback_hours
    end_idx = n_samples - max_horizon
    n_valid_samples = end_idx - start_idx
    
    if n_valid_samples <= 0:
        raise ValueError(f"Not enough data: need at least {lookback_hours + max_horizon} samples, got {n_samples}")
    
    print(f"Creating sequences:")
    print(f"  Input features: {n_features}")
    print(f"  Target variables: {n_targets} ({target_cols})")
    print(f"  Forecast horizons: {forecast_horizons}")
    print(f"  Lookback window: {lookback_hours} hours")
    print(f"  Valid samples: {n_valid_samples} (from {n_samples} total)")
    
    # Initialize arrays
    X = np.zeros((n_valid_samples, lookback_hours, n_features))
    y = np.zeros((n_valid_samples, n_horizons * n_targets))
    valid_indices = []
    
    # Create sequences
    for i in range(n_valid_samples):
        sample_idx = start_idx + i
        
        # Input sequence: lookback_hours of features
        input_start = sample_idx - lookback_hours
        input_end = sample_idx
        X[i] = df_sorted[feature_cols].iloc[input_start:input_end].values
        
        # Output targets: future values at specified horizons
        target_idx = 0
        for horizon in forecast_horizons:
            future_idx = sample_idx + horizon
            for target_col in target_cols:
                y[i, target_idx] = df_sorted[target_col].iloc[future_idx]
                target_idx += 1
        
        valid_indices.append(sample_idx)
    
    print(f"  Sequences created: X shape {X.shape}, y shape {y.shape}")
    
    return X, y, valid_indices

def prepare_linear_regression_data(X):
    """
    Flatten sequences for linear regression input.
    
    Args:
        X: Sequence data (n_samples, lookback_hours, n_features)
    
    Returns:
        X_flat: Flattened features (n_samples, lookback_hours * n_features)
    """
    n_samples, lookback_hours, n_features = X.shape
    X_flat = X.reshape(n_samples, lookback_hours * n_features)
    
    print(f"Flattened sequences for linear regression: {X_flat.shape}")
    return X_flat

def create_target_names(target_cols, forecast_horizons):
    """Create descriptive names for multi-horizon targets."""
    target_names = []
    for horizon in forecast_horizons:
        for target_col in target_cols:
            target_names.append(f"{target_col}_{horizon}h")
    return target_names

def process_station_sequences(station_id, lookback_hours=24, forecast_horizons=[1, 3, 6], 
                            data_dir="/Users/maxpintchouk/Code/DeepLearning/Surf-Forecast/data"):
    """
    Process a station's data into sequences and save train/val/test splits.
    
    Args:
        station_id: Station ID (e.g., '46012')
        lookback_hours: Hours of historical data for input
        forecast_horizons: Hours ahead to predict
        data_dir: Base data directory
    
    Returns:
        Dictionary with sequence statistics and file paths
    """
    
    print(f"\n{'='*60}")
    print(f"Creating Sequences for Station {station_id}")
    print(f"{'='*60}")
    
    # Load splits
    splits_dir = Path(data_dir) / "splits" / station_id
    
    # Load feature summary to get column information
    with open(splits_dir / "feature_summary.json", 'r') as f:
        import json
        feature_summary = json.load(f)
    
    # Define feature and target columns
    all_cols = feature_summary['feature_columns']
    target_cols = ['WVHT', 'DPD']
    
    # Remove datetime and target columns from features
    feature_cols = [col for col in all_cols if col not in ['datetime'] + target_cols]
    
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Target columns: {target_cols}")
    
    # Process each split
    sequences_dir = splits_dir / "sequences"
    sequences_dir.mkdir(exist_ok=True)
    
    split_stats = {}
    scalers = {}
    
    for split_name in ['train', 'val', 'test']:
        print(f"\nProcessing {split_name} split...")
        
        # Load split data
        df = pd.read_csv(splits_dir / f"{split_name}.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Check for missing values in key columns
        missing_targets = df[target_cols].isnull().sum()
        missing_features = df[feature_cols].isnull().sum().sum()
        
        if missing_targets.sum() > 0:
            print(f"Warning: Missing target values in {split_name}:")
            for col, count in missing_targets.items():
                if count > 0:
                    print(f"  {col}: {count} missing")
        
        if missing_features > 0:
            print(f"Warning: {missing_features} missing feature values in {split_name}")
            # Fill missing features with forward fill then backward fill
            df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
            # Fill any remaining NaN with median
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        # Also handle missing target values
        df[target_cols] = df[target_cols].fillna(method='ffill').fillna(method='bfill')
        df[target_cols] = df[target_cols].fillna(df[target_cols].median())
        
        # Create sequences
        try:
            X, y, valid_indices = create_sequences(
                df, feature_cols, target_cols, 
                lookback_hours, forecast_horizons
            )
            
            # Scale features (fit scaler on train split only)
            if split_name == 'train':
                # Fit scaler on flattened training data
                X_flat = prepare_linear_regression_data(X)
                scaler = StandardScaler()
                X_scaled_flat = scaler.fit_transform(X_flat)
                # Reshape back to sequences
                X_scaled = X_scaled_flat.reshape(X.shape)
                scalers['feature_scaler'] = scaler
                
                # Fit target scaler
                target_scaler = StandardScaler()
                y_scaled = target_scaler.fit_transform(y)
                scalers['target_scaler'] = target_scaler
                
            else:
                # Transform using fitted scalers
                X_flat = prepare_linear_regression_data(X)
                X_scaled_flat = scalers['feature_scaler'].transform(X_flat)
                X_scaled = X_scaled_flat.reshape(X.shape)
                y_scaled = scalers['target_scaler'].transform(y)
            
            # Save sequences
            np.save(sequences_dir / f"X_{split_name}.npy", X_scaled)
            np.save(sequences_dir / f"y_{split_name}.npy", y_scaled)
            np.save(sequences_dir / f"indices_{split_name}.npy", valid_indices)
            
            # Also save flattened version for linear regression
            X_flat_scaled = prepare_linear_regression_data(X_scaled)
            np.save(sequences_dir / f"X_{split_name}_flat.npy", X_flat_scaled)
            
            split_stats[split_name] = {
                'n_samples': len(X),
                'sequence_shape': X.shape,
                'target_shape': y.shape,
                'date_range': {
                    'start': df['datetime'].iloc[valid_indices[0]].isoformat(),
                    'end': df['datetime'].iloc[valid_indices[-1]].isoformat()
                }
            }
            
            print(f"  ✓ Saved {len(X)} sequences")
            
        except Exception as e:
            print(f"  ❌ Error creating sequences: {e}")
            continue
    
    # Save scalers
    joblib.dump(scalers['feature_scaler'], sequences_dir / "feature_scaler.pkl")
    joblib.dump(scalers['target_scaler'], sequences_dir / "target_scaler.pkl")
    
    # Save metadata
    target_names = create_target_names(target_cols, forecast_horizons)
    
    metadata = {
        'station_id': station_id,
        'lookback_hours': lookback_hours,
        'forecast_horizons': forecast_horizons,
        'feature_columns': feature_cols,
        'target_columns': target_cols,
        'target_names': target_names,
        'n_features': len(feature_cols),
        'n_targets': len(target_cols),
        'n_horizons': len(forecast_horizons),
        'splits': split_stats
    }
    
    with open(sequences_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Sequences saved to {sequences_dir}")
    print(f"  Files: X_{{split}}.npy, y_{{split}}.npy, X_{{split}}_flat.npy")
    print(f"  Scalers: feature_scaler.pkl, target_scaler.pkl")
    print(f"  Metadata: metadata.json")
    
    return metadata

def main():
    """Process all stations to create sequence data."""
    stations = ['46012', '46221', '46026']
    
    for station_id in stations:
        try:
            metadata = process_station_sequences(station_id)
            
            print(f"\n✓ Station {station_id} sequences complete!")
            print(f"  Features: {metadata['n_features']}")
            print(f"  Lookback: {metadata['lookback_hours']} hours")
            print(f"  Horizons: {metadata['forecast_horizons']}")
            
            # Print split info
            for split_name, stats in metadata['splits'].items():
                print(f"  {split_name.capitalize()}: {stats['n_samples']:,} sequences")
                
        except Exception as e:
            print(f"\n❌ Error processing station {station_id}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Sequence generation complete!")
    print("Ready for model training.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()