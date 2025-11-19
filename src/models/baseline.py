import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))
from evaluation import evaluate_model_comprehensive, print_metrics_summary

class PersistenceModel:
    """
    Naive baseline: predict t+n = t (current value persists)
    """
    
    def __init__(self, forecast_horizons=[1, 3, 6]):
        self.forecast_horizons = forecast_horizons
        self.target_cols = ['WVHT', 'DPD']
        
    def fit(self, X, y):
        """No training needed for persistence model"""
        pass
    
    def predict(self, X):
        """
        Predict by repeating current values (last timestep of sequence).
        
        Args:
            X: Input sequences (n_samples, lookback_hours, n_features)
            
        Returns:
            Predictions (n_samples, n_horizons * n_targets)
        """
        # Get current WVHT and DPD values (last timestep)
        # Assuming WVHT and DPD are the first 2 features
        current_values = X[:, -1, :2]  # Shape: (n_samples, 2)
        
        # Repeat for each forecast horizon
        n_samples = X.shape[0]
        n_targets = len(self.target_cols)
        n_horizons = len(self.forecast_horizons)
        
        predictions = np.zeros((n_samples, n_horizons * n_targets))
        
        for i, horizon in enumerate(self.forecast_horizons):
            start_idx = i * n_targets
            end_idx = start_idx + n_targets
            predictions[:, start_idx:end_idx] = current_values
            
        return predictions

class LinearRegressionModel:
    """
    Multi-output Ridge regression for multi-horizon forecasting
    """
    
    def __init__(self, alpha=1.0, forecast_horizons=[1, 3, 6]):
        self.alpha = alpha
        self.forecast_horizons = forecast_horizons
        self.target_cols = ['WVHT', 'DPD']
        self.model = None
        
    def fit(self, X_flat, y):
        """
        Train Ridge regression model.
        
        Args:
            X_flat: Flattened input features (n_samples, lookback_hours * n_features)
            y: Multi-horizon targets (n_samples, n_horizons * n_targets)
        """
        print(f"Training Ridge regression with alpha={self.alpha}")
        print(f"  Input shape: {X_flat.shape}")
        print(f"  Target shape: {y.shape}")
        
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_flat, y)
        
        print("  ✓ Training complete")
        
    def predict(self, X_flat):
        """Make predictions using trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X_flat)
    
    def save(self, filepath):
        """Save trained model."""
        joblib.dump(self.model, filepath)
        
    def load(self, filepath):
        """Load trained model."""
        self.model = joblib.load(filepath)

# Removed - now using evaluation module functions

def train_baseline_models(station_id, data_dir="/Users/maxpintchouk/Code/DeepLearning/Surf-Forecast/data"):
    """
    Train and evaluate baseline models for a station.
    
    Args:
        station_id: Station ID (e.g., '46012')
        data_dir: Base data directory
        
    Returns:
        Dictionary with all results
    """
    
    print(f"\n{'='*60}")
    print(f"Training Baseline Models - Station {station_id}")
    print(f"{'='*60}")
    
    # Load sequence data
    sequences_dir = Path(data_dir) / "splits" / station_id / "sequences"
    
    # Load metadata and scalers
    with open(sequences_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load target scaler for inverse transformation
    target_scaler = joblib.load(sequences_dir / "target_scaler.pkl")
    
    target_names = metadata['target_names']
    print(f"Target variables: {target_names}")
    
    # Load training data
    print("\nLoading training data...")
    X_train = np.load(sequences_dir / "X_train.npy")
    X_train_flat = np.load(sequences_dir / "X_train_flat.npy") 
    y_train = np.load(sequences_dir / "y_train.npy")
    
    # Load validation data
    print("Loading validation data...")
    X_val = np.load(sequences_dir / "X_val.npy")
    X_val_flat = np.load(sequences_dir / "X_val_flat.npy")
    y_val = np.load(sequences_dir / "y_val.npy")
    
    # Check for NaN values
    print("Checking for NaN values...")
    nan_counts = {
        'X_train': np.isnan(X_train).sum(),
        'X_train_flat': np.isnan(X_train_flat).sum(),
        'y_train': np.isnan(y_train).sum(),
        'X_val': np.isnan(X_val).sum(),
        'X_val_flat': np.isnan(X_val_flat).sum(),
        'y_val': np.isnan(y_val).sum()
    }
    
    for data_name, nan_count in nan_counts.items():
        if nan_count > 0:
            print(f"  ⚠️  {data_name}: {nan_count} NaN values")
        else:
            print(f"  ✓ {data_name}: No NaN values")
    
    # Handle any remaining NaN values
    if nan_counts['X_train_flat'] > 0 or nan_counts['y_train'] > 0:
        print("Filling remaining NaN values with median...")
        X_train_flat = np.nan_to_num(X_train_flat, nan=np.nanmedian(X_train_flat))
        y_train = np.nan_to_num(y_train, nan=np.nanmedian(y_train))
        
    if nan_counts['X_val_flat'] > 0 or nan_counts['y_val'] > 0:
        X_val_flat = np.nan_to_num(X_val_flat, nan=np.nanmedian(X_val_flat))
        y_val = np.nan_to_num(y_val, nan=np.nanmedian(y_val))
        
    # Update sequence data with cleaned versions
    X_train = np.nan_to_num(X_train, nan=np.nanmedian(X_train))
    X_val = np.nan_to_num(X_val, nan=np.nanmedian(X_val))
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Feature dimensions: {X_train.shape}")
    print(f"Flattened dimensions: {X_train_flat.shape}")
    
    results = {}
    
    # 1. Persistence Model
    print(f"\n{'-'*30}")
    print("Training Persistence Model")
    print(f"{'-'*30}")
    
    persistence_model = PersistenceModel(metadata['forecast_horizons'])
    persistence_model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred_persistence = persistence_model.predict(X_val)
    
    # Inverse transform for surf metrics calculation
    y_val_original = target_scaler.inverse_transform(y_val)
    y_pred_persistence_original = target_scaler.inverse_transform(y_pred_persistence)
    
    persistence_results = evaluate_model_comprehensive(
        y_val_original, y_pred_persistence_original, target_names, "Persistence"
    )
    print_metrics_summary(persistence_results)
    
    results['persistence'] = persistence_results
    
    # 2. Ridge Regression Model
    print(f"\n{'-'*30}")
    print("Training Ridge Regression Model")
    print(f"{'-'*30}")
    
    # Try different alpha values
    alphas = [0.1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_score = float('inf')
    best_model = None
    
    for alpha in alphas:
        ridge_model = LinearRegressionModel(alpha=alpha, forecast_horizons=metadata['forecast_horizons'])
        ridge_model.fit(X_train_flat, y_train)
        
        y_pred_ridge = ridge_model.predict(X_val_flat)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_ridge))
        
        print(f"  Alpha {alpha:6.1f}: Validation RMSE = {val_rmse:.4f}")
        
        if val_rmse < best_score:
            best_score = val_rmse
            best_alpha = alpha
            best_model = ridge_model
    
    print(f"\nBest alpha: {best_alpha} (RMSE: {best_score:.4f})")
    
    # Evaluate best model
    y_pred_ridge = best_model.predict(X_val_flat)
    
    # Inverse transform for surf metrics calculation
    y_pred_ridge_original = target_scaler.inverse_transform(y_pred_ridge)
    
    ridge_results = evaluate_model_comprehensive(
        y_val_original, y_pred_ridge_original, target_names, f"Ridge (α={best_alpha})",
        baseline_metrics=persistence_results['primary_metrics']
    )
    print_metrics_summary(ridge_results)
    
    results['ridge'] = ridge_results
    results['ridge']['best_alpha'] = best_alpha
    
    # Save best model
    model_dir = sequences_dir / "models"
    model_dir.mkdir(exist_ok=True)
    best_model.save(model_dir / "ridge_best.pkl")
    
    # Model comparison
    print(f"\n{'-'*30}")
    print("Model Comparison")
    print(f"{'-'*30}")
    
    persistence_rmse = persistence_results['primary_metrics']['overall']['rmse']
    ridge_rmse = ridge_results['primary_metrics']['overall']['rmse']
    improvement = ((persistence_rmse - ridge_rmse) / persistence_rmse) * 100
    
    print(f"Persistence RMSE: {persistence_rmse:.4f}")
    print(f"Ridge RMSE:       {ridge_rmse:.4f}")
    print(f"Improvement:      {improvement:.1f}%")
    
    if improvement >= 20:
        print("✅ Ridge model meets 20% improvement target!")
    else:
        print("❌ Ridge model does not meet 20% improvement target")
    
    results['comparison'] = {
        'persistence_rmse': persistence_rmse,
        'ridge_rmse': ridge_rmse,
        'improvement_pct': improvement
    }
    
    # Save results
    results_path = sequences_dir / "baseline_results.json"
    
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert numpy types before saving
    results_serializable = convert_numpy_types(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    
    return results

def main():
    """Train baseline models for all stations."""
    stations = ['46012', '46221']
    
    all_results = {}
    
    for station_id in stations:
        try:
            results = train_baseline_models(station_id)
            all_results[station_id] = results
            
            print(f"\n✓ Station {station_id} baseline training complete!")
            
        except Exception as e:
            print(f"\n❌ Error training baselines for station {station_id}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("Baseline Model Training Summary")
    print(f"{'='*60}")
    
    for station_id, results in all_results.items():
        print(f"\nStation {station_id}:")
        improvement = results['ridge']['comparative_metrics']['overall_improvement']['improvement_pct']
        print(f"  Ridge improvement over persistence: {improvement:.1f}%")
        
        if improvement >= 20:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"  20% improvement target: {status}")

if __name__ == "__main__":
    main()