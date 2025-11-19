import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))
from evaluation import evaluate_model_comprehensive, print_metrics_summary

def enhanced_linear_model_search(X_train_flat, y_train, X_val_flat, y_val, target_names):
    """
    Extended hyperparameter search across multiple linear models.
    
    Args:
        X_train_flat: Training features (n_samples, n_features)
        y_train: Training targets (n_samples, n_targets)
        X_val_flat: Validation features
        y_val: Validation targets
        target_names: List of target names
        
    Returns:
        Dictionary with best model results
    """
    
    print(f"\n{'='*60}")
    print("Enhanced Linear Model Search")
    print(f"{'='*60}")
    
    models_to_try = [
        # Extended Ridge search
        ('Ridge', Ridge, {'alpha': np.logspace(-3, 3, 15)}),  # 0.001 to 1000
        
        # ElasticNet (L1 + L2 regularization)
        ('ElasticNet', ElasticNet, {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Mix of L1 and L2
        }),
        
        # Lasso (L1 regularization - feature selection)
        ('Lasso', Lasso, {'alpha': [0.001, 0.01, 0.1, 1.0]})
        
        # RandomForest removed - too slow for this dataset size
    ]
    
    best_overall = {
        'model_name': None,
        'model': None,
        'params': None,
        'val_rmse': float('inf'),
        'results': None
    }
    
    all_results = {}
    
    for model_name, model_class, param_grid in models_to_try:
        print(f"\n{'-'*40}")
        print(f"Testing {model_name}")
        print(f"{'-'*40}")
        
        # Generate parameter combinations
        if model_name == 'ElasticNet':
            # Special handling for ElasticNet with two parameters
            param_combinations = []
            for alpha in param_grid['alpha']:
                for l1_ratio in param_grid['l1_ratio']:
                    param_combinations.append({'alpha': alpha, 'l1_ratio': l1_ratio})
        else:
            # Single parameter models (Ridge, Lasso)
            param_combinations = [{'alpha': alpha} for alpha in param_grid['alpha']]
        
        best_for_model = {
            'params': None,
            'val_rmse': float('inf'),
            'model': None
        }
        
        # Try each parameter combination
        for params in param_combinations:
            try:
                # Create and train model
                model = model_class(**params)
                model.fit(X_train_flat, y_train)
                
                # Validate
                y_pred = model.predict(X_val_flat)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                
                # Print progress
                param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
                print(f"  {param_str}: RMSE = {val_rmse:.4f}")
                
                # Track best for this model type
                if val_rmse < best_for_model['val_rmse']:
                    best_for_model['val_rmse'] = val_rmse
                    best_for_model['params'] = params.copy()
                    best_for_model['model'] = model
                
                # Track best overall
                if val_rmse < best_overall['val_rmse']:
                    best_overall['val_rmse'] = val_rmse
                    best_overall['model_name'] = model_name
                    best_overall['params'] = params.copy()
                    best_overall['model'] = model
                    
            except Exception as e:
                print(f"  Error with {params}: {e}")
                continue
        
        # Store best result for this model type
        if best_for_model['model'] is not None:
            print(f"\n✓ Best {model_name}: {best_for_model['params']}")
            print(f"  Validation RMSE: {best_for_model['val_rmse']:.4f}")
            
            all_results[model_name] = {
                'params': best_for_model['params'],
                'val_rmse': best_for_model['val_rmse'],
                'model': best_for_model['model']
            }
    
    print(f"\n{'='*60}")
    print("Model Comparison Summary")
    print(f"{'='*60}")
    
    # Sort results by validation RMSE
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['val_rmse'])
    
    for i, (model_name, result) in enumerate(sorted_results):
        rank = i + 1
        rmse = result['val_rmse']
        params = result['params']
        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        print(f"{rank}. {model_name:<15}: RMSE = {rmse:.4f} ({param_str})")
    
    return best_overall, all_results

def train_enhanced_baseline_models(station_id, data_dir="/Users/maxpintchouk/Code/DeepLearning/Surf-Forecast/data"):
    """
    Train enhanced baseline models with extended search.
    """
    
    print(f"\n{'='*60}")
    print(f"Enhanced Baseline Training - Station {station_id}")
    print(f"{'='*60}")
    
    # Load sequence data (same as before)
    sequences_dir = Path(data_dir) / "splits" / station_id / "sequences"
    
    # Load metadata and scalers
    with open(sequences_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    target_scaler = joblib.load(sequences_dir / "target_scaler.pkl")
    target_names = metadata['target_names']
    
    # Load data
    print("\nLoading data...")
    X_train = np.load(sequences_dir / "X_train.npy")
    X_train_flat = np.load(sequences_dir / "X_train_flat.npy") 
    y_train = np.load(sequences_dir / "y_train.npy")
    
    X_val = np.load(sequences_dir / "X_val.npy")
    X_val_flat = np.load(sequences_dir / "X_val_flat.npy")
    y_val = np.load(sequences_dir / "y_val.npy")
    
    # Handle NaN values (same as before)
    print("Checking for NaN values...")
    X_train_flat = np.nan_to_num(X_train_flat, nan=np.nanmedian(X_train_flat))
    X_val_flat = np.nan_to_num(X_val_flat, nan=np.nanmedian(X_val_flat))
    y_train = np.nan_to_num(y_train, nan=np.nanmedian(y_train))
    y_val = np.nan_to_num(y_val, nan=np.nanmedian(y_val))
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Features: {X_train_flat.shape[1]}")
    
    # Enhanced model search
    best_model_info, all_results = enhanced_linear_model_search(
        X_train_flat, y_train, X_val_flat, y_val, target_names
    )
    
    print(f"\n{'='*60}")
    print("Best Model Evaluation")
    print(f"{'='*60}")
    
    # Evaluate best model with comprehensive metrics
    best_model = best_model_info['model']
    y_pred_best = best_model.predict(X_val_flat)
    
    # Inverse transform for surf metrics
    y_val_original = target_scaler.inverse_transform(y_val)
    y_pred_best_original = target_scaler.inverse_transform(y_pred_best)
    
    # Comprehensive evaluation
    best_results = evaluate_model_comprehensive(
        y_val_original, y_pred_best_original, target_names, 
        f"{best_model_info['model_name']} (Best)"
    )
    print_metrics_summary(best_results)
    
    # Save best model
    model_dir = sequences_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    best_model_path = model_dir / "enhanced_best.pkl"
    joblib.dump(best_model, best_model_path)
    
    # Save results
    results = {
        'best_model': {
            'name': best_model_info['model_name'],
            'params': best_model_info['params'],
            'val_rmse': best_model_info['val_rmse'],
            'evaluation': best_results
        },
        'all_models': {}
    }
    
    # Add evaluation for all models
    for model_name, model_info in all_results.items():
        model = model_info['model']
        y_pred = model.predict(X_val_flat)
        y_pred_original = target_scaler.inverse_transform(y_pred)
        
        model_results = evaluate_model_comprehensive(
            y_val_original, y_pred_original, target_names, model_name
        )
        
        results['all_models'][model_name] = {
            'params': model_info['params'],
            'val_rmse': model_info['val_rmse'],
            'evaluation': model_results
        }
    
    # Convert numpy types and save
    def convert_numpy_types(obj):
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
    
    results_serializable = convert_numpy_types(results)
    
    results_path = sequences_dir / "enhanced_baseline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Enhanced baseline results saved to {results_path}")
    print(f"✓ Best model saved to {best_model_path}")
    
    return results

def main():
    """Train enhanced baseline models for all stations."""
    stations = ['46012', '46221']
    
    all_results = {}
    
    for station_id in stations:
        try:
            results = train_enhanced_baseline_models(station_id)
            all_results[station_id] = results
            
            best_name = results['best_model']['name']
            best_rmse = results['best_model']['val_rmse']
            print(f"\n✓ Station {station_id}: Best model is {best_name} (RMSE: {best_rmse:.4f})")
            
        except Exception as e:
            print(f"\n❌ Error training enhanced baselines for station {station_id}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Enhanced Baseline Training Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()