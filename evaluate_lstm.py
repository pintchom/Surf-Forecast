import numpy as np
import pandas as pd
from pathlib import Path
import json
from src.models.lstm_model import LSTMSurfModel
from src.evaluation.metrics import evaluate_model_comprehensive, print_metrics_summary

def load_and_evaluate_lstm(station_id):
    """Load trained LSTM model and evaluate on test set."""
    
    print(f"\n=== Evaluating LSTM for Station {station_id} ===")
    
    # Load test data
    data_path = Path(f"data/splits/{station_id}/sequences")
    X_test = np.load(data_path / 'X_test.npy')
    y_test = np.load(data_path / 'y_test.npy')
    
    # Clean NaN values (same as training)
    if np.isnan(X_test).sum() > 0:
        print("Cleaning NaN values in test data...")
        X_test = np.nan_to_num(X_test, nan=0.0)
    
    print(f"Test data shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Load trained model
    model_path = Path(f"models/lstm/{station_id}")
    if not model_path.exists():
        print(f"No trained model found at {model_path}")
        return None
        
    model = LSTMSurfModel.load_model(model_path)
    print("Model loaded successfully!")
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    target_names = ['WVHT_1h', 'DPD_1h', 'WVHT_3h', 'DPD_3h', 'WVHT_6h', 'DPD_6h']
    metrics_result = evaluate_model_comprehensive(
        y_true=y_test,
        y_pred=y_pred,
        target_names=target_names,
        model_name=f"LSTM-{station_id}"
    )
    
    # Extract metrics from the returned structure
    if isinstance(metrics_result, dict) and 'primary_metrics' in metrics_result:
        metrics = metrics_result['primary_metrics']
    else:
        metrics = metrics_result
    
    # Load baseline results for comparison
    baseline_path = data_path / 'enhanced_baseline_results.json'
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)
            
        # Get best baseline RMSE
        baseline_rmse = baseline_results['best_model']['val_rmse']
        best_baseline_name = baseline_results['best_model']['name']
        
        # Calculate improvement
        lstm_rmse = metrics['overall']['rmse']
        improvement = ((baseline_rmse - lstm_rmse) / baseline_rmse) * 100
        
        print(f"\n=== Performance Comparison ===")
        print(f"Best Baseline ({best_baseline_name}): RMSE = {baseline_rmse:.4f}")
        print(f"LSTM: RMSE = {lstm_rmse:.4f}")
        print(f"Improvement: {improvement:.1f}%")
        
        metrics['baseline_comparison'] = {
            'baseline_model': best_baseline_name,
            'baseline_rmse': baseline_rmse,
            'lstm_rmse': lstm_rmse,
            'improvement_percent': improvement
        }
    
    # Save LSTM results
    results_path = data_path / 'lstm_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                metrics_json[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                   for k, v in value.items()}
            else:
                metrics_json[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
        
        json.dump(metrics_json, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    return metrics

def compare_stations():
    """Compare LSTM performance across both stations."""
    
    stations = ['46012', '46221']
    results = {}
    
    for station in stations:
        try:
            metrics = load_and_evaluate_lstm(station)
            if metrics:
                results[station] = metrics
        except Exception as e:
            print(f"Error evaluating station {station}: {e}")
            continue
    
    if len(results) == 2:
        print(f"\n=== Station Comparison ===")
        for station, metrics in results.items():
            print(f"\nStation {station}:")
            print(f"  Overall RMSE: {metrics['overall']['rmse']:.4f}")
            print(f"  WVHT_1h RMSE: {metrics['by_target']['WVHT_1h']['rmse']:.4f}")
            print(f"  WVHT_6h RMSE: {metrics['by_target']['WVHT_6h']['rmse']:.4f}")
            if 'baseline_comparison' in metrics:
                print(f"  vs Baseline: {metrics['baseline_comparison']['improvement_percent']:.1f}% improvement")
    
    return results

if __name__ == "__main__":
    results = compare_stations()