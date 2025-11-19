import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))
from evaluation import evaluate_model_comprehensive, print_metrics_summary, ModelComparison

class SurfForecastingModel:
    """
    Wrapper class for trained surf forecasting models.
    """
    
    def __init__(self, station_id, model_name="enhanced_best", data_dir="/Users/maxpintchouk/Code/DeepLearning/Surf-Forecast/data"):
        self.station_id = station_id
        self.model_name = model_name
        self.sequences_dir = Path(data_dir) / "splits" / station_id / "sequences"
        
        # Load model and scalers
        self.model = None
        self.target_scaler = None
        self.feature_scaler = None
        self.metadata = None
        
        self._load_model_and_scalers()
        
    def _load_model_and_scalers(self):
        """Load the trained model and preprocessing scalers."""
        
        # Load metadata
        metadata_path = self.sequences_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load scalers
        self.target_scaler = joblib.load(self.sequences_dir / "target_scaler.pkl")
        self.feature_scaler = joblib.load(self.sequences_dir / "feature_scaler.pkl")
        
        # Load model
        model_path = self.sequences_dir / "models" / f"{self.model_name}.pkl"
        if model_path.exists():
            self.model = joblib.load(model_path)
            print(f"✓ Loaded model: {model_path}")
        else:
            available_models = list((self.sequences_dir / "models").glob("*.pkl"))
            available_names = [m.stem for m in available_models if 'scaler' not in m.name]
            raise FileNotFoundError(f"Model {model_path} not found. Available models: {available_names}")
    
    def predict(self, X_sequences):
        """
        Make predictions on sequence data.
        
        Args:
            X_sequences: Input sequences (n_samples, lookback_hours, n_features) or pre-flattened
            
        Returns:
            predictions: Denormalized predictions in original units
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Handle sequence vs flattened input
        if len(X_sequences.shape) == 3:
            # Flatten sequences for linear models
            n_samples, lookback, n_features = X_sequences.shape
            X_flat = X_sequences.reshape(n_samples, lookback * n_features)
        else:
            X_flat = X_sequences
        
        # Make predictions (normalized)
        predictions_scaled = self.model.predict(X_flat)
        
        # Denormalize to original units
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        
        return predictions
    
    def get_target_names(self):
        """Get the names of prediction targets."""
        return self.metadata['target_names']
    
    def get_feature_info(self):
        """Get information about input features."""
        return {
            'n_features': self.metadata['n_features'],
            'lookback_hours': self.metadata['lookback_hours'],
            'forecast_horizons': self.metadata['forecast_horizons'],
            'feature_columns': self.metadata['feature_columns'][:10]  # Show first 10
        }

def load_test_data(station_id, data_dir="/Users/maxpintchouk/Code/DeepLearning/Surf-Forecast/data"):
    """
    Load test set data for evaluation.
    
    Returns:
        X_test: Test input sequences
        y_test: Test target values (denormalized)
        timestamps: Test timestamps for plotting
    """
    
    sequences_dir = Path(data_dir) / "splits" / station_id / "sequences"
    
    # Load test sequences
    X_test = np.load(sequences_dir / "X_test.npy")
    y_test_scaled = np.load(sequences_dir / "y_test.npy")
    test_indices = np.load(sequences_dir / "indices_test.npy")
    
    # Load target scaler for denormalization
    target_scaler = joblib.load(sequences_dir / "target_scaler.pkl")
    y_test = target_scaler.inverse_transform(y_test_scaled)
    
    # Load original test data to get timestamps
    test_df = pd.read_csv(sequences_dir.parent / "test.csv")
    test_df['datetime'] = pd.to_datetime(test_df['datetime'])
    
    # Get timestamps corresponding to test indices
    timestamps = test_df['datetime'].iloc[test_indices].values
    
    print(f"✓ Loaded test data: {len(X_test)} samples")
    print(f"  Date range: {timestamps[0]} to {timestamps[-1]}")
    
    return X_test, y_test, timestamps

def evaluate_on_test_set(station_id, model_names=None, data_dir="/Users/maxpintchouk/Code/DeepLearning/Surf-Forecast/data"):
    """
    Evaluate trained models on the test set.
    
    Args:
        station_id: Station to evaluate
        model_names: List of model names to evaluate (None = all available)
        data_dir: Data directory path
        
    Returns:
        Dictionary with evaluation results
    """
    
    print(f"\n{'='*60}")
    print(f"Test Set Evaluation - Station {station_id}")
    print(f"{'='*60}")
    
    sequences_dir = Path(data_dir) / "splits" / station_id / "sequences"
    models_dir = sequences_dir / "models"
    
    # Find available models
    if model_names is None:
        available_models = [f.stem for f in models_dir.glob("*.pkl") if 'scaler' not in f.name]
        print(f"Available models: {available_models}")
        model_names = available_models
    
    # Load test data
    X_test, y_test, timestamps = load_test_data(station_id, data_dir)
    
    # Load metadata for target names
    with open(sequences_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    target_names = metadata['target_names']
    
    # Initialize model comparison
    comparison = ModelComparison()
    results = {}
    
    for model_name in model_names:
        print(f"\n{'-'*40}")
        print(f"Evaluating {model_name}")
        print(f"{'-'*40}")
        
        try:
            # Load model
            model = SurfForecastingModel(station_id, model_name, data_dir)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Comprehensive evaluation
            model_results = evaluate_model_comprehensive(
                y_test, y_pred, target_names, model_name
            )
            
            print_metrics_summary(model_results)
            
            # Add to comparison
            comparison.add_model(model_name, y_pred, y_test, target_names, 
                               model_info={'type': 'baseline'})
            
            results[model_name] = model_results
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
    
    # Model comparison summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Model Comparison on Test Set")
        print(f"{'='*60}")
        
        comparison.print_comparison_summary()
        
        # Save comparison results
        comparison_path = sequences_dir / "test_set_comparison.json"
        comparison_report = comparison.create_comparison_report(comparison_path)
    
    return results

def quick_prediction_demo(station_id, model_name="enhanced_best", n_samples=5):
    """
    Quick demo showing model predictions on recent test data.
    
    Args:
        station_id: Station ID
        model_name: Model to use
        n_samples: Number of samples to show
    """
    
    print(f"\n{'='*60}")
    print(f"Quick Prediction Demo - {model_name}")
    print(f"{'='*60}")
    
    # Load model
    model = SurfForecastingModel(station_id, model_name)
    
    # Load test data
    X_test, y_test, timestamps = load_test_data(station_id)
    
    # Get target names
    target_names = model.get_target_names()
    
    # Sample random predictions
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    print(f"Showing {n_samples} random predictions:\n")
    
    for i, idx in enumerate(sample_indices):
        timestamp = timestamps[idx]
        
        # Make prediction for single sample
        X_sample = X_test[idx:idx+1]  # Keep batch dimension
        y_pred_sample = model.predict(X_sample)[0]  # Remove batch dimension
        y_true_sample = y_test[idx]
        
        print(f"Sample {i+1} - {timestamp}")
        print("-" * 40)
        
        for j, target_name in enumerate(target_names):
            true_val = y_true_sample[j]
            pred_val = y_pred_sample[j]
            error = abs(pred_val - true_val)
            
            if 'WVHT' in target_name:
                unit = 'm'
            else:
                unit = 's'
            
            print(f"  {target_name:<8}: True={true_val:6.2f}{unit}, Pred={pred_val:6.2f}{unit}, Error={error:6.2f}{unit}")
        
        print()

def main():
    """Interactive model testing interface."""
    
    stations = ['46012', '46221']
    
    print("Surf Forecasting Model Testing Interface")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Evaluate all models on test set")
        print("2. Quick prediction demo")
        print("3. Model info")
        print("4. Exit")
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                station = input(f"Enter station ID ({'/'.join(stations)}): ").strip()
                if station in stations:
                    results = evaluate_on_test_set(station)
                    print(f"\n✓ Test evaluation complete for station {station}")
                else:
                    print(f"Invalid station. Available: {stations}")
            
            elif choice == '2':
                station = input(f"Enter station ID ({'/'.join(stations)}): ").strip()
                if station in stations:
                    # List available models
                    sequences_dir = Path("/Users/maxpintchouk/Code/DeepLearning/Surf-Forecast/data") / "splits" / station / "sequences"
                    models_dir = sequences_dir / "models"
                    available = [f.stem for f in models_dir.glob("*.pkl") if 'scaler' not in f.name]
                    
                    print(f"Available models: {available}")
                    model_name = input("Enter model name (or press Enter for enhanced_best): ").strip()
                    if not model_name:
                        model_name = "enhanced_best"
                    
                    if f"{model_name}.pkl" in [f.name for f in models_dir.glob("*.pkl")]:
                        quick_prediction_demo(station, model_name)
                    else:
                        print(f"Model {model_name} not found")
                else:
                    print(f"Invalid station. Available: {stations}")
            
            elif choice == '3':
                station = input(f"Enter station ID ({'/'.join(stations)}): ").strip()
                if station in stations:
                    try:
                        model = SurfForecastingModel(station)
                        info = model.get_feature_info()
                        print(f"\nModel Info for Station {station}:")
                        print(f"  Features: {info['n_features']}")
                        print(f"  Lookback: {info['lookback_hours']} hours")
                        print(f"  Horizons: {info['forecast_horizons']}")
                        print(f"  Targets: {model.get_target_names()}")
                        print(f"  Sample features: {info['feature_columns'][:5]}...")
                    except Exception as e:
                        print(f"Error loading model: {e}")
                else:
                    print(f"Invalid station. Available: {stations}")
            
            elif choice == '4':
                print("Goodbye!")
                break
            
            else:
                print("Invalid option. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # For direct script execution, run test evaluation
    if len(sys.argv) > 1:
        station_id = sys.argv[1]
        evaluate_on_test_set(station_id)
    else:
        main()