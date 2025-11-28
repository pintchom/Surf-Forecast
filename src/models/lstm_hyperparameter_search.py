import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
import itertools
import json
from pathlib import Path
import time
import sys
sys.path.append(str(Path(__file__).parent.parent))
from evaluation import evaluate_model_comprehensive

class LSTMHyperparameterSearch:
    """
    Comprehensive hyperparameter search for LSTM surf forecasting model.
    """
    
    def __init__(self, station_id):
        self.station_id = station_id
        self.results = []
        self.best_model = None
        self.best_params = None
        self.best_score = float('inf')
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load training and validation data."""
        data_path = Path(f"data/splits/{self.station_id}/sequences")
        
        print(f"Loading data for station {self.station_id}...")
        self.X_train = np.load(data_path / 'X_train.npy')
        self.y_train = np.load(data_path / 'y_train.npy')
        self.X_val = np.load(data_path / 'X_val.npy')
        self.y_val = np.load(data_path / 'y_val.npy')
        
        # Clean NaN values
        if np.isnan(self.X_train).sum() > 0 or np.isnan(self.X_val).sum() > 0:
            print("Cleaning NaN values...")
            self.X_train = np.nan_to_num(self.X_train, nan=0.0)
            self.X_val = np.nan_to_num(self.X_val, nan=0.0)
        
        print(f"Data shapes: X_train: {self.X_train.shape}, y_train: {self.y_train.shape}")
    
    def create_model(self, params):
        """Create LSTM model with given parameters."""
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            params['lstm_units_1'], 
            return_sequences=len(params['lstm_layers']) > 1,
            input_shape=(self.X_train.shape[1], self.X_train.shape[2])
        ))
        model.add(Dropout(params['dropout_rate']))
        
        # Second LSTM layer (if specified)
        if len(params['lstm_layers']) > 1:
            model.add(LSTM(
                params['lstm_units_2'], 
                return_sequences=len(params['lstm_layers']) > 2
            ))
            model.add(Dropout(params['dropout_rate']))
        
        # Third LSTM layer (if specified) 
        if len(params['lstm_layers']) > 2:
            model.add(LSTM(params['lstm_units_3'], return_sequences=False))
            model.add(Dropout(params['dropout_rate']))
        
        # Dense layers
        if params['dense_layers'] >= 1:
            model.add(Dense(params['dense_units'], activation='relu'))
            model.add(Dropout(params['dropout_rate']))
        
        if params['dense_layers'] >= 2:
            model.add(Dense(params['dense_units'] // 2, activation='relu'))
        
        # Output layer
        model.add(Dense(6))  # 6 outputs: WVHT_1h, DPD_1h, WVHT_3h, DPD_3h, WVHT_6h, DPD_6h
        
        # Compile model
        if params['optimizer'] == 'adam':
            optimizer = Adam(learning_rate=params['learning_rate'], clipnorm=1.0)
        else:  # adamw
            optimizer = AdamW(learning_rate=params['learning_rate'], weight_decay=params['weight_decay'], clipnorm=1.0)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train_model(self, params, verbose=0):
        """Train model with given parameters."""
        
        model = self.create_model(params)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=params['patience'],
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=params['patience'] // 2,
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        # Train model
        start_time = time.time()
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=params['max_epochs'],
            batch_size=params['batch_size'],
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        training_time = time.time() - start_time
        
        # Get best validation loss
        best_val_loss = min(history.history['val_loss'])
        epochs_trained = len(history.history['loss'])
        
        return model, best_val_loss, epochs_trained, training_time
    
    def search(self, search_space=None, max_trials=20, verbose=1):
        """
        Perform hyperparameter search.
        
        Args:
            search_space: Dictionary of parameter ranges
            max_trials: Maximum number of configurations to try
            verbose: Verbosity level
        """
        
        if search_space is None:
            search_space = self.get_default_search_space()
        
        print(f"\n=== LSTM Hyperparameter Search for Station {self.station_id} ===")
        print(f"Search space combinations: {self.count_combinations(search_space)}")
        print(f"Max trials: {max_trials}")
        
        # Generate all parameter combinations
        param_combinations = self.generate_param_combinations(search_space)
        
        # Limit to max_trials
        if len(param_combinations) > max_trials:
            # Sample random combinations
            np.random.seed(42)
            indices = np.random.choice(len(param_combinations), max_trials, replace=False)
            param_combinations = [param_combinations[i] for i in indices]
            print(f"Randomly selected {max_trials} combinations from {len(param_combinations)} total")
        
        # Search through combinations
        for i, params in enumerate(param_combinations):
            if verbose >= 1:
                print(f"\nTrial {i+1}/{len(param_combinations)}")
                print(f"Params: {self.format_params(params)}")
            
            try:
                model, val_loss, epochs, train_time = self.train_model(params, verbose=0)
                
                # Record results
                result = {
                    'trial': i + 1,
                    'params': params.copy(),
                    'val_loss': float(val_loss),
                    'epochs_trained': int(epochs),
                    'training_time': float(train_time),
                    'success': True
                }
                
                self.results.append(result)
                
                # Update best model
                if val_loss < self.best_score:
                    self.best_score = val_loss
                    self.best_params = params.copy()
                    self.best_model = model
                    if verbose >= 1:
                        print(f"✅ New best! Val loss: {val_loss:.4f}")
                
                if verbose >= 1:
                    print(f"Val loss: {val_loss:.4f}, Epochs: {epochs}, Time: {train_time:.1f}s")
                
            except Exception as e:
                if verbose >= 1:
                    print(f"❌ Failed: {str(e)}")
                
                result = {
                    'trial': i + 1,
                    'params': params.copy(),
                    'error': str(e),
                    'success': False
                }
                self.results.append(result)
                continue
        
        # Print summary
        successful_trials = [r for r in self.results if r['success']]
        print(f"\n=== Search Complete ===")
        print(f"Successful trials: {len(successful_trials)}/{len(self.results)}")
        print(f"Best validation loss: {self.best_score:.4f}")
        print(f"Best parameters: {self.format_params(self.best_params)}")
        
        return self.best_model, self.best_params, self.results
    
    def get_default_search_space(self):
        """Default hyperparameter search space."""
        return {
            # Architecture parameters
            'lstm_layers': [[64], [64, 32], [128, 64], [128, 64, 32]],
            'dense_layers': [1, 2],
            'dense_units': [32, 64, 128],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4],
            
            # Training parameters  
            'learning_rate': [0.0001, 0.0005, 0.001, 0.002],
            'batch_size': [32, 64, 128],
            'optimizer': ['adam', 'adamw'],
            'weight_decay': [1e-4, 1e-3],  # Only for AdamW
            
            # Training control
            'max_epochs': [200],
            'patience': [20, 30]
        }
    
    def generate_param_combinations(self, search_space):
        """Generate all parameter combinations."""
        
        combinations = []
        
        # Get all combinations
        keys = list(search_space.keys())
        values = list(search_space.values())
        
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            
            # Set LSTM units based on layers
            lstm_layers = params['lstm_layers']
            params['lstm_units_1'] = lstm_layers[0] if len(lstm_layers) > 0 else 64
            params['lstm_units_2'] = lstm_layers[1] if len(lstm_layers) > 1 else 32  
            params['lstm_units_3'] = lstm_layers[2] if len(lstm_layers) > 2 else 16
            
            # Skip weight_decay for Adam optimizer
            if params['optimizer'] == 'adam':
                params['weight_decay'] = 0.0
            
            combinations.append(params)
        
        return combinations
    
    def count_combinations(self, search_space):
        """Count total combinations in search space."""
        count = 1
        for values in search_space.values():
            count *= len(values)
        return count
    
    def format_params(self, params):
        """Format parameters for display."""
        formatted = {}
        for key, value in params.items():
            if key == 'lstm_layers':
                formatted[key] = value
            elif key in ['learning_rate', 'weight_decay']:
                formatted[key] = f"{value:.1e}"
            elif key.startswith('lstm_units_') or key in ['dense_units', 'batch_size', 'max_epochs', 'patience', 'dense_layers']:
                formatted[key] = value
            elif key in ['dropout_rate']:
                formatted[key] = f"{value:.1f}"
            else:
                formatted[key] = value
        return formatted
    
    def evaluate_best_model(self):
        """Evaluate the best model on validation set."""
        if self.best_model is None:
            print("No best model found!")
            return None
        
        predictions = self.best_model.predict(self.X_val, verbose=0)
        target_names = ['WVHT_1h', 'DPD_1h', 'WVHT_3h', 'DPD_3h', 'WVHT_6h', 'DPD_6h']
        
        metrics_result = evaluate_model_comprehensive(
            y_true=self.y_val,
            y_pred=predictions,
            target_names=target_names,
            model_name="LSTM-Optimized"
        )
        
        if isinstance(metrics_result, dict) and 'primary_metrics' in metrics_result:
            metrics = metrics_result['primary_metrics']
        else:
            metrics = metrics_result
        
        return metrics
    
    def save_results(self):
        """Save search results to file."""
        results_path = Path(f"data/splits/{self.station_id}/sequences")
        
        # Save all results
        with open(results_path / 'lstm_hyperparameter_search.json', 'w') as f:
            json.dump({
                'station_id': self.station_id,
                'search_results': self.results,
                'best_params': self.best_params,
                'best_score': float(self.best_score),
                'total_trials': len(self.results),
                'successful_trials': len([r for r in self.results if r['success']])
            }, f, indent=2)
        
        # Save best model
        if self.best_model is not None:
            model_path = Path(f"models/lstm_optimized/{self.station_id}")
            model_path.mkdir(parents=True, exist_ok=True)
            self.best_model.save(str(model_path / 'model.keras'))
            
            # Save best config
            with open(model_path / 'config.json', 'w') as f:
                json.dump(self.best_params, f, indent=2)
        
        print(f"Results saved to {results_path}")


def run_hyperparameter_search(station_id, max_trials=15, verbose=1):
    """Run hyperparameter search for a station."""
    
    search = LSTMHyperparameterSearch(station_id)
    
    # Custom search space (focused on most promising parameters)
    focused_search_space = {
        'lstm_layers': [[64], [64, 32], [128, 64]],  # Avoid too deep
        'dense_layers': [1],  # Keep simple
        'dense_units': [64, 128], 
        'dropout_rate': [0.2, 0.3],  # Moderate regularization
        'learning_rate': [0.0001, 0.0005, 0.001],  # Conservative range
        'batch_size': [32, 64],  # Reasonable sizes
        'optimizer': ['adam'],  # Stick with Adam
        'weight_decay': [0.0],  # Not used with Adam
        'max_epochs': [200],
        'patience': [25]  # Sufficient patience
    }
    
    # Run search
    best_model, best_params, results = search.search(
        search_space=focused_search_space,
        max_trials=max_trials,
        verbose=verbose
    )
    
    # Evaluate best model
    if best_model is not None:
        print("\n=== Best Model Evaluation ===")
        metrics = search.evaluate_best_model()
        
        if metrics:
            print(f"Overall RMSE: {metrics['overall']['rmse']:.4f}")
            print(f"WVHT_1h RMSE: {metrics['by_target']['WVHT_1h']['rmse']:.4f}")
            print(f"WVHT_6h RMSE: {metrics['by_target']['WVHT_6h']['rmse']:.4f}")
            
            # Compare with baseline original LSTM
            print("\n=== Comparison with Original LSTM ===")
            if station_id == '46012':
                original_rmse = 0.6039
                original_1h = 0.3822
                original_6h = 0.5136
            else:  # 46221
                original_rmse = 0.5965
                original_1h = 0.3603
                original_6h = 0.5545
            
            optimized_rmse = metrics['overall']['rmse']
            optimized_1h = metrics['by_target']['WVHT_1h']['rmse']
            optimized_6h = metrics['by_target']['WVHT_6h']['rmse']
            
            improvement = ((original_rmse - optimized_rmse) / original_rmse) * 100
            improvement_1h = ((original_1h - optimized_1h) / original_1h) * 100
            improvement_6h = ((original_6h - optimized_6h) / original_6h) * 100
            
            print(f"Original LSTM RMSE: {original_rmse:.4f}")
            print(f"Optimized LSTM RMSE: {optimized_rmse:.4f}")
            print(f"Overall improvement: {improvement:+.1f}%")
            print(f"WVHT_1h improvement: {improvement_1h:+.1f}%")
            print(f"WVHT_6h improvement: {improvement_6h:+.1f}%")
    
    # Save results
    search.save_results()
    
    return search


if __name__ == "__main__":
    # Run hyperparameter search for both stations
    stations = ['46012', '46221']
    
    for station in stations:
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER SEARCH FOR STATION {station}")
        print(f"{'='*60}")
        
        try:
            search = run_hyperparameter_search(station, max_trials=15, verbose=1)
            print(f"✅ Hyperparameter search completed for station {station}")
            
        except Exception as e:
            print(f"❌ Error in hyperparameter search for station {station}: {e}")
            continue