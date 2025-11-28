import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from evaluation import evaluate_model_comprehensive

class LSTMSurfModel:
    """
    LSTM model for multi-horizon surf forecasting.
    Predicts wave height (WVHT) and period (DPD) for 1h, 3h, 6h horizons.
    """
    
    def __init__(self, 
                 lookback_hours=24,
                 forecast_horizons=[1, 3, 6],
                 target_cols=['WVHT', 'DPD'],
                 lstm_layers=[64, 32],
                 dropout_rate=0.2,
                 learning_rate=0.001,
                 random_state=42):
        """
        Initialize LSTM model.
        
        Args:
            lookback_hours: Hours of history to use as input
            forecast_horizons: List of forecast horizons (hours)
            target_cols: Target variable names
            lstm_layers: List of LSTM layer sizes
            dropout_rate: Dropout probability
            learning_rate: Adam learning rate
            random_state: Random seed for reproducibility
        """
        self.lookback_hours = lookback_hours
        self.forecast_horizons = forecast_horizons
        self.target_cols = target_cols
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.model = None
        self.history = None
        self.n_features = None
        self.n_targets = len(target_cols)
        self.n_horizons = len(forecast_horizons)
        
    def build_model(self, input_shape):
        """
        Build LSTM architecture according to PRD specifications.
        
        Args:
            input_shape: (timesteps, features) - batch size excluded
        """
        self.n_features = input_shape[1]
        
        model = Sequential([
            # First LSTM layer with return_sequences=True
            LSTM(self.lstm_layers[0], 
                 return_sequences=len(self.lstm_layers) > 1,
                 input_shape=input_shape,
                 name='lstm_1'),
            Dropout(self.dropout_rate, name='dropout_1'),
            
            # Second LSTM layer (if specified)
            *([LSTM(self.lstm_layers[1], 
                   return_sequences=len(self.lstm_layers) > 2,
                   name='lstm_2'),
               Dropout(self.dropout_rate, name='dropout_2')] 
              if len(self.lstm_layers) > 1 else []),
            
            # Third LSTM layer (if specified)
            *([LSTM(self.lstm_layers[2], 
                   return_sequences=False,
                   name='lstm_3'),
               Dropout(self.dropout_rate, name='dropout_3')] 
              if len(self.lstm_layers) > 2 else []),
            
            # Dense layers for final prediction
            Dense(32, activation='relu', name='dense_1'),
            Dense(self.n_horizons * self.n_targets, name='output')
        ])
        
        # Add gradient clipping to prevent exploding gradients
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=200, batch_size=32, patience=20, verbose=1):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences (n_samples, timesteps, features)
            y_train: Training targets (n_samples, n_horizons * n_targets)
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            verbose: Keras verbosity level
            
        Returns:
            Training history
        """
        # Build model if not already built
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=verbose
            )
        ]
        
        # Add validation callbacks if validation data provided
        if X_val is not None and y_val is not None:
            callbacks.append(
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=verbose
                )
            )
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input sequences (n_samples, timesteps, features)
            
        Returns:
            Predictions (n_samples, n_horizons * n_targets)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance using comprehensive metrics.
        
        Args:
            X: Input sequences
            y: True targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        
        # Use comprehensive evaluation from evaluation module
        target_names = [f"{col}_{h}h" for h in self.forecast_horizons for col in self.target_cols]
        metrics = evaluate_model_comprehensive(
            y_true=y,
            y_pred=predictions,
            target_names=target_names,
            model_name="LSTM"
        )
        
        return metrics
    
    def save_model(self, filepath):
        """Save model and configuration."""
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(str(filepath / 'lstm_model.keras'))
        
        # Save configuration
        config = {
            'lookback_hours': self.lookback_hours,
            'forecast_horizons': self.forecast_horizons,
            'target_cols': self.target_cols,
            'lstm_layers': self.lstm_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'n_features': self.n_features
        }
        
        with open(filepath / 'lstm_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load saved model and configuration."""
        filepath = Path(filepath)
        
        # Load configuration
        with open(filepath / 'lstm_config.json', 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(**{k: v for k, v in config.items() 
                         if k != 'n_features'})
        instance.n_features = config['n_features']
        
        # Load Keras model
        instance.model = tf.keras.models.load_model(
            str(filepath / 'lstm_model.keras')
        )
        
        return instance


def train_lstm_model(station_id, 
                     lstm_layers=[64, 32],
                     dropout_rate=0.2,
                     learning_rate=0.0001,
                     batch_size=32,
                     epochs=200,
                     patience=20,
                     verbose=1):
    """
    Train LSTM model for a specific station.
    
    Args:
        station_id: Station identifier (e.g., '46012')
        lstm_layers: List of LSTM layer sizes
        dropout_rate: Dropout probability
        learning_rate: Adam learning rate
        batch_size: Training batch size
        epochs: Maximum epochs
        patience: Early stopping patience
        verbose: Training verbosity
        
    Returns:
        Trained LSTM model, training history, validation metrics
    """
    print(f"\n=== Training LSTM Model for Station {station_id} ===")
    
    # Load sequence data
    data_path = Path(f"data/splits/{station_id}/sequences")
    
    print("Loading sequence data...")
    X_train = np.load(data_path / 'X_train.npy')
    y_train = np.load(data_path / 'y_train.npy') 
    X_val = np.load(data_path / 'X_val.npy')
    y_val = np.load(data_path / 'y_val.npy')
    
    print(f"Data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")
    
    # Check for NaN values
    print(f"\nData quality checks:")
    print(f"  X_train NaN count: {np.isnan(X_train).sum()}")
    print(f"  y_train NaN count: {np.isnan(y_train).sum()}")
    print(f"  X_val NaN count: {np.isnan(X_val).sum()}")
    print(f"  y_val NaN count: {np.isnan(y_val).sum()}")
    
    # Clean NaN values
    if np.isnan(X_train).sum() > 0 or np.isnan(X_val).sum() > 0:
        print("Found NaN values in input data. Cleaning...")
        
        # Replace NaN with 0 (since data should be standardized)
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        
        print("NaN values cleaned.")
    
    # Check data ranges after cleaning
    print(f"\nData ranges (after cleaning):")
    print(f"  X_train: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"  y_train: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"  X_val: [{X_val.min():.3f}, {X_val.max():.3f}]")
    print(f"  y_val: [{y_val.min():.3f}, {y_val.max():.3f}]")
    
    # Initialize model
    model = LSTMSurfModel(
        lstm_layers=lstm_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # Train model
    print(f"\nTraining LSTM with architecture: {lstm_layers}")
    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        verbose=verbose
    )
    
    # Evaluate on validation set
    print("\nEvaluating model performance...")
    val_metrics = model.evaluate(X_val, y_val)
    
    # Save model
    model_path = Path(f"models/lstm/{station_id}")
    model_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    model.save_model(model_path)
    
    return model, history, val_metrics


if __name__ == "__main__":
    # Train LSTM for both stations
    stations = ['46012', '46221']
    print("Training LSTM models for stations: ", stations)
    for station in stations:
        try:
            model, history, metrics = train_lstm_model(station)
            print(f"\nStation {station} LSTM training completed successfully!")
            
        except Exception as e:
            print(f"Error training LSTM for station {station}: {e}")
            continue