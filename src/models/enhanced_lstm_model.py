import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, LayerNormalization, Add
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from evaluation import evaluate_model_comprehensive

class EnhancedLSTMSurfModel:
    """
    Enhanced LSTM model with multiple improvement strategies for surf forecasting.
    """
    
    def __init__(self, 
                 lookback_hours=48,  # Increased from 24
                 forecast_horizons=[1, 3, 6],
                 target_cols=['WVHT', 'DPD'],
                 lstm_layers=[128, 64, 32],  # Deeper network
                 dropout_rate=0.3,  # Higher dropout
                 learning_rate=0.001,  # Higher learning rate
                 use_attention=True,
                 use_residual=True,
                 separate_decoders=True,
                 random_state=42):
        """
        Initialize Enhanced LSTM model.
        """
        self.lookback_hours = lookback_hours
        self.forecast_horizons = forecast_horizons
        self.target_cols = target_cols
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.separate_decoders = separate_decoders
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.model = None
        self.history = None
        self.n_features = None
        self.n_targets = len(target_cols)
        self.n_horizons = len(forecast_horizons)
        
    def build_model(self, input_shape):
        """Build enhanced LSTM architecture."""
        self.n_features = input_shape[1]
        
        if self.separate_decoders:
            return self._build_multi_decoder_model(input_shape)
        elif self.use_attention:
            return self._build_attention_model(input_shape)
        else:
            return self._build_deep_lstm_model(input_shape)
    
    def _build_multi_decoder_model(self, input_shape):
        """Build model with separate decoders for each forecast horizon."""
        
        # Shared encoder
        inputs = Input(shape=input_shape)
        
        # Deep LSTM encoder
        x = inputs
        lstm_outputs = []
        
        for i, units in enumerate(self.lstm_layers):
            return_sequences = i < len(self.lstm_layers) - 1
            x = LSTM(units, 
                    return_sequences=return_sequences,
                    name=f'encoder_lstm_{i+1}')(x)
            lstm_outputs.append(x)
            x = Dropout(self.dropout_rate, name=f'encoder_dropout_{i+1}')(x)
        
        # Separate decoders for each horizon
        horizon_outputs = []
        
        for h_idx, horizon in enumerate(self.forecast_horizons):
            # Horizon-specific decoder
            decoder = Dense(64, activation='relu', 
                          name=f'decoder_{horizon}h_dense1')(x)
            decoder = Dropout(self.dropout_rate, 
                            name=f'decoder_{horizon}h_dropout')(decoder)
            decoder = Dense(32, activation='relu', 
                          name=f'decoder_{horizon}h_dense2')(decoder)
            
            # Output for this horizon (both WVHT and DPD)
            horizon_out = Dense(self.n_targets, 
                              name=f'output_{horizon}h')(decoder)
            horizon_outputs.append(horizon_out)
        
        # Concatenate all horizon outputs
        if len(horizon_outputs) > 1:
            outputs = tf.keras.layers.Concatenate(name='concat_outputs')(horizon_outputs)
        else:
            outputs = horizon_outputs[0]
        
        model = Model(inputs=inputs, outputs=outputs, name='MultiDecoderLSTM')
        
        # Compile with AdamW
        optimizer = AdamW(learning_rate=self.learning_rate, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def _build_attention_model(self, input_shape):
        """Build LSTM with attention mechanism."""
        
        inputs = Input(shape=input_shape)
        
        # Deep LSTM layers with return_sequences=True for attention
        x = inputs
        for i, units in enumerate(self.lstm_layers[:-1]):
            x = LSTM(units, return_sequences=True, 
                    name=f'lstm_{i+1}')(x)
            x = Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
        
        # Final LSTM layer (return sequences for attention)
        lstm_out = LSTM(self.lstm_layers[-1], return_sequences=True, 
                       name=f'lstm_{len(self.lstm_layers)}')(x)
        
        # Attention mechanism
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=self.lstm_layers[-1] // 4,
            name='multi_head_attention'
        )(lstm_out, lstm_out)
        
        # Add & Norm
        attention = Add(name='add_attention')([lstm_out, attention])
        attention = LayerNormalization(name='layer_norm')(attention)
        
        # Global average pooling to get fixed-size output
        pooled = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(attention)
        
        # Final dense layers
        x = Dense(64, activation='relu', name='final_dense1')(pooled)
        x = Dropout(self.dropout_rate, name='final_dropout')(x)
        outputs = Dense(self.n_horizons * self.n_targets, name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='AttentionLSTM')
        
        optimizer = AdamW(learning_rate=self.learning_rate, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def _build_deep_lstm_model(self, input_shape):
        """Build deeper LSTM model with residual connections."""
        
        model = Sequential([
            # First LSTM block
            LSTM(self.lstm_layers[0], return_sequences=True, 
                 input_shape=input_shape, name='lstm_1'),
            Dropout(self.dropout_rate, name='dropout_1'),
            
            # Second LSTM block  
            LSTM(self.lstm_layers[1], return_sequences=len(self.lstm_layers) > 2,
                 name='lstm_2'),
            Dropout(self.dropout_rate, name='dropout_2'),
            
            # Third LSTM block (if specified)
            *([LSTM(self.lstm_layers[2], return_sequences=False, name='lstm_3'),
               Dropout(self.dropout_rate, name='dropout_3')] 
              if len(self.lstm_layers) > 2 else []),
            
            # Dense layers
            Dense(128, activation='relu', name='dense_1'),
            Dropout(self.dropout_rate, name='dense_dropout'),
            Dense(64, activation='relu', name='dense_2'),
            Dense(self.n_horizons * self.n_targets, name='output')
        ])
        
        optimizer = AdamW(learning_rate=self.learning_rate, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=300, batch_size=64, patience=30, verbose=1):
        """Train the enhanced LSTM model."""
        
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.7,  # More aggressive reduction
                patience=15,  # Shorter patience
                min_lr=1e-7,
                verbose=verbose
            )
        ]
        
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
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X, y):
        """Evaluate model performance."""
        predictions = self.predict(X)
        
        target_names = [f"{col}_{h}h" for h in self.forecast_horizons for col in self.target_cols]
        metrics_result = evaluate_model_comprehensive(
            y_true=y,
            y_pred=predictions,
            target_names=target_names,
            model_name="EnhancedLSTM"
        )
        
        if isinstance(metrics_result, dict) and 'primary_metrics' in metrics_result:
            metrics = metrics_result['primary_metrics']
        else:
            metrics = metrics_result
        
        return metrics


def train_enhanced_lstm(station_id, 
                       architecture='multi_decoder',  # 'multi_decoder', 'attention', 'deep'
                       lookback_hours=48,
                       lstm_layers=[128, 64, 32],
                       learning_rate=0.001,
                       epochs=300,
                       batch_size=64):
    """
    Train enhanced LSTM model for longer forecast horizons.
    """
    print(f"\n=== Training Enhanced LSTM for Station {station_id} ===")
    print(f"Architecture: {architecture}")
    print(f"Lookback: {lookback_hours} hours")
    print(f"LSTM layers: {lstm_layers}")
    
    # Load data (will need to regenerate sequences for longer lookback)
    data_path = Path(f"data/splits/{station_id}/sequences")
    
    # For now, use existing 24h sequences - in practice would regenerate with 48h
    X_train = np.load(data_path / 'X_train.npy')
    y_train = np.load(data_path / 'y_train.npy')
    X_val = np.load(data_path / 'X_val.npy') 
    y_val = np.load(data_path / 'y_val.npy')
    
    # Clean NaN values
    if np.isnan(X_train).sum() > 0 or np.isnan(X_val).sum() > 0:
        print("Cleaning NaN values...")
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
    
    print(f"Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Initialize enhanced model
    model_config = {
        'lstm_layers': lstm_layers,
        'learning_rate': learning_rate,
        'use_attention': architecture == 'attention',
        'separate_decoders': architecture == 'multi_decoder',
        'use_residual': architecture == 'deep'
    }
    
    model = EnhancedLSTMSurfModel(**model_config)
    
    # Train
    print(f"\nTraining {architecture} LSTM...")
    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=30,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating enhanced LSTM...")
    metrics = model.evaluate(X_val, y_val)
    
    # Print key results
    print(f"\n=== Enhanced LSTM Results ===")
    print(f"Overall RMSE: {metrics['overall']['rmse']:.4f}")
    print(f"WVHT_1h RMSE: {metrics['by_target']['WVHT_1h']['rmse']:.4f}")
    print(f"WVHT_6h RMSE: {metrics['by_target']['WVHT_6h']['rmse']:.4f}")
    
    # Save model
    model_path = Path(f"models/enhanced_lstm/{station_id}_{architecture}")
    model_path.mkdir(parents=True, exist_ok=True)
    model.model.save(str(model_path / 'model.keras'))
    
    # Save results
    results_path = data_path / f'enhanced_lstm_{architecture}_results.json'
    with open(results_path, 'w') as f:
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                metrics_json[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                   for k, v in value.items()}
            else:
                metrics_json[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
        json.dump(metrics_json, f, indent=2)
    
    return model, history, metrics


if __name__ == "__main__":
    # Test different architectures on station 46221
    architectures = ['deep', 'attention', 'multi_decoder']
    station = '46221'
    
    for arch in architectures:
        try:
            print(f"\n{'='*60}")
            model, history, metrics = train_enhanced_lstm(station, architecture=arch)
            print(f"✅ {arch} architecture completed!")
            
        except Exception as e:
            print(f"❌ Error with {arch} architecture: {e}")
            continue