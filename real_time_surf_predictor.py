"""
Real-Time Surf Prediction Script

Fetches live NOAA buoy data, applies feature engineering, and makes predictions
using our trained Ridge/Lasso models for current surf conditions.
"""

import numpy as np
import pandas as pd
import requests
import joblib
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RealTimeSurfPredictor:
    """
    Real-time surf prediction using trained models and live NOAA data.
    """
    
    def __init__(self):
        self.stations = {
            '46012': {'name': 'Half Moon Bay', 'lat': 37.361, 'lon': -122.881},
            '46221': {'name': 'Santa Barbara', 'lat': 34.274, 'lon': -119.878}
        }
        self.models = {}
        self.scalers = {}
        self.feature_cols = []
        self.target_names = ['WVHT_1h', 'DPD_1h', 'WVHT_3h', 'DPD_3h', 'WVHT_6h', 'DPD_6h']
        self.METERS_TO_FEET = 3.28084  # Conversion factor
    
    def meters_to_feet(self, meters):
        """Convert meters to feet."""
        if meters is None or (isinstance(meters, float) and np.isnan(meters)):
            return None
        return meters * self.METERS_TO_FEET
        
    def load_trained_models(self):
        """Load trained Ridge/Lasso models and scalers."""
        print("Loading trained models...")
        
        for station_id in self.stations.keys():
            try:
                sequences_path = Path(f"data/splits/{station_id}/sequences")
                
                # Load metadata for feature columns
                with open(sequences_path / 'metadata.json', 'r') as f:
                    metadata = json.load(f)
                self.feature_cols = metadata['feature_columns']
                
                # Load trained models
                ridge_model = joblib.load(sequences_path / 'models' / 'ridge_best.pkl')
                lasso_model = joblib.load(sequences_path / 'models' / 'enhanced_best.pkl')
                
                # Load scalers
                feature_scaler = joblib.load(sequences_path / 'feature_scaler.pkl')
                target_scaler = joblib.load(sequences_path / 'target_scaler.pkl')
                
                self.models[station_id] = {
                    'ridge': ridge_model,
                    'lasso': lasso_model
                }
                
                self.scalers[station_id] = {
                    'feature': feature_scaler,
                    'target': target_scaler
                }
                
                print(f"✅ Loaded models for station {station_id} ({self.stations[station_id]['name']})")
                
            except Exception as e:
                print(f"❌ Error loading models for station {station_id}: {e}")
        
        return len(self.models) > 0
    
    def fetch_live_buoy_data(self, station_id, hours_back=48):
        """
        Fetch recent buoy data from NOAA API.
        
        Args:
            station_id: NOAA station ID
            hours_back: Hours of historical data to fetch
            
        Returns:
            DataFrame with recent buoy measurements
        """
        print(f"Fetching live data for station {station_id}...")
        
        # NOAA API endpoint for real-time data
        # Using the last 2 days of data to ensure we have enough for feature engineering
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours_back)
        
        # NOAA real-time data URL (45-day rolling window)
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
        
        try:
            # Fetch real-time data
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the fixed-width format
            lines = response.text.strip().split('\n')
            
            # Find header line (look for #YY or YY pattern)
            data_start = 0
            header_line = None
            for i, line in enumerate(lines):
                if line.startswith("#YY") or (i == 0 and "YY" in line):
                    header_line = line.replace("#", "").strip()
                    data_start = i + 2  # Skip header and units line
                    break
            
            if header_line is None and len(lines) >= 2:
                # Fallback: use first line as header
                header_line = lines[0].replace("#", "").strip()
                data_start = 2
            
            if header_line is None or len(lines) < data_start + 1:
                raise ValueError("Insufficient data from NOAA API")
            
            # Column names (varies by station)
            columns = header_line.split()
            
            # Parse data rows
            data_rows = []
            for line in lines[data_start:]:
                if line.strip() and not line.strip().startswith('#'):
                    row = line.split()
                    if len(row) >= len(columns):
                        data_rows.append(row[:len(columns)])
                    elif len(row) >= 5:  # Minimum required columns
                        # Pad with None if needed
                        padded_row = row + [None] * (len(columns) - len(row))
                        data_rows.append(padded_row[:len(columns)])
            
            if not data_rows:
                raise ValueError("No valid data rows found")
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=columns)
            
            # Convert time columns to numeric first
            time_cols = ['YY', 'MM', 'DD', 'hh', 'mm']
            for col in time_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert other columns to numeric (handle missing values)
            numeric_columns = ['WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 
                             'DEWP', 'VIS', 'PTDY', 'TIDE', 'WSPD', 'GST', 'WDIR']
            
            for col in numeric_columns:
                if col in df.columns:
                    # Replace missing value codes
                    df[col] = df[col].replace(['99.00', '999.0', '99.0', '999', '99', 'MM', '9999'], pd.NA)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create datetime index - handle both YY (2-digit) and full year formats
            if 'YY' in df.columns and 'MM' in df.columns and 'DD' in df.columns:
                year_col = df['YY'].copy()
                # Handle 2-digit years
                if year_col.max() < 100:
                    year_col = year_col + 2000
                
                # Build datetime
                datetime_dict = {
                    'year': year_col,
                    'month': df['MM'],
                    'day': df['DD'],
                    'hour': df['hh'] if 'hh' in df.columns else 0,
                    'minute': df['mm'] if 'mm' in df.columns else 0
                }
                
                df['datetime'] = pd.to_datetime(datetime_dict, errors='coerce')
            else:
                raise ValueError(f"Missing required time columns. Found: {df.columns.tolist()}")
            
            # Remove rows with invalid datetime
            df = df.dropna(subset=['datetime'])
            
            if len(df) == 0:
                raise ValueError("No valid datetime rows after parsing")
            
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # Drop time columns (no longer needed)
            time_cols_to_drop = [col for col in time_cols if col in df.columns]
            df = df.drop(columns=time_cols_to_drop, errors='ignore')
            
            # Keep only last N hours
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            df = df[df.index >= cutoff_time]
            
            print(f"✅ Fetched {len(df)} hours of data for {self.stations[station_id]['name']}")
            print(f"   Latest data: {df.index[-1] if len(df) > 0 else 'No data'}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for station {station_id}: {e}")
            print("   Falling back to synthetic data for demonstration...")
            
            # Create synthetic data for demonstration
            return self._generate_synthetic_data(station_id, hours_back)
    
    def _generate_synthetic_data(self, station_id, hours_back=48):
        """Generate realistic synthetic data for demonstration."""
        
        # Create timestamp range
        end_time = datetime.utcnow()
        timestamps = pd.date_range(
            start=end_time - timedelta(hours=hours_back),
            end=end_time,
            freq='1H'
        )
        
        # Base patterns for each station
        if station_id == '46012':  # Half Moon Bay - more variable
            base_wvht = 1.5 + 0.5 * np.sin(np.linspace(0, 4*np.pi, len(timestamps)))
            base_dpd = 10 + 3 * np.sin(np.linspace(0, 2*np.pi, len(timestamps)))
            base_wspd = 8 + 4 * np.random.random(len(timestamps))
            base_pres = 1013 + 10 * np.sin(np.linspace(0, np.pi, len(timestamps)))
            base_wtmp = 15 + 2 * np.sin(np.linspace(0, 2*np.pi, len(timestamps)))
        else:  # Santa Barbara - more stable
            base_wvht = 1.8 + 0.3 * np.sin(np.linspace(0, 3*np.pi, len(timestamps)))
            base_dpd = 12 + 2 * np.sin(np.linspace(0, 2*np.pi, len(timestamps)))
            base_wspd = 6 + 3 * np.random.random(len(timestamps))
            base_pres = 1015 + 8 * np.sin(np.linspace(0, np.pi, len(timestamps)))
            base_wtmp = 18 + 1 * np.sin(np.linspace(0, 2*np.pi, len(timestamps)))
        
        # Add some realistic noise
        noise_factor = 0.1
        wvht = np.maximum(0.1, base_wvht + noise_factor * np.random.normal(0, 1, len(timestamps)))
        dpd = np.maximum(3, base_dpd + noise_factor * np.random.normal(0, 2, len(timestamps)))
        wspd = np.maximum(0, base_wspd + noise_factor * np.random.normal(0, 1, len(timestamps)))
        pres = base_pres + noise_factor * np.random.normal(0, 2, len(timestamps))
        wtmp = base_wtmp + noise_factor * np.random.normal(0, 0.5, len(timestamps))
        
        # Create DataFrame with all required columns
        df = pd.DataFrame({
            'WVHT': wvht,
            'DPD': dpd,
            'WSPD': wspd,
            'PRES': pres,
            'WTMP': wtmp,
            'WDIR': 270 + 30 * np.sin(np.linspace(0, 2*np.pi, len(timestamps)))  # Variable wind direction
        }, index=timestamps)
        
        print(f"✅ Generated synthetic data for demonstration ({len(df)} hours)")
        return df
    
    def engineer_features(self, df):
        """Apply the same feature engineering as training."""
        
        print("Engineering features...")
        
        # Ensure we have required columns
        required_cols = ['WVHT', 'DPD', 'WSPD', 'PRES', 'WTMP']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create feature dataframe
        features_df = df.copy()
        
        # Fill missing values in base columns before feature engineering
        for col in required_cols:
            if col in features_df.columns:
                # Forward fill, then backward fill, then fill with 0
                features_df[col] = features_df[col].ffill().bfill().fillna(0)
        
        # Add temporal features
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df.index.hour / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df.index.hour / 24)
        features_df['day_sin'] = np.sin(2 * np.pi * features_df.index.dayofyear / 365.25)
        features_df['day_cos'] = np.cos(2 * np.pi * features_df.index.dayofyear / 365.25)
        
        # Add wave power (ensure no NaN from multiplication)
        wvht_filled = features_df['WVHT'].fillna(0)
        dpd_filled = features_df['DPD'].fillna(0)
        features_df['wave_power'] = wvht_filled**2 * dpd_filled
        
        # Add pressure gradients (simple first difference)
        pres_filled = features_df['PRES'].ffill().bfill().fillna(1013)
        features_df['pressure_gradient'] = pres_filled.diff().fillna(0)
        features_df['pressure_3h_change'] = pres_filled.diff(3).fillna(0)
        features_df['pressure_6h_change'] = pres_filled.diff(6).fillna(0)
        
        # Add rolling statistics (6h, 12h, 24h windows)
        base_vars = ['WVHT', 'DPD', 'WSPD', 'PRES', 'WTMP']
        windows = [6, 12, 24]
        stats = ['mean', 'std', 'min', 'max']
        
        for var in base_vars:
            if var not in features_df.columns:
                continue
            var_filled = features_df[var].ffill().bfill().fillna(0)
            for window in windows:
                for stat in stats:
                    col_name = f"{var}_{window}h_{stat}"
                    if stat == 'mean':
                        features_df[col_name] = var_filled.rolling(window=window, min_periods=1).mean()
                    elif stat == 'std':
                        features_df[col_name] = var_filled.rolling(window=window, min_periods=1).std().fillna(0)
                    elif stat == 'min':
                        features_df[col_name] = var_filled.rolling(window=window, min_periods=1).min()
                    elif stat == 'max':
                        features_df[col_name] = var_filled.rolling(window=window, min_periods=1).max()
        
        # Add lag features
        lag_vars = ['WVHT', 'DPD']
        lags = [1, 3, 6, 12]
        
        for var in lag_vars:
            if var not in features_df.columns:
                continue
            var_filled = features_df[var].ffill().bfill().fillna(0)
            for lag in lags:
                features_df[f"{var}_lag_{lag}h"] = var_filled.shift(lag)
        
        # Fill NaN values aggressively - multiple passes
        # First pass: forward fill
        features_df = features_df.ffill()
        # Second pass: backward fill
        features_df = features_df.bfill()
        # Third pass: fill remaining with 0
        features_df = features_df.fillna(0)
        
        # Ensure we have all required features - create missing ones with defaults
        for col in self.feature_cols:
            if col not in features_df.columns:
                print(f"Warning: Missing feature {col}, using default value 0")
                features_df[col] = 0
        
        # Select only the features used in training, in the correct order
        features_df = features_df[self.feature_cols]
        
        # Final NaN check and replacement - use numpy to ensure no NaNs
        nan_count = features_df.isnull().sum().sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values after feature engineering, replacing with 0")
            features_df = features_df.fillna(0)
        
        # Convert to numpy and replace any remaining NaNs/Infs
        features_array = features_df.values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        features_df = pd.DataFrame(features_array, index=features_df.index, columns=features_df.columns)
        
        print(f"✅ Engineered {features_df.shape[1]} features from {len(features_df)} time points")
        
        return features_df
    
    def create_prediction_sequences(self, features_df, lookback_hours=24):
        """Create 24-hour sequences for model input."""
        
        if len(features_df) < lookback_hours:
            raise ValueError(f"Insufficient data: need {lookback_hours} hours, got {len(features_df)}")
        
        # Use the most recent 24 hours
        recent_features = features_df.iloc[-lookback_hours:].copy()
        
        # Final NaN safety check - multiple passes
        nan_count = recent_features.isnull().sum().sum()
        if nan_count > 0:
            print(f"Warning: Found {nan_count} NaN values in sequence data, replacing with 0")
            recent_features = recent_features.ffill().bfill().fillna(0)
        
        # Create sequence (flatten for linear models)
        sequence = recent_features.values.flatten()
        
        # Ensure no NaN or infinite values using numpy
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Final validation - check for any remaining issues
        if np.any(np.isnan(sequence)) or np.any(np.isinf(sequence)):
            print(f"Warning: Found NaN/Inf in sequence after cleaning, replacing with 0")
            sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape for model input
        sequence_2d = sequence.reshape(1, -1)
        
        print(f"✅ Created sequence shape: {sequence_2d.shape}")
        print(f"   Sequence stats: min={sequence.min():.3f}, max={sequence.max():.3f}, mean={sequence.mean():.3f}")
        print(f"   NaN count: {np.isnan(sequence).sum()}, Inf count: {np.isinf(sequence).sum()}")
        
        return sequence_2d  # Shape: (1, 24*n_features)
    
    def make_predictions(self, station_id, X_sequence):
        """Make predictions using trained models."""
        
        print(f"Making predictions for {self.stations[station_id]['name']}...")
        
        if station_id not in self.models:
            raise ValueError(f"No trained models found for station {station_id}")
        
        # Final NaN check before scaling
        if np.any(np.isnan(X_sequence)) or np.any(np.isinf(X_sequence)):
            print(f"Warning: Found NaN/Inf in X_sequence before scaling, replacing with 0")
            X_sequence = np.nan_to_num(X_sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        feature_scaler = self.scalers[station_id]['feature']
        target_scaler = self.scalers[station_id]['target']
        
        try:
            X_scaled = feature_scaler.transform(X_sequence)
            
            # Check for NaN after scaling
            if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
                print(f"Warning: Found NaN/Inf after scaling, replacing with 0")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Make predictions with both models
            ridge_pred_scaled = self.models[station_id]['ridge'].predict(X_scaled)
            lasso_pred_scaled = self.models[station_id]['lasso'].predict(X_scaled)
            
            # Inverse transform predictions
            ridge_pred = target_scaler.inverse_transform(ridge_pred_scaled.reshape(1, -1))
            lasso_pred = target_scaler.inverse_transform(lasso_pred_scaled.reshape(1, -1))
            
            # Format predictions
            predictions = {
                'ridge': {
                    'WVHT_1h': ridge_pred[0][0],
                    'DPD_1h': ridge_pred[0][1],
                    'WVHT_3h': ridge_pred[0][2],
                    'DPD_3h': ridge_pred[0][3],
                    'WVHT_6h': ridge_pred[0][4],
                    'DPD_6h': ridge_pred[0][5]
                },
                'lasso': {
                    'WVHT_1h': lasso_pred[0][0],
                    'DPD_1h': lasso_pred[0][1],
                    'WVHT_3h': lasso_pred[0][2],
                    'DPD_3h': lasso_pred[0][3],
                    'WVHT_6h': lasso_pred[0][4],
                    'DPD_6h': lasso_pred[0][5]
                }
            }
            
            return predictions
            
        except ValueError as e:
            if "NaN" in str(e) or "missing values" in str(e).lower():
                print(f"Error: NaN values detected in input. Attempting to fix...")
                # One more aggressive cleanup
                X_sequence = np.nan_to_num(X_sequence, nan=0.0, posinf=0.0, neginf=0.0)
                X_scaled = feature_scaler.transform(X_sequence)
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                
                ridge_pred_scaled = self.models[station_id]['ridge'].predict(X_scaled)
                lasso_pred_scaled = self.models[station_id]['lasso'].predict(X_scaled)
                
                ridge_pred = target_scaler.inverse_transform(ridge_pred_scaled.reshape(1, -1))
                lasso_pred = target_scaler.inverse_transform(lasso_pred_scaled.reshape(1, -1))
                
                predictions = {
                    'ridge': {
                        'WVHT_1h': ridge_pred[0][0],
                        'DPD_1h': ridge_pred[0][1],
                        'WVHT_3h': ridge_pred[0][2],
                        'DPD_3h': ridge_pred[0][3],
                        'WVHT_6h': ridge_pred[0][4],
                        'DPD_6h': ridge_pred[0][5]
                    },
                    'lasso': {
                        'WVHT_1h': lasso_pred[0][0],
                        'DPD_1h': lasso_pred[0][1],
                        'WVHT_3h': lasso_pred[0][2],
                        'DPD_3h': lasso_pred[0][3],
                        'WVHT_6h': lasso_pred[0][4],
                        'DPD_6h': lasso_pred[0][5]
                    }
                }
                return predictions
            else:
                raise
    
    def classify_surf_conditions(self, predictions, current_data=None):
        """Enhanced surf conditions classification."""
        
        # Import enhanced classifier
        from enhanced_surf_classifier import EnhancedSurfClassifier
        classifier = EnhancedSurfClassifier()
        
        surf_conditions = {}
        
        # Get current wind data if available
        wind_data = None
        if current_data is not None and not current_data.empty:
            latest = current_data.iloc[-1]
            wind_data = {
                'WSPD': latest.get('WSPD'),
                'WDIR': latest.get('WDIR', 270)  # Default west wind if no direction data
            }
        
        for model_name, preds in predictions.items():
            conditions = {}
            
            for horizon in ['1h', '3h', '6h']:
                wvht = preds[f'WVHT_{horizon}']
                dpd = preds[f'DPD_{horizon}']
                
                # Enhanced classification
                enhanced_analysis = classifier.classify_surf_conditions(
                    wave_height=wvht,
                    wave_period=dpd,
                    wind_speed=wind_data['WSPD'] if wind_data else None,
                    wind_direction=wind_data['WDIR'] if wind_data else None
                )
                
                # Legacy format for compatibility + enhanced info
                conditions[horizon] = {
                    'wave_height_m': wvht,
                    'wave_period_s': dpd,
                    'good_surf': enhanced_analysis['surfable'],
                    'size_category': enhanced_analysis['size_category'],
                    'quality': enhanced_analysis['quality_category'],
                    'enhanced': enhanced_analysis  # Full enhanced analysis
                }
            
            surf_conditions[model_name] = conditions
        
        return surf_conditions
    
    def predict_current_conditions(self):
        """Main function to predict current surf conditions."""
        
        print("=" * 60)
        print("REAL-TIME SURF FORECAST")
        print(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print("=" * 60)
        
        # Load models
        if not self.load_trained_models():
            print("❌ Failed to load trained models")
            return None
        
        results = {}
        
        # Process each station
        for station_id, station_info in self.stations.items():
            if station_id not in self.models:
                continue
                
            try:
                print(f"\n📍 {station_info['name']} (Station {station_id})")
                print("-" * 40)
                
                # Fetch live data
                buoy_data = self.fetch_live_buoy_data(station_id)
                
                if buoy_data.empty:
                    print("❌ No data available")
                    continue
                
                # Engineer features
                features = self.engineer_features(buoy_data)
                
                # Create prediction sequence
                X_sequence = self.create_prediction_sequences(features)
                
                # Make predictions
                predictions = self.make_predictions(station_id, X_sequence)
                
                # Classify conditions
                surf_conditions = self.classify_surf_conditions(predictions, buoy_data)
                
                # Display results
                self.display_predictions(station_info['name'], surf_conditions, buoy_data)
                
                results[station_id] = {
                    'station_name': station_info['name'],
                    'predictions': predictions,
                    'surf_conditions': surf_conditions,
                    'current_conditions': {
                        'WVHT': buoy_data['WVHT'].iloc[-1] if 'WVHT' in buoy_data.columns else None,
                        'DPD': buoy_data['DPD'].iloc[-1] if 'DPD' in buoy_data.columns else None,
                        'WSPD': buoy_data['WSPD'].iloc[-1] if 'WSPD' in buoy_data.columns else None
                    }
                }
                
            except Exception as e:
                print(f"❌ Error processing {station_info['name']}: {e}")
                continue
        
        return results
    
    def display_predictions(self, station_name, surf_conditions, current_data):
        """Display formatted predictions."""
        
        print(f"\n🌊 CURRENT CONDITIONS:")
        if not current_data.empty:
            latest = current_data.iloc[-1]
            wvht_m = latest.get('WVHT', None)
            if wvht_m is not None and not (isinstance(wvht_m, float) and np.isnan(wvht_m)):
                wvht_ft = self.meters_to_feet(wvht_m)
                print(f"   Wave Height: {wvht_ft:.1f}ft ({wvht_m:.1f}m)")
            else:
                print(f"   Wave Height: N/A")
            print(f"   Wave Period: {latest.get('DPD', 'N/A'):.1f}s") 
            print(f"   Wind Speed: {latest.get('WSPD', 'N/A'):.1f} m/s")
            print(f"   Time: {current_data.index[-1].strftime('%Y-%m-%d %H:%M UTC')}")
        
        # Display predictions for both models
        for model_name in ['ridge', 'lasso']:
            model_display = "Ridge (Linear)" if model_name == 'ridge' else "Lasso (Feature Selection)"
            print(f"\n🔮 {model_display.upper()} PREDICTIONS:")
            
            conditions = surf_conditions[model_name]
            
            for horizon in ['1h', '3h', '6h']:
                cond = conditions[horizon]
                enhanced = cond.get('enhanced', {})
                
                # Get enhanced classification
                condition_level = enhanced.get('condition_level', 'Unknown')
                overall_score = enhanced.get('overall_score', 0)
                wind_category = enhanced.get('wind_category', 'Unknown')
                
                # Choose emoji based on condition level
                if condition_level == 'Excellent':
                    surf_emoji = "🤩"
                elif condition_level == 'Very Good':
                    surf_emoji = "🏄‍♂️"
                elif condition_level == 'Good':
                    surf_emoji = "😊"
                elif condition_level == 'Fair':
                    surf_emoji = "🤔"
                else:
                    surf_emoji = "😔"
                
                wave_height_m = cond['wave_height_m']
                wave_height_ft = self.meters_to_feet(wave_height_m)
                print(f"   {horizon:3} | {wave_height_ft:4.1f}ft ({wave_height_m:.1f}m) {cond['wave_period_s']:4.1f}s | "
                      f"{cond['size_category']:12} | Score: {overall_score:4.1f}/10 | {surf_emoji} {condition_level}")
                
                # Show wind effects if available
                if wind_category != 'Unknown':
                    print(f"       Wind: {wind_category}")
                
                # Show skill level recommendations
                surfability = enhanced.get('surfability', {})
                if surfability.get('beginner_friendly'):
                    print(f"       👶 Beginner friendly")
                elif surfability.get('expert_only'):
                    print(f"       ⚡ Advanced/Expert only")
                elif surfability.get('too_dangerous'):
                    print(f"       ⚠️  Dangerous conditions")


def main():
    """Run real-time surf prediction."""
    predictor = RealTimeSurfPredictor()
    results = predictor.predict_current_conditions()
    
    if results:
        print("\n" + "=" * 60)
        print("PREDICTION SUMMARY")
        print("=" * 60)
        
        for station_id, result in results.items():
            print(f"\n📍 {result['station_name']}:")
            
            # Compare model predictions
            ridge_1h = result['predictions']['ridge']['WVHT_1h']
            lasso_1h = result['predictions']['lasso']['WVHT_1h']
            
            ridge_1h_ft = predictor.meters_to_feet(ridge_1h)
            lasso_1h_ft = predictor.meters_to_feet(lasso_1h)
            print(f"   1h Wave Height: Ridge={ridge_1h_ft:.1f}ft ({ridge_1h:.1f}m), Lasso={lasso_1h_ft:.1f}ft ({lasso_1h:.1f}m)")
            
            # Good surf consensus
            ridge_good = result['surf_conditions']['ridge']['1h']['good_surf']
            lasso_good = result['surf_conditions']['lasso']['1h']['good_surf']
            
            if ridge_good and lasso_good:
                print(f"   🏄‍♂️ CONSENSUS: GOOD SURF CONDITIONS")
            elif ridge_good or lasso_good:
                print(f"   🤔 MIXED: One model predicts good surf")
            else:
                print(f"   😔 CONSENSUS: Poor surf conditions")
    
    else:
        print("❌ No predictions generated")


if __name__ == "__main__":
    main()