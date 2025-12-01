"""
Historical Surf Prediction Script for Station 46026

Makes predictions for station 46026 (San Francisco) for 1h, 3h, and 6h horizons
starting from a specific historical timestamp (2025-11-30 20:00 UTC).
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

class Historical46026Predictor:
    """
    Historical surf prediction for station 46026 using trained models.
    """
    
    def __init__(self):
        self.station_id = '46026'
        self.station_info = {'name': 'San Francisco', 'lat': 37.750, 'lon': -122.838}
        self.models = {}
        self.scalers = {}
        self.feature_cols = []
        self.target_names = ['WVHT_1h', 'DPD_1h', 'WVHT_3h', 'DPD_3h', 'WVHT_6h', 'DPD_6h']
        self.METERS_TO_FEET = 3.28084
    
    def meters_to_feet(self, meters):
        """Convert meters to feet."""
        if meters is None or (isinstance(meters, float) and np.isnan(meters)):
            return None
        return meters * self.METERS_TO_FEET
    
    def load_trained_models(self):
        """Load trained Ridge/Lasso models and scalers for station 46026."""
        print("Loading trained models for station 46026...")
        
        try:
            sequences_path = Path(f"data/splits/{self.station_id}/sequences")
            
            # Load metadata for feature columns
            with open(sequences_path / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            self.feature_cols = metadata['feature_columns']
            
            # Load trained models - try different model names
            model_files = [
                'enhanced_best.pkl',
                'ridge_best.pkl',
                'lasso_best.pkl'
            ]
            
            models_loaded = {}
            for model_file in model_files:
                model_path = sequences_path / 'models' / model_file
                if model_path.exists():
                    model = joblib.load(model_path)
                    model_name = model_file.replace('_best.pkl', '').replace('.pkl', '')
                    models_loaded[model_name] = model
                    print(f"✅ Loaded {model_name} model")
            
            if not models_loaded:
                raise FileNotFoundError("No model files found")
            
            # Load scalers
            feature_scaler = joblib.load(sequences_path / 'feature_scaler.pkl')
            target_scaler = joblib.load(sequences_path / 'target_scaler.pkl')
            
            self.models[self.station_id] = models_loaded
            self.scalers[self.station_id] = {
                'feature': feature_scaler,
                'target': target_scaler
            }
            
            print(f"✅ Loaded models for station {self.station_id} ({self.station_info['name']})")
            print(f"   Available models: {list(models_loaded.keys())}")
            
        except Exception as e:
            print(f"❌ Error loading models for station {self.station_id}: {e}")
            return False
        
        return True
    
    def fetch_historical_data(self, target_datetime, hours_back=48):
        """
        Fetch historical data around the target timestamp.
        
        Args:
            target_datetime: datetime object for the prediction time
            hours_back: Hours of historical data to fetch before target time
        """
        print(f"Fetching historical data for station {self.station_id} around {target_datetime}...")
        
        # Try real-time endpoint first (covers last 45 days)
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{self.station_id}.txt"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the real-time data format
            lines = response.text.strip().split('\n')
            
            # Find header line
            data_start = 0
            header_line = None
            for i, line in enumerate(lines):
                if line.startswith("#YY") or (i == 0 and "YY" in line):
                    header_line = line.replace("#", "").strip()
                    data_start = i + 2  # Skip header and units line
                    break
            
            if header_line is None:
                raise ValueError("Could not find header in data")
            
            # Column names
            columns = header_line.split()
            
            # Parse data rows
            data_rows = []
            for line in lines[data_start:]:
                if line.strip() and not line.startswith('#'):
                    row = line.split()
                    if len(row) >= len(columns):
                        data_rows.append(row[:len(columns)])
            
            if not data_rows:
                raise ValueError("No data rows found")
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=columns)
            
            # Convert time columns
            time_cols = ['YY', 'MM', 'DD', 'hh', 'mm']
            for col in time_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert other columns to numeric
            numeric_columns = ['WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 
                             'DEWP', 'VIS', 'PTDY', 'TIDE', 'WSPD', 'GST', 'WDIR']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].replace(['99.00', '999.0', '99.0', '999', '99', 'MM', '9999'], pd.NA)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create datetime
            year_col = df['YY'].copy()
            if year_col.max() < 100:
                year_col = year_col + 2000
            
            datetime_dict = {
                'year': year_col,
                'month': df['MM'],
                'day': df['DD'],
                'hour': df['hh'] if 'hh' in df.columns else 0,
                'minute': df['mm'] if 'mm' in df.columns else 0
            }
            
            df['datetime'] = pd.to_datetime(datetime_dict, errors='coerce')
            df = df.dropna(subset=['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # Drop time columns
            time_cols_to_drop = [col for col in time_cols if col in df.columns]
            df = df.drop(columns=time_cols_to_drop, errors='ignore')
            
            # Filter for the time window we need
            start_time = target_datetime - timedelta(hours=hours_back)
            end_time = target_datetime
            
            mask = (df.index >= start_time) & (df.index <= end_time)
            df_filtered = df[mask]
            
            print(f"✅ Fetched {len(df_filtered)} hours of data around target time")
            print(f"   Date range: {df_filtered.index.min()} to {df_filtered.index.max()}")
            
            if len(df_filtered) < 24:
                print(f"⚠️ Warning: Only {len(df_filtered)} hours of data available (need 24 for prediction)")
            
            return df_filtered
            
        except Exception as e:
            print(f"❌ Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def engineer_features(self, df):
        """Apply the same feature engineering as training."""
        
        print("Engineering features...")
        
        # Ensure we have required columns
        required_cols = ['WVHT', 'DPD', 'WSPD', 'PRES', 'WTMP']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}, using defaults")
            for col in missing_cols:
                df[col] = 0
        
        # Create feature dataframe
        features_df = df.copy()
        
        # Fill missing values in base columns before feature engineering
        for col in required_cols:
            if col in features_df.columns:
                features_df[col] = features_df[col].ffill().bfill().fillna(0)
        
        # Add temporal features
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df.index.hour / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df.index.hour / 24)
        features_df['day_sin'] = np.sin(2 * np.pi * features_df.index.dayofyear / 365.25)
        features_df['day_cos'] = np.cos(2 * np.pi * features_df.index.dayofyear / 365.25)
        
        # Add wave power
        wvht_filled = features_df['WVHT'].fillna(0)
        dpd_filled = features_df['DPD'].fillna(0)
        features_df['wave_power'] = wvht_filled**2 * dpd_filled
        
        # Add pressure gradients
        pres_filled = features_df['PRES'].ffill().bfill().fillna(1013)
        features_df['pressure_gradient'] = pres_filled.diff().fillna(0)
        features_df['pressure_3h_change'] = pres_filled.diff(3).fillna(0)
        features_df['pressure_6h_change'] = pres_filled.diff(6).fillna(0)
        
        # Add rolling statistics
        base_vars = ['WVHT', 'DPD', 'WSPD', 'PRES', 'WTMP']
        windows = [6, 12, 24]
        
        for var in base_vars:
            if var not in features_df.columns:
                continue
            var_filled = features_df[var].ffill().bfill().fillna(0)
            for window in windows:
                features_df[f'{var}_{window}h_mean'] = var_filled.rolling(window=window, min_periods=1).mean()
                features_df[f'{var}_{window}h_std'] = var_filled.rolling(window=window, min_periods=1).std().fillna(0)
                if var == 'WVHT':
                    features_df[f'{var}_{window}h_min'] = var_filled.rolling(window=window, min_periods=1).min()
                    features_df[f'{var}_{window}h_max'] = var_filled.rolling(window=window, min_periods=1).max()
        
        # Add lag features
        lag_vars = ['WVHT', 'DPD']
        lags = [1, 3, 6, 12]
        
        for var in lag_vars:
            if var not in features_df.columns:
                continue
            var_filled = features_df[var].ffill().bfill().fillna(0)
            for lag in lags:
                features_df[f'{var}_lag_{lag}h'] = var_filled.shift(lag)
        
        # Fill NaN values
        features_df = features_df.ffill().bfill().fillna(0)
        
        # Ensure we have all required features
        for col in self.feature_cols:
            if col not in features_df.columns:
                print(f"Warning: Missing feature {col}, using default value 0")
                features_df[col] = 0
        
        # Select only the features used in training
        features_df = features_df[self.feature_cols]
        
        # Final cleanup
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
        
        # Create sequence (flatten for linear models)
        sequence = recent_features.values.flatten()
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        sequence_2d = sequence.reshape(1, -1)
        
        print(f"✅ Created sequence shape: {sequence_2d.shape}")
        
        return sequence_2d
    
    def make_predictions(self, X_sequence):
        """Make predictions using trained models."""
        
        print(f"Making predictions for {self.station_info['name']}...")
        
        if self.station_id not in self.models:
            raise ValueError(f"No trained models found for station {self.station_id}")
        
        # Scale features
        feature_scaler = self.scalers[self.station_id]['feature']
        target_scaler = self.scalers[self.station_id]['target']
        
        X_scaled = feature_scaler.transform(X_sequence)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        predictions = {}
        
        # Make predictions with available models
        for model_name, model in self.models[self.station_id].items():
            try:
                pred_scaled = model.predict(X_scaled)
                pred = target_scaler.inverse_transform(pred_scaled.reshape(1, -1))
                
                predictions[model_name] = {
                    'WVHT_1h': pred[0][0],
                    'DPD_1h': pred[0][1],
                    'WVHT_3h': pred[0][2],
                    'DPD_3h': pred[0][3],
                    'WVHT_6h': pred[0][4],
                    'DPD_6h': pred[0][5]
                }
                
            except Exception as e:
                print(f"Warning: Error with {model_name} model: {e}")
                continue
        
        return predictions
    
    def classify_surf_conditions(self, wave_height_m, wave_period_s):
        """Simple surf condition classification."""
        
        wave_height_ft = self.meters_to_feet(wave_height_m)
        
        # Size categories
        if wave_height_ft < 2:
            size_category = "Small"
            surfable = False
        elif wave_height_ft < 4:
            size_category = "Small-Medium"
            surfable = True
        elif wave_height_ft < 6:
            size_category = "Medium"
            surfable = True
        elif wave_height_ft < 8:
            size_category = "Medium-Large"
            surfable = True
        elif wave_height_ft < 12:
            size_category = "Large"
            surfable = True
        else:
            size_category = "Very Large"
            surfable = False  # Too dangerous
        
        # Quality based on period
        if wave_period_s < 8:
            quality = "Poor (Wind Chop)"
        elif wave_period_s < 10:
            quality = "Fair"
        elif wave_period_s < 12:
            quality = "Good"
        elif wave_period_s < 14:
            quality = "Very Good"
        else:
            quality = "Excellent (Groundswell)"
        
        return {
            'surfable': surfable,
            'size_category': size_category,
            'quality': quality,
            'wave_height_ft': wave_height_ft,
            'wave_height_m': wave_height_m,
            'wave_period_s': wave_period_s
        }
    
    def fetch_actual_conditions(self, prediction_times):
        """Fetch actual conditions at the predicted times for validation."""
        print("\n🔍 Fetching actual conditions for validation...")
        
        # Get the latest prediction time and add some buffer
        latest_time = max(prediction_times.values())
        end_time = latest_time + timedelta(hours=1)
        
        # Fetch data using the same method as the main prediction
        try:
            df = self.fetch_historical_data(end_time, hours_back=72)  # 3 days of data
            
            if df.empty:
                print("❌ No data available for validation")
                return {}
            
            # Find actual conditions for each prediction time
            actual_conditions = {}
            for horizon, pred_time in prediction_times.items():
                # Find closest data point within 30 minutes
                time_diffs = [(abs((idx - pred_time).total_seconds()), idx) for idx in df.index]
                
                if not time_diffs:
                    actual_conditions[horizon] = None
                    print(f"❌ No data available for {horizon}")
                    continue
                
                # Find minimum time difference
                min_diff_seconds, closest_time = min(time_diffs)
                min_diff_minutes = min_diff_seconds / 60
                
                if min_diff_minutes <= 30:
                    closest_data = df.loc[closest_time]
                    actual_conditions[horizon] = {
                        'time': closest_time,
                        'WVHT': closest_data.get('WVHT'),
                        'DPD': closest_data.get('DPD'),
                        'WSPD': closest_data.get('WSPD'),
                        'time_diff_minutes': min_diff_minutes
                    }
                    print(f"✅ Found actual data for {horizon}: {closest_time} (±{min_diff_minutes:.0f}min)")
                else:
                    actual_conditions[horizon] = None
                    print(f"❌ No actual data found for {horizon} within 30 minutes (closest: ±{min_diff_minutes:.0f}min)")
            
            return actual_conditions
            
        except Exception as e:
            print(f"❌ Error fetching actual conditions: {e}")
            return {}
    
    def calculate_prediction_errors(self, predictions, actual_conditions):
        """Calculate prediction errors and statistics."""
        print("\n📊 PREDICTION VS ACTUAL COMPARISON")
        print("=" * 80)
        
        errors = {}
        
        for model_name, preds in predictions.items():
            model_errors = {}
            
            print(f"\n🔮 {model_name.upper()} MODEL ACCURACY:")
            print("   Time | Predicted     | Actual        | Error (WVHT) | Error (DPD) | Status")
            print("   -----|---------------|---------------|--------------|-------------|--------")
            
            for horizon in ['1h', '3h', '6h']:
                if horizon in actual_conditions and actual_conditions[horizon] is not None:
                    actual = actual_conditions[horizon]
                    
                    pred_wvht = preds[f'WVHT_{horizon}']
                    pred_dpd = preds[f'DPD_{horizon}']
                    
                    actual_wvht = actual['WVHT']
                    actual_dpd = actual['DPD']
                    
                    if actual_wvht is not None and not (isinstance(actual_wvht, float) and np.isnan(actual_wvht)):
                        wvht_error = pred_wvht - actual_wvht
                        wvht_error_pct = (wvht_error / actual_wvht) * 100 if actual_wvht != 0 else 0
                        wvht_status = "✅" if abs(wvht_error) < 0.5 else "⚠️" if abs(wvht_error) < 1.0 else "❌"
                    else:
                        wvht_error = None
                        wvht_error_pct = None
                        wvht_status = "N/A"
                    
                    if actual_dpd is not None and not (isinstance(actual_dpd, float) and np.isnan(actual_dpd)):
                        dpd_error = pred_dpd - actual_dpd
                        dpd_error_pct = (dpd_error / actual_dpd) * 100 if actual_dpd != 0 else 0
                        dpd_status = "✅" if abs(dpd_error) < 2.0 else "⚠️" if abs(dpd_error) < 4.0 else "❌"
                    else:
                        dpd_error = None
                        dpd_error_pct = None
                        dpd_status = "N/A"
                    
                    # Format display
                    pred_str = f"{self.meters_to_feet(pred_wvht):.1f}ft, {pred_dpd:.1f}s"
                    
                    if actual_wvht is not None and not np.isnan(actual_wvht):
                        actual_str = f"{self.meters_to_feet(actual_wvht):.1f}ft, {actual_dpd:.1f}s"
                        wvht_err_str = f"{wvht_error:+.2f}m ({wvht_error_pct:+.1f}%)" if wvht_error is not None else "N/A"
                        dpd_err_str = f"{dpd_error:+.2f}s ({dpd_error_pct:+.1f}%)" if dpd_error is not None else "N/A"
                        status = f"{wvht_status}/{dpd_status}"
                    else:
                        actual_str = "N/A"
                        wvht_err_str = "N/A"
                        dpd_err_str = "N/A"
                        status = "N/A"
                    
                    pred_time = actual['time']
                    print(f"   {pred_time.strftime('%H:%M'):4} | {pred_str:13} | {actual_str:13} | {wvht_err_str:12} | {dpd_err_str:11} | {status:6}")
                    
                    # Store errors for statistics
                    model_errors[horizon] = {
                        'wvht_error': wvht_error,
                        'dpd_error': dpd_error,
                        'wvht_error_pct': wvht_error_pct,
                        'dpd_error_pct': dpd_error_pct,
                        'predicted': {'WVHT': pred_wvht, 'DPD': pred_dpd},
                        'actual': {'WVHT': actual_wvht, 'DPD': actual_dpd}
                    }
                else:
                    pred_wvht = preds[f'WVHT_{horizon}']
                    pred_dpd = preds[f'DPD_{horizon}']
                    pred_str = f"{self.meters_to_feet(pred_wvht):.1f}ft, {pred_dpd:.1f}s"
                    print(f"   {horizon:4} | {pred_str:13} | No actual data | N/A          | N/A         | N/A")
            
            errors[model_name] = model_errors
            
            # Calculate summary statistics
            valid_wvht_errors = [err['wvht_error'] for err in model_errors.values() if err.get('wvht_error') is not None]
            valid_dpd_errors = [err['dpd_error'] for err in model_errors.values() if err.get('dpd_error') is not None]
            
            if valid_wvht_errors:
                mae_wvht = np.mean(np.abs(valid_wvht_errors))
                rmse_wvht = np.sqrt(np.mean(np.square(valid_wvht_errors)))
                print(f"\n   📈 Wave Height Accuracy:")
                print(f"      MAE: {mae_wvht:.3f}m ({self.meters_to_feet(mae_wvht):.2f}ft)")
                print(f"      RMSE: {rmse_wvht:.3f}m ({self.meters_to_feet(rmse_wvht):.2f}ft)")
            
            if valid_dpd_errors:
                mae_dpd = np.mean(np.abs(valid_dpd_errors))
                rmse_dpd = np.sqrt(np.mean(np.square(valid_dpd_errors)))
                print(f"   📈 Wave Period Accuracy:")
                print(f"      MAE: {mae_dpd:.2f}s")
                print(f"      RMSE: {rmse_dpd:.2f}s")
        
        return errors
    
    def predict_for_timestamp(self, target_datetime_str):
        """Main function to predict surf conditions for a specific timestamp."""
        
        target_datetime = datetime.strptime(target_datetime_str, '%Y-%m-%d %H:%M')
        
        print("=" * 80)
        print("HISTORICAL SURF PREDICTION FOR STATION 46026 (SAN FRANCISCO)")
        print(f"Target Time: {target_datetime.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"Prediction Horizons: 1h, 3h, 6h after target time")
        print("=" * 80)
        
        # Load models
        if not self.load_trained_models():
            print("❌ Failed to load trained models")
            return None
        
        try:
            # Fetch historical data
            buoy_data = self.fetch_historical_data(target_datetime)
            
            if buoy_data.empty:
                print("❌ No historical data available for target time")
                return None
            
            # Engineer features
            features = self.engineer_features(buoy_data)
            
            # Create prediction sequence
            X_sequence = self.create_prediction_sequences(features)
            
            # Make predictions
            predictions = self.make_predictions(X_sequence)
            
            if not predictions:
                print("❌ No predictions generated")
                return None
            
            # Display results
            self.display_predictions(target_datetime, predictions, buoy_data)
            
            # Calculate prediction times
            prediction_times = {
                '1h': target_datetime + timedelta(hours=1),
                '3h': target_datetime + timedelta(hours=3),
                '6h': target_datetime + timedelta(hours=6)
            }
            
            # Fetch actual conditions for validation
            actual_conditions = self.fetch_actual_conditions(prediction_times)
            
            # Calculate and display errors
            prediction_errors = self.calculate_prediction_errors(predictions, actual_conditions)
            
            return {
                'target_datetime': target_datetime,
                'station_id': self.station_id,
                'station_name': self.station_info['name'],
                'predictions': predictions,
                'actual_conditions': actual_conditions,
                'prediction_errors': prediction_errors,
                'current_conditions': {
                    'WVHT': buoy_data['WVHT'].iloc[-1] if 'WVHT' in buoy_data.columns and len(buoy_data) > 0 else None,
                    'DPD': buoy_data['DPD'].iloc[-1] if 'DPD' in buoy_data.columns and len(buoy_data) > 0 else None,
                    'WSPD': buoy_data['WSPD'].iloc[-1] if 'WSPD' in buoy_data.columns and len(buoy_data) > 0 else None
                }
            }
            
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            return None
    
    def display_predictions(self, target_datetime, predictions, current_data):
        """Display formatted predictions."""
        
        print(f"\n🌊 CONDITIONS AT TARGET TIME ({target_datetime.strftime('%Y-%m-%d %H:%M UTC')}):")
        if not current_data.empty:
            latest = current_data.iloc[-1]
            wvht_m = latest.get('WVHT', None)
            if wvht_m is not None and not (isinstance(wvht_m, float) and np.isnan(wvht_m)):
                wvht_ft = self.meters_to_feet(wvht_m)
                print(f"   Wave Height: {wvht_ft:.1f}ft ({wvht_m:.1f}m)")
            else:
                print(f"   Wave Height: N/A")
            
            dpd = latest.get('DPD', None)
            if dpd is not None and not (isinstance(dpd, float) and np.isnan(dpd)):
                print(f"   Wave Period: {dpd:.1f}s")
            else:
                print(f"   Wave Period: N/A")
            
            wspd = latest.get('WSPD', None)
            if wspd is not None and not (isinstance(wspd, float) and np.isnan(wspd)):
                print(f"   Wind Speed: {wspd:.1f} m/s")
            else:
                print(f"   Wind Speed: N/A")
        
        # Display predictions for each model
        for model_name, preds in predictions.items():
            model_display = model_name.upper()
            print(f"\n🔮 {model_display} MODEL PREDICTIONS:")
            print("   Time | Wave Height | Period | Size Category | Quality     | Surfable")
            print("   -----|-------------|--------|---------------|-------------|----------")
            
            for horizon in ['1h', '3h', '6h']:
                pred_time = target_datetime + timedelta(hours=int(horizon[:-1]))
                wvht_m = preds[f'WVHT_{horizon}']
                dpd_s = preds[f'DPD_{horizon}']
                
                # Classify conditions
                conditions = self.classify_surf_conditions(wvht_m, dpd_s)
                
                wvht_ft = conditions['wave_height_ft']
                surfable_emoji = "✅" if conditions['surfable'] else "❌"
                
                print(f"   {pred_time.strftime('%H:%M'):4} | {wvht_ft:6.1f}ft ({wvht_m:4.1f}m) | {dpd_s:5.1f}s | "
                      f"{conditions['size_category']:13} | {conditions['quality']:11} | {surfable_emoji}")


def main():
    """Run historical surf prediction for specific timestamp."""
    
    # Target prediction time: November 30, 2025 19:50 UTC
    target_time = "2025-11-30 19:50"
    
    predictor = Historical46026Predictor()
    results = predictor.predict_for_timestamp(target_time)
    
    if results:
        print("\n" + "=" * 80)
        print("PREDICTION SUMMARY")
        print("=" * 80)
        
        predictions = results['predictions']
        
        print(f"\n📍 {results['station_name']} (Station {results['station_id']}):")
        
        # Show model comparison
        model_names = list(predictions.keys())
        if len(model_names) >= 2:
            model1, model2 = model_names[0], model_names[1]
            
            for horizon in ['1h', '3h', '6h']:
                pred_time = results['target_datetime'] + timedelta(hours=int(horizon[:-1]))
                wvht1 = predictions[model1][f'WVHT_{horizon}']
                wvht2 = predictions[model2][f'WVHT_{horizon}']
                wvht1_ft = predictor.meters_to_feet(wvht1)
                wvht2_ft = predictor.meters_to_feet(wvht2)
                
                print(f"   {pred_time.strftime('%H:%M')} | {model1}: {wvht1_ft:.1f}ft ({wvht1:.1f}m) | "
                      f"{model2}: {wvht2_ft:.1f}ft ({wvht2:.1f}m)")
        
        # Current conditions
        current = results['current_conditions']
        print(f"\n📊 Current Conditions at Target Time:")
        if current['WVHT'] is not None and not (isinstance(current['WVHT'], float) and np.isnan(current['WVHT'])):
            current_ft = predictor.meters_to_feet(current['WVHT'])
            print(f"   Wave Height: {current_ft:.1f}ft ({current['WVHT']:.1f}m)")
        else:
            print(f"   Wave Height: N/A")
        
        if current['DPD'] is not None and not (isinstance(current['DPD'], float) and np.isnan(current['DPD'])):
            print(f"   Wave Period: {current['DPD']:.1f}s")
        else:
            print(f"   Wave Period: N/A")
            
        if current['WSPD'] is not None and not (isinstance(current['WSPD'], float) and np.isnan(current['WSPD'])):
            print(f"   Wind Speed: {current['WSPD']:.1f} m/s")
        else:
            print(f"   Wind Speed: N/A")
    
    else:
        print("❌ No predictions generated")


if __name__ == "__main__":
    main()