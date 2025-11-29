import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

def analyze_feature_importance(station_id):
    """Analyze feature importance from trained linear models."""
    
    print(f"=== Feature Importance Analysis for Station {station_id} ===")
    
    # Load metadata to get feature names
    sequences_path = Path(f"data/splits/{station_id}/sequences")
    with open(sequences_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    feature_cols = metadata['feature_columns']
    target_names = metadata['target_names']
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Targets: {target_names}")
    
    # Load trained models
    ridge_model = joblib.load(sequences_path / 'models' / 'ridge_best.pkl')
    lasso_model = joblib.load(sequences_path / 'models' / 'enhanced_best.pkl')  # This is likely Lasso
    
    print(f"\nModel types loaded:")
    print(f"Ridge: {type(ridge_model)}")
    print(f"Enhanced (Lasso): {type(lasso_model)}")
    
    # Get model coefficients
    ridge_coefs = ridge_model.coef_
    lasso_coefs = lasso_model.coef_
    
    print(f"\nCoefficient shapes:")
    print(f"Ridge: {ridge_coefs.shape}")
    print(f"Lasso: {lasso_coefs.shape}")
    
    # Create expanded feature names (feature_name_hour_X format)
    lookback_hours = 24
    expanded_feature_names = []
    for hour in range(lookback_hours):
        for feat in feature_cols:
            expanded_feature_names.append(f"{feat}_hour_{hour}")
    
    print(f"Expected features: {len(expanded_feature_names)}")
    print(f"Model coefficient length: {ridge_coefs.shape[1]}")
    
    # For each target, analyze top features
    results = {}
    
    for target_idx, target_name in enumerate(target_names):
        print(f"\n{'='*50}")
        print(f"TOP FEATURES FOR {target_name}")
        print(f"{'='*50}")
        
        # Get coefficients for this target
        ridge_target_coefs = ridge_coefs[target_idx] if len(ridge_coefs.shape) > 1 else ridge_coefs
        lasso_target_coefs = lasso_coefs[target_idx] if len(lasso_coefs.shape) > 1 else lasso_coefs
        
        # Create feature importance dataframe with expanded names
        importance_df = pd.DataFrame({
            'feature': expanded_feature_names,
            'ridge_coef': ridge_target_coefs,
            'lasso_coef': lasso_target_coefs,
            'ridge_abs': np.abs(ridge_target_coefs),
            'lasso_abs': np.abs(lasso_target_coefs)
        })
        
        # Also aggregate by base feature name (sum across all hours)
        base_feature_importance = {}
        for base_feat in feature_cols:
            # Find all coefficients for this base feature across hours
            base_feat_mask = importance_df['feature'].str.startswith(f"{base_feat}_hour_")
            ridge_sum = importance_df.loc[base_feat_mask, 'ridge_abs'].sum()
            lasso_sum = importance_df.loc[base_feat_mask, 'lasso_abs'].sum()
            base_feature_importance[base_feat] = {'ridge_total': ridge_sum, 'lasso_total': lasso_sum}
        
        aggregated_df = pd.DataFrame.from_dict(base_feature_importance, orient='index')
        aggregated_df = aggregated_df.reset_index().rename(columns={'index': 'base_feature'})
        
        # Sort by absolute coefficient values
        top_ridge = importance_df.nlargest(15, 'ridge_abs')
        top_lasso = importance_df.nlargest(15, 'lasso_abs')
        top_ridge_aggregated = aggregated_df.nlargest(10, 'ridge_total')
        top_lasso_aggregated = aggregated_df.nlargest(10, 'lasso_total')
        
        print(f"\nTOP 10 BASE FEATURES (AGGREGATED) - RIDGE MODEL:")
        print("-" * 60)
        for idx, row in top_ridge_aggregated.iterrows():
            print(f"{row['base_feature']:30} | Total Importance: {row['ridge_total']:8.4f}")
        
        print(f"\nTOP 10 BASE FEATURES (AGGREGATED) - LASSO MODEL:")
        print("-" * 60)
        for idx, row in top_lasso_aggregated.iterrows():
            print(f"{row['base_feature']:30} | Total Importance: {row['lasso_total']:8.4f}")
        
        print(f"\nTOP 10 INDIVIDUAL TIMESTEP FEATURES - RIDGE MODEL:")
        print("-" * 80)
        for idx, row in top_ridge.head(10).iterrows():
            print(f"{row['feature']:45} | Coef: {row['ridge_coef']:8.4f}")
        
        print(f"\nTOP 10 INDIVIDUAL TIMESTEP FEATURES - LASSO MODEL:")
        print("-" * 80)
        for idx, row in top_lasso.head(10).iterrows():
            print(f"{row['feature']:45} | Coef: {row['lasso_coef']:8.4f}")
        
        # Count non-zero features in Lasso (feature selection effect)
        lasso_nonzero = np.sum(np.abs(lasso_target_coefs) > 1e-6)
        total_features = len(expanded_feature_names)
        print(f"\nLasso Feature Selection:")
        print(f"Non-zero features: {lasso_nonzero}/{total_features} ({lasso_nonzero/total_features*100:.1f}%)")
        
        results[target_name] = {
            'top_ridge': top_ridge,
            'top_lasso': top_lasso,
            'top_ridge_aggregated': top_ridge_aggregated,
            'top_lasso_aggregated': top_lasso_aggregated,
            'lasso_nonzero_count': lasso_nonzero
        }
    
    # Analyze patterns across targets
    print(f"\n{'='*60}")
    print("OVERALL FEATURE IMPORTANCE PATTERNS")
    print(f"{'='*60}")
    
    # Find most commonly important features across all targets
    all_important_features = set()
    for target_name, result in results.items():
        all_important_features.update(result['top_ridge']['feature'].tolist())
        all_important_features.update(result['top_lasso']['feature'].tolist())
    
    # Analyze feature types
    feature_types = {
        'lag_features': [f for f in all_important_features if '_lag_' in f],
        'rolling_stats': [f for f in all_important_features if '_mean' in f or '_std' in f or '_min' in f or '_max' in f],
        'temporal': [f for f in all_important_features if 'hour' in f or 'day' in f],
        'pressure': [f for f in all_important_features if 'PRES' in f or 'pressure' in f],
        'wave_power': [f for f in all_important_features if 'wave_power' in f],
        'wind': [f for f in all_important_features if 'WSPD' in f],
        'base_features': [f for f in all_important_features if f in ['WSPD', 'PRES', 'WTMP']]
    }
    
    print("\nFEATURE TYPE ANALYSIS:")
    for feature_type, features in feature_types.items():
        if features:
            print(f"\n{feature_type.upper()} ({len(features)} features):")
            for feat in sorted(features)[:5]:  # Show top 5
                print(f"  - {feat}")
    
    return results

def analyze_good_surf_classification(station_id):
    """Analyze what features predict 'good surf' conditions."""
    
    print(f"\n{'='*60}")
    print(f"GOOD SURF PREDICTION ANALYSIS - Station {station_id}")
    print(f"{'='*60}")
    
    # Load data
    sequences_path = Path(f"data/splits/{station_id}/sequences")
    X_test_flat = np.load(sequences_path / 'X_test_flat.npy')
    y_test = np.load(sequences_path / 'y_test.npy')
    
    with open(sequences_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    feature_cols = metadata['feature_columns']
    
    # Load trained model (use Lasso for feature selection)
    lasso_model = joblib.load(sequences_path / 'models' / 'enhanced_best.pkl')
    
    # Make predictions
    y_pred = lasso_model.predict(X_test_flat)
    
    # Extract WVHT and DPD predictions (assuming order is WVHT_1h, DPD_1h, WVHT_3h, DPD_3h, WVHT_6h, DPD_6h)
    wvht_1h_pred = y_pred[:, 0]  # WVHT_1h predictions
    dpd_1h_pred = y_pred[:, 1]   # DPD_1h predictions
    
    # Create good surf classification
    good_surf_pred = (wvht_1h_pred >= 1.2) & (dpd_1h_pred >= 12.0)
    good_surf_actual = (y_test[:, 0] >= 1.2) & (y_test[:, 1] >= 12.0)
    
    print(f"Good surf conditions:")
    print(f"Predicted: {np.sum(good_surf_pred)}/{len(good_surf_pred)} ({np.mean(good_surf_pred)*100:.1f}%)")
    print(f"Actual: {np.sum(good_surf_actual)}/{len(good_surf_actual)} ({np.mean(good_surf_actual)*100:.1f}%)")
    
    # Classification accuracy
    accuracy = np.mean(good_surf_pred == good_surf_actual)
    print(f"Good surf classification accuracy: {accuracy:.3f}")
    
    # Analyze which input features are most predictive of good surf
    good_surf_indices = np.where(good_surf_actual)[0]
    poor_surf_indices = np.where(~good_surf_actual)[0]
    
    # Sample equal numbers to avoid bias
    n_samples = min(len(good_surf_indices), len(poor_surf_indices), 1000)
    good_sample_idx = np.random.choice(good_surf_indices, n_samples, replace=False)
    poor_sample_idx = np.random.choice(poor_surf_indices, n_samples, replace=False)
    
    good_surf_features = X_test_flat[good_sample_idx]
    poor_surf_features = X_test_flat[poor_sample_idx]
    
    # Calculate feature differences
    feature_diff = np.mean(good_surf_features, axis=0) - np.mean(poor_surf_features, axis=0)
    feature_importance_for_good_surf = np.abs(feature_diff)
    
    # Create dataframe and sort
    surf_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'good_surf_mean': np.mean(good_surf_features, axis=0),
        'poor_surf_mean': np.mean(poor_surf_features, axis=0), 
        'difference': feature_diff,
        'abs_difference': feature_importance_for_good_surf
    })
    
    top_surf_features = surf_importance_df.nlargest(15, 'abs_difference')
    
    print(f"\nTOP 15 FEATURES FOR IDENTIFYING GOOD SURF:")
    print("-" * 80)
    print(f"{'Feature':<30} {'Good Surf':<10} {'Poor Surf':<10} {'Difference':<12}")
    print("-" * 80)
    
    for idx, row in top_surf_features.iterrows():
        print(f"{row['feature']:<30} {row['good_surf_mean']:<10.3f} {row['poor_surf_mean']:<10.3f} {row['difference']:<12.3f}")
    
    return surf_importance_df

if __name__ == "__main__":
    for station in ['46012', '46221']:
        try:
            # Analyze feature importance for all targets
            results = analyze_feature_importance(station)
            
            # Analyze good surf prediction specifically  
            surf_analysis = analyze_good_surf_classification(station)
            
            print(f"\n✅ Analysis complete for station {station}")
            
        except Exception as e:
            print(f"❌ Error analyzing station {station}: {e}")
            import traceback
            traceback.print_exc()
            continue