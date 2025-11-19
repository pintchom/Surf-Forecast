import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import warnings

def calculate_primary_metrics(y_true, y_pred, target_names=None):
    """
    Calculate primary evaluation metrics (RMSE, MAE, R²).
    
    Args:
        y_true: True values (n_samples, n_outputs)
        y_pred: Predictions (n_samples, n_outputs)
        target_names: List of target names (e.g., ['WVHT_1h', 'DPD_1h', ...])
        
    Returns:
        Dictionary with overall and per-target metrics
    """
    
    if target_names is None:
        target_names = [f"target_{i}" for i in range(y_true.shape[1])]
    
    metrics = {
        'overall': {},
        'by_target': {}
    }
    
    # Overall metrics
    metrics['overall']['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['overall']['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['overall']['r2'] = r2_score(y_true, y_pred)
    
    # Per-target metrics
    n_targets = y_true.shape[1]
    for i, target_name in enumerate(target_names):
        target_metrics = {}
        target_metrics['rmse'] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        target_metrics['mae'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
        target_metrics['r2'] = r2_score(y_true[:, i], y_pred[:, i])
        
        metrics['by_target'][target_name] = target_metrics
    
    return metrics

def calculate_surf_specific_metrics(y_true, y_pred, target_names, 
                                  good_surf_threshold={'WVHT': 1.2, 'DPD': 12.0},
                                  extreme_wave_threshold=3.0):
    """
    Calculate surf-specific evaluation metrics.
    
    Args:
        y_true: True values (n_samples, n_outputs)
        y_pred: Predictions (n_samples, n_outputs) 
        target_names: List of target names
        good_surf_threshold: Dict with WVHT and DPD thresholds for good surf
        extreme_wave_threshold: Threshold for extreme wave detection (meters)
        
    Returns:
        Dictionary with surf-specific metrics
    """
    
    surf_metrics = {}
    
    # Find WVHT and DPD indices for different horizons
    wvht_indices = [(i, name) for i, name in enumerate(target_names) if 'WVHT' in name]
    dpd_indices = [(i, name) for i, name in enumerate(target_names) if 'DPD' in name]
    
    # Good surf classification for each horizon
    for (wvht_idx, wvht_name), (dpd_idx, dpd_name) in zip(wvht_indices, dpd_indices):
        horizon = wvht_name.split('_')[-1]  # Extract horizon (e.g., '1h')
        
        # Define good surf conditions
        good_surf_true = ((y_true[:, wvht_idx] >= good_surf_threshold['WVHT']) & 
                         (y_true[:, dpd_idx] >= good_surf_threshold['DPD']))
        good_surf_pred = ((y_pred[:, wvht_idx] >= good_surf_threshold['WVHT']) & 
                         (y_pred[:, dpd_idx] >= good_surf_threshold['DPD']))
        
        # Calculate classification metrics
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            surf_metrics[f'good_surf_{horizon}'] = {
                'accuracy': accuracy_score(good_surf_true, good_surf_pred),
                'precision': precision_score(good_surf_true, good_surf_pred, zero_division=0),
                'recall': recall_score(good_surf_true, good_surf_pred, zero_division=0),
                'f1_score': f1_score(good_surf_true, good_surf_pred, zero_division=0),
                'true_positives': np.sum(good_surf_true & good_surf_pred),
                'total_good_surf_days': np.sum(good_surf_true)
            }
        except Exception as e:
            warnings.warn(f"Could not calculate good surf metrics for {horizon}: {e}")
    
    # Extreme wave detection (safety metric)
    for wvht_idx, wvht_name in wvht_indices:
        horizon = wvht_name.split('_')[-1]
        
        extreme_true = y_true[:, wvht_idx] >= extreme_wave_threshold
        extreme_pred = y_pred[:, wvht_idx] >= extreme_wave_threshold
        
        if np.sum(extreme_true) > 0:  # Only if there are extreme waves to detect
            try:
                from sklearn.metrics import precision_score, recall_score
                
                surf_metrics[f'extreme_wave_{horizon}'] = {
                    'precision': precision_score(extreme_true, extreme_pred, zero_division=0),
                    'recall': recall_score(extreme_true, extreme_pred, zero_division=0),
                    'extreme_wave_days': np.sum(extreme_true),
                    'detected_extreme_days': np.sum(extreme_pred)
                }
            except Exception as e:
                warnings.warn(f"Could not calculate extreme wave metrics for {horizon}: {e}")
    
    return surf_metrics

def calculate_comparative_metrics(baseline_metrics, model_metrics, model_name="Model"):
    """
    Calculate improvement metrics compared to baseline.
    
    Args:
        baseline_metrics: Baseline model metrics dictionary
        model_metrics: Model metrics dictionary  
        model_name: Name of the model being compared
        
    Returns:
        Dictionary with comparative metrics
    """
    
    comparative = {
        'model_name': model_name,
        'overall_improvement': {},
        'by_target_improvement': {}
    }
    
    # Overall improvement
    baseline_rmse = baseline_metrics['overall']['rmse']
    model_rmse = model_metrics['overall']['rmse']
    
    improvement_pct = ((baseline_rmse - model_rmse) / baseline_rmse) * 100
    comparative['overall_improvement'] = {
        'baseline_rmse': baseline_rmse,
        'model_rmse': model_rmse,
        'improvement_pct': improvement_pct,
        'meets_20pct_target': improvement_pct >= 20.0
    }
    
    # Per-target improvement
    for target_name in baseline_metrics['by_target'].keys():
        baseline_target_rmse = baseline_metrics['by_target'][target_name]['rmse']
        model_target_rmse = model_metrics['by_target'][target_name]['rmse']
        
        target_improvement = ((baseline_target_rmse - model_target_rmse) / baseline_target_rmse) * 100
        
        comparative['by_target_improvement'][target_name] = {
            'baseline_rmse': baseline_target_rmse,
            'model_rmse': model_target_rmse,
            'improvement_pct': target_improvement
        }
    
    return comparative

def statistical_significance_test(y_true, y_pred_baseline, y_pred_model, alpha=0.05):
    """
    Perform paired t-test to assess if model improvement is statistically significant.
    
    Args:
        y_true: True values (n_samples, n_outputs)
        y_pred_baseline: Baseline predictions (n_samples, n_outputs)
        y_pred_model: Model predictions (n_samples, n_outputs)
        alpha: Significance level
        
    Returns:
        Dictionary with statistical test results
    """
    
    # Calculate squared errors for each prediction
    baseline_errors = (y_true - y_pred_baseline) ** 2
    model_errors = (y_true - y_pred_model) ** 2
    
    # Flatten for overall test
    baseline_errors_flat = baseline_errors.flatten()
    model_errors_flat = model_errors.flatten()
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_errors_flat, model_errors_flat)
    
    significance_test = {
        'test_type': 'paired_t_test',
        'null_hypothesis': 'Model and baseline have equal performance',
        'alternative': 'Model has significantly different performance',
        't_statistic': t_stat,
        'p_value': p_value,
        'alpha': alpha,
        'is_significant': p_value < alpha,
        'conclusion': 'significant' if p_value < alpha else 'not_significant'
    }
    
    return significance_test

def calculate_confidence_intervals(y_true, y_pred, confidence=0.95, n_bootstrap=1000):
    """
    Calculate confidence intervals for RMSE using bootstrap resampling.
    
    Args:
        y_true: True values (n_samples, n_outputs)
        y_pred: Predictions (n_samples, n_outputs)
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with confidence intervals
    """
    
    n_samples = len(y_true)
    bootstrap_rmse = []
    
    # Bootstrap resampling
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate RMSE for bootstrap sample
        rmse_boot = np.sqrt(mean_squared_error(y_true_boot, y_pred_boot))
        bootstrap_rmse.append(rmse_boot)
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_rmse, lower_percentile)
    ci_upper = np.percentile(bootstrap_rmse, upper_percentile)
    
    confidence_intervals = {
        'confidence_level': confidence,
        'n_bootstrap': n_bootstrap,
        'rmse_mean': np.mean(bootstrap_rmse),
        'rmse_std': np.std(bootstrap_rmse),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower
    }
    
    return confidence_intervals

def evaluate_model_comprehensive(y_true, y_pred, target_names, model_name="Model",
                               baseline_metrics=None, calculate_surf_metrics=True):
    """
    Comprehensive model evaluation with all metrics.
    
    Args:
        y_true: True values (n_samples, n_outputs)
        y_pred: Predictions (n_samples, n_outputs)
        target_names: List of target names
        model_name: Name of model for reporting
        baseline_metrics: Baseline metrics for comparison (optional)
        calculate_surf_metrics: Whether to calculate surf-specific metrics
        
    Returns:
        Complete evaluation results dictionary
    """
    
    results = {
        'model_name': model_name,
        'evaluation_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Primary metrics
    results['primary_metrics'] = calculate_primary_metrics(y_true, y_pred, target_names)
    
    # Surf-specific metrics
    if calculate_surf_metrics:
        try:
            results['surf_metrics'] = calculate_surf_specific_metrics(y_true, y_pred, target_names)
        except Exception as e:
            warnings.warn(f"Could not calculate surf metrics: {e}")
            results['surf_metrics'] = {}
    
    # Comparative metrics
    if baseline_metrics is not None:
        results['comparative_metrics'] = calculate_comparative_metrics(
            baseline_metrics, results['primary_metrics'], model_name
        )
    
    # Confidence intervals
    try:
        results['confidence_intervals'] = calculate_confidence_intervals(y_true, y_pred)
    except Exception as e:
        warnings.warn(f"Could not calculate confidence intervals: {e}")
    
    return results

def print_metrics_summary(results, show_targets=True, show_surf=True, show_comparison=True):
    """
    Print a formatted summary of evaluation results.
    
    Args:
        results: Results dictionary from evaluate_model_comprehensive
        show_targets: Whether to show per-target metrics
        show_surf: Whether to show surf-specific metrics  
        show_comparison: Whether to show baseline comparison
    """
    
    print(f"\n{results['model_name']} Evaluation Results")
    print("=" * 60)
    
    # Primary metrics
    primary = results['primary_metrics']
    print(f"\nOverall Performance:")
    print(f"  RMSE: {primary['overall']['rmse']:.4f}")
    print(f"  MAE:  {primary['overall']['mae']:.4f}")
    print(f"  R²:   {primary['overall']['r2']:.4f}")
    
    # Confidence intervals
    if 'confidence_intervals' in results:
        ci = results['confidence_intervals']
        print(f"  95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    
    # Per-target metrics
    if show_targets:
        print(f"\nBy Target:")
        for target_name, metrics in primary['by_target'].items():
            print(f"  {target_name}:")
            print(f"    RMSE: {metrics['rmse']:.4f}")
            print(f"    MAE:  {metrics['mae']:.4f}")
            print(f"    R²:   {metrics['r2']:.4f}")
    
    # Surf-specific metrics
    if show_surf and 'surf_metrics' in results:
        surf = results['surf_metrics']
        if surf:
            print(f"\nSurf-Specific Metrics:")
            for metric_name, values in surf.items():
                if 'good_surf' in metric_name:
                    horizon = metric_name.split('_')[-1]
                    print(f"  Good Surf Classification ({horizon}):")
                    print(f"    Accuracy:  {values['accuracy']:.3f}")
                    print(f"    Precision: {values['precision']:.3f}")
                    print(f"    Recall:    {values['recall']:.3f}")
                    print(f"    F1-Score:  {values['f1_score']:.3f}")
    
    # Comparison metrics
    if show_comparison and 'comparative_metrics' in results:
        comp = results['comparative_metrics']
        print(f"\nBaseline Comparison:")
        print(f"  Baseline RMSE: {comp['overall_improvement']['baseline_rmse']:.4f}")
        print(f"  Model RMSE:    {comp['overall_improvement']['model_rmse']:.4f}")
        improvement = comp['overall_improvement']['improvement_pct']
        print(f"  Improvement:   {improvement:.1f}%")
        
        meets_target = comp['overall_improvement']['meets_20pct_target']
        status = "✅ PASS" if meets_target else "❌ FAIL"
        print(f"  20% Target:    {status}")

def check_prd_targets(results, target_names):
    """
    Check if model meets PRD performance targets.
    
    Args:
        results: Results dictionary from evaluation
        target_names: List of target names
        
    Returns:
        Dictionary with PRD compliance status
    """
    
    prd_targets = {
        'WVHT_1h': {'rmse_target': 0.30, 'r2_target': 0.85},
        'WVHT_6h': {'rmse_target': 0.50, 'r2_target': 0.65},
        'overall_improvement': {'target': 20.0}  # percentage
    }
    
    compliance = {}
    
    # Wave height targets
    for target, thresholds in prd_targets.items():
        if target in target_names:
            metrics = results['primary_metrics']['by_target'][target]
            
            compliance[target] = {
                'rmse_actual': metrics['rmse'],
                'rmse_target': thresholds['rmse_target'],
                'rmse_meets_target': metrics['rmse'] <= thresholds['rmse_target'],
                'r2_actual': metrics['r2'],
                'r2_target': thresholds['r2_target'],
                'r2_meets_target': metrics['r2'] >= thresholds['r2_target']
            }
    
    # Improvement target
    if 'comparative_metrics' in results:
        improvement = results['comparative_metrics']['overall_improvement']['improvement_pct']
        compliance['improvement'] = {
            'actual': improvement,
            'target': prd_targets['overall_improvement']['target'],
            'meets_target': improvement >= prd_targets['overall_improvement']['target']
        }
    
    return compliance