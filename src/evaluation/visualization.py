import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_training_curves(history, model_name="Model", save_path=None, show_plot=True):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Training history (Keras history object or dict with 'loss' and 'val_loss')
        model_name: Name of model for title
        save_path: Path to save plot (optional)
        show_plot: Whether to display plot
        
    Returns:
        matplotlib figure object
    """
    
    # Extract loss history
    if hasattr(history, 'history'):
        hist_dict = history.history
    else:
        hist_dict = history
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(hist_dict['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in hist_dict:
        axes[0].plot(hist_dict['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title(f'{model_name} - Training Curves', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Additional metrics if available
    metric_key = None
    for key in hist_dict.keys():
        if 'mae' in key.lower() and 'val' not in key:
            metric_key = key
            break
    
    if metric_key:
        val_metric_key = f'val_{metric_key}'
        axes[1].plot(hist_dict[metric_key], label=f'Training {metric_key.upper()}', linewidth=2)
        if val_metric_key in hist_dict:
            axes[1].plot(hist_dict[val_metric_key], label=f'Validation {metric_key.upper()}', linewidth=2)
        axes[1].set_title(f'{model_name} - {metric_key.upper()} Curves', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_key.upper())
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # If no additional metrics, plot loss in log scale
        axes[1].plot(hist_dict['loss'], label='Training Loss (log scale)', linewidth=2)
        if 'val_loss' in hist_dict:
            axes[1].plot(hist_dict['val_loss'], label='Validation Loss (log scale)', linewidth=2)
        axes[1].set_yscale('log')
        axes[1].set_title(f'{model_name} - Loss (Log Scale)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (MSE, log scale)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    
    return fig

def plot_predictions_vs_actual(y_true, y_pred, target_names, model_name="Model", 
                              save_path=None, show_plot=True):
    """
    Create scatter plots of predictions vs actual values.
    
    Args:
        y_true: True values (n_samples, n_outputs)
        y_pred: Predictions (n_samples, n_outputs)
        target_names: List of target names
        model_name: Name of model for title
        save_path: Path to save plot
        show_plot: Whether to display plot
        
    Returns:
        matplotlib figure object
    """
    
    n_targets = len(target_names)
    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_targets == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, target_name in enumerate(target_names):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Scatter plot
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R² and add to plot
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i])**2))
        
        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'True {target_name}')
        ax.set_ylabel(f'Predicted {target_name}')
        ax.set_title(f'{target_name} - {model_name}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for i in range(n_targets, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f'{model_name} - Predictions vs Actual', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    
    return fig

def plot_time_series_predictions(y_true, y_pred, timestamps, target_names, 
                                model_name="Model", n_samples=1000, save_path=None, show_plot=True):
    """
    Plot time series overlay of actual vs predicted values.
    
    Args:
        y_true: True values (n_samples, n_outputs)
        y_pred: Predictions (n_samples, n_outputs)
        timestamps: Array of timestamps
        target_names: List of target names
        model_name: Name of model for title
        n_samples: Number of samples to plot (for readability)
        save_path: Path to save plot
        show_plot: Whether to display plot
        
    Returns:
        matplotlib figure object
    """
    
    # Sample data if too many points
    if len(y_true) > n_samples:
        indices = np.random.choice(len(y_true), n_samples, replace=False)
        indices = np.sort(indices)  # Keep temporal order
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
        timestamps_plot = timestamps[indices] if timestamps is not None else None
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        timestamps_plot = timestamps
    
    n_targets = len(target_names)
    fig, axes = plt.subplots(n_targets, 1, figsize=(15, 4*n_targets))
    if n_targets == 1:
        axes = [axes]
    
    x_axis = timestamps_plot if timestamps_plot is not None else np.arange(len(y_true_plot))
    
    for i, target_name in enumerate(target_names):
        axes[i].plot(x_axis, y_true_plot[:, i], label='Actual', linewidth=2, alpha=0.8)
        axes[i].plot(x_axis, y_pred_plot[:, i], label='Predicted', linewidth=2, alpha=0.8)
        
        axes[i].set_title(f'{target_name} - {model_name}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(target_name)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        if timestamps_plot is not None:
            axes[i].tick_params(axis='x', rotation=45)
    
    axes[-1].set_xlabel('Time')
    plt.suptitle(f'{model_name} - Time Series Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    
    return fig

def plot_error_distributions(y_true, y_pred, target_names, model_name="Model", 
                           save_path=None, show_plot=True):
    """
    Plot error distribution histograms and box plots.
    
    Args:
        y_true: True values (n_samples, n_outputs)
        y_pred: Predictions (n_samples, n_outputs)  
        target_names: List of target names
        model_name: Name of model for title
        save_path: Path to save plot
        show_plot: Whether to display plot
        
    Returns:
        matplotlib figure object
    """
    
    errors = y_pred - y_true
    n_targets = len(target_names)
    
    fig, axes = plt.subplots(2, n_targets, figsize=(5*n_targets, 8))
    if n_targets == 1:
        axes = axes.reshape(2, 1)
    
    for i, target_name in enumerate(target_names):
        # Histogram
        axes[0, i].hist(errors[:, i], bins=50, alpha=0.7, density=True)
        axes[0, i].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0, i].set_title(f'{target_name} - Error Distribution', fontweight='bold')
        axes[0, i].set_xlabel('Prediction Error')
        axes[0, i].set_ylabel('Density')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Box plot
        axes[1, i].boxplot([errors[:, i]], labels=[target_name])
        axes[1, i].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1, i].set_title(f'{target_name} - Error Box Plot', fontweight='bold')
        axes[1, i].set_ylabel('Prediction Error')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Error Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    
    return fig

def plot_horizon_comparison(results_dict, metric='rmse', save_path=None, show_plot=True):
    """
    Create bar chart comparing performance across forecast horizons.
    
    Args:
        results_dict: Dictionary of model results
        metric: Metric to compare ('rmse', 'mae', or 'r2')
        save_path: Path to save plot
        show_plot: Whether to display plot
        
    Returns:
        matplotlib figure object
    """
    
    # Extract data for plotting
    models = list(results_dict.keys())
    horizons = ['1h', '3h', '6h']
    variables = ['WVHT', 'DPD']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for var_idx, variable in enumerate(['WVHT', 'DPD']):
        data_to_plot = []
        model_names = []
        
        for model_name, results in results_dict.items():
            model_data = []
            for horizon in horizons:
                target_name = f"{variable}_{horizon}"
                if target_name in results['primary_metrics']['by_target']:
                    value = results['primary_metrics']['by_target'][target_name][metric]
                    model_data.append(value)
                else:
                    model_data.append(np.nan)
            
            data_to_plot.append(model_data)
            model_names.append(model_name)
        
        # Create bar chart
        x = np.arange(len(horizons))
        width = 0.35
        
        for i, (model_data, model_name) in enumerate(zip(data_to_plot, model_names)):
            offset = (i - len(model_names)/2 + 0.5) * width
            axes[var_idx].bar(x + offset, model_data, width, label=model_name, alpha=0.8)
        
        axes[var_idx].set_xlabel('Forecast Horizon')
        axes[var_idx].set_ylabel(metric.upper())
        axes[var_idx].set_title(f'{variable} {metric.upper()} by Horizon', fontweight='bold')
        axes[var_idx].set_xticks(x)
        axes[var_idx].set_xticklabels(horizons)
        axes[var_idx].legend()
        axes[var_idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Model Comparison - {metric.upper()} by Forecast Horizon', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    
    return fig

def plot_feature_importance(feature_importance, feature_names, model_name="Model",
                          top_n=20, save_path=None, show_plot=True):
    """
    Plot feature importance bar chart.
    
    Args:
        feature_importance: Array of feature importance values
        feature_names: List of feature names
        model_name: Name of model for title
        top_n: Number of top features to show
        save_path: Path to save plot
        show_plot: Whether to display plot
        
    Returns:
        matplotlib figure object
    """
    
    # Sort features by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Select top N features
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(range(len(top_features)), top_features['importance'].values, alpha=0.8)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{model_name} - Top {top_n} Feature Importances', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    
    return fig

def plot_surf_classification_confusion_matrix(y_true, y_pred, target_names, 
                                            good_surf_threshold={'WVHT': 1.2, 'DPD': 12.0},
                                            model_name="Model", save_path=None, show_plot=True):
    """
    Plot confusion matrices for good surf classification.
    
    Args:
        y_true: True values (n_samples, n_outputs)
        y_pred: Predictions (n_samples, n_outputs)
        target_names: List of target names
        good_surf_threshold: Thresholds for good surf classification
        model_name: Name of model for title
        save_path: Path to save plot
        show_plot: Whether to display plot
        
    Returns:
        matplotlib figure object
    """
    
    # Find WVHT and DPD indices for different horizons
    wvht_indices = [(i, name) for i, name in enumerate(target_names) if 'WVHT' in name]
    dpd_indices = [(i, name) for i, name in enumerate(target_names) if 'DPD' in name]
    
    n_horizons = len(wvht_indices)
    fig, axes = plt.subplots(1, n_horizons, figsize=(5*n_horizons, 4))
    if n_horizons == 1:
        axes = [axes]
    
    for i, ((wvht_idx, wvht_name), (dpd_idx, dpd_name)) in enumerate(zip(wvht_indices, dpd_indices)):
        horizon = wvht_name.split('_')[-1]
        
        # Define good surf conditions
        good_surf_true = ((y_true[:, wvht_idx] >= good_surf_threshold['WVHT']) & 
                         (y_true[:, dpd_idx] >= good_surf_threshold['DPD']))
        good_surf_pred = ((y_pred[:, wvht_idx] >= good_surf_threshold['WVHT']) & 
                         (y_pred[:, dpd_idx] >= good_surf_threshold['DPD']))
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(good_surf_true, good_surf_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Poor Surf', 'Good Surf'],
                   yticklabels=['Poor Surf', 'Good Surf'])
        
        axes[i].set_title(f'Good Surf Classification - {horizon}\n{model_name}', 
                         fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    
    return fig

def create_model_summary_dashboard(results_dict, save_path=None, show_plot=True):
    """
    Create a comprehensive dashboard summarizing all model results.
    
    Args:
        results_dict: Dictionary of model results
        save_path: Path to save plot
        show_plot: Whether to display plot
        
    Returns:
        matplotlib figure object
    """
    
    fig = plt.figure(figsize=(20, 12))
    
    # Overall performance comparison
    plt.subplot(2, 3, 1)
    models = list(results_dict.keys())
    overall_rmse = [results['primary_metrics']['overall']['rmse'] for results in results_dict.values()]
    
    bars = plt.bar(models, overall_rmse, alpha=0.8)
    plt.title('Overall RMSE Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, overall_rmse):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Wave height performance by horizon
    plt.subplot(2, 3, 2)
    horizons = ['1h', '3h', '6h']
    x = np.arange(len(horizons))
    width = 0.35
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        wvht_rmse = []
        for horizon in horizons:
            target_name = f"WVHT_{horizon}"
            if target_name in results['primary_metrics']['by_target']:
                rmse = results['primary_metrics']['by_target'][target_name]['rmse']
                wvht_rmse.append(rmse)
            else:
                wvht_rmse.append(np.nan)
        
        offset = (i - len(results_dict)/2 + 0.5) * width
        plt.bar(x + offset, wvht_rmse, width, label=model_name, alpha=0.8)
    
    plt.title('Wave Height RMSE by Horizon', fontsize=14, fontweight='bold')
    plt.xlabel('Forecast Horizon')
    plt.ylabel('RMSE (meters)')
    plt.xticks(x, horizons)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Wave period performance by horizon  
    plt.subplot(2, 3, 3)
    for i, (model_name, results) in enumerate(results_dict.items()):
        dpd_rmse = []
        for horizon in horizons:
            target_name = f"DPD_{horizon}"
            if target_name in results['primary_metrics']['by_target']:
                rmse = results['primary_metrics']['by_target'][target_name]['rmse']
                dpd_rmse.append(rmse)
            else:
                dpd_rmse.append(np.nan)
        
        offset = (i - len(results_dict)/2 + 0.5) * width
        plt.bar(x + offset, dpd_rmse, width, label=model_name, alpha=0.8)
    
    plt.title('Wave Period RMSE by Horizon', fontsize=14, fontweight='bold')
    plt.xlabel('Forecast Horizon')
    plt.ylabel('RMSE (seconds)')
    plt.xticks(x, horizons)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # R² comparison
    plt.subplot(2, 3, 4)
    overall_r2 = [results['primary_metrics']['overall']['r2'] for results in results_dict.values()]
    
    bars = plt.bar(models, overall_r2, alpha=0.8, color='green')
    plt.title('Overall R² Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, overall_r2):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05,
                f'{value:.3f}', ha='center', va='top', fontweight='bold', color='white')
    
    # Improvement over baseline
    plt.subplot(2, 3, 5)
    improvements = []
    model_names_with_improvement = []
    
    for model_name, results in results_dict.items():
        if 'comparative_metrics' in results:
            improvement = results['comparative_metrics']['overall_improvement']['improvement_pct']
            improvements.append(improvement)
            model_names_with_improvement.append(model_name)
    
    if improvements:
        bars = plt.bar(model_names_with_improvement, improvements, alpha=0.8, color='orange')
        plt.title('Improvement over Baseline (%)', fontsize=14, fontweight='bold')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        plt.axhline(y=20, color='red', linestyle='--', linewidth=2, label='PRD Target (20%)')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, improvements):
            color = 'white' if value > 30 else 'black'
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{value:.1f}%', ha='center', va='center', fontweight='bold', color=color)
    
    # Performance summary table (text)
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Create summary text
    summary_text = "Model Performance Summary\n\n"
    for model_name, results in results_dict.items():
        primary = results['primary_metrics']
        summary_text += f"{model_name}:\n"
        summary_text += f"  RMSE: {primary['overall']['rmse']:.3f}\n"
        summary_text += f"  MAE:  {primary['overall']['mae']:.3f}\n"
        summary_text += f"  R²:   {primary['overall']['r2']:.3f}\n"
        
        if 'comparative_metrics' in results:
            improvement = results['comparative_metrics']['overall_improvement']['improvement_pct']
            summary_text += f"  Improvement: {improvement:.1f}%\n"
        
        summary_text += "\n"
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Model Performance Dashboard', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    
    return fig