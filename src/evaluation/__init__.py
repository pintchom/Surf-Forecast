"""
Evaluation module for surf forecasting models.

This module provides comprehensive evaluation tools including:
- Primary metrics (RMSE, MAE, R²)
- Surf-specific metrics (good surf classification, extreme wave detection)
- Statistical significance testing
- Visualization tools for model performance analysis
- Model comparison utilities
"""

from .metrics import (
    calculate_primary_metrics,
    calculate_surf_specific_metrics,
    calculate_comparative_metrics,
    statistical_significance_test,
    calculate_confidence_intervals,
    evaluate_model_comprehensive,
    print_metrics_summary,
    check_prd_targets
)

from .visualization import (
    plot_training_curves,
    plot_predictions_vs_actual,
    plot_time_series_predictions,
    plot_error_distributions,
    plot_horizon_comparison,
    plot_feature_importance,
    plot_surf_classification_confusion_matrix,
    create_model_summary_dashboard
)

from .model_comparison import (
    ModelComparison,
    load_model_results,
    compare_saved_results
)

__all__ = [
    # Metrics
    'calculate_primary_metrics',
    'calculate_surf_specific_metrics', 
    'calculate_comparative_metrics',
    'statistical_significance_test',
    'calculate_confidence_intervals',
    'evaluate_model_comprehensive',
    'print_metrics_summary',
    'check_prd_targets',
    
    # Visualization
    'plot_training_curves',
    'plot_predictions_vs_actual',
    'plot_time_series_predictions',
    'plot_error_distributions',
    'plot_horizon_comparison',
    'plot_feature_importance',
    'plot_surf_classification_confusion_matrix',
    'create_model_summary_dashboard',
    
    # Model Comparison
    'ModelComparison',
    'load_model_results',
    'compare_saved_results'
]