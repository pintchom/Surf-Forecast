import numpy as np
import pandas as pd
import json
from pathlib import Path
import warnings
from .metrics import evaluate_model_comprehensive, statistical_significance_test
from .visualization import create_model_summary_dashboard, plot_horizon_comparison

class ModelComparison:
    """
    Class to handle comparison between multiple models.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.baseline_results = None
        
    def add_model(self, model_name, model_predictions, true_values, target_names, 
                  model_info=None):
        """
        Add a model and its predictions for comparison.
        
        Args:
            model_name: Name of the model
            model_predictions: Model predictions (n_samples, n_outputs)
            true_values: True values (n_samples, n_outputs)
            target_names: List of target names
            model_info: Additional model information (optional)
        """
        
        self.models[model_name] = {
            'predictions': model_predictions,
            'true_values': true_values,
            'target_names': target_names,
            'info': model_info or {}
        }
        
        # Evaluate model
        baseline_metrics = self.baseline_results['primary_metrics'] if self.baseline_results else None
        
        self.results[model_name] = evaluate_model_comprehensive(
            true_values, model_predictions, target_names, model_name, baseline_metrics
        )
        
    def set_baseline(self, baseline_name):
        """
        Set one of the models as the baseline for comparison.
        
        Args:
            baseline_name: Name of the model to use as baseline
        """
        
        if baseline_name not in self.results:
            raise ValueError(f"Model '{baseline_name}' not found")
            
        self.baseline_results = self.results[baseline_name]
        
        # Recalculate comparative metrics for all models
        for model_name, model_data in self.models.items():
            if model_name != baseline_name:
                self.results[model_name] = evaluate_model_comprehensive(
                    model_data['true_values'], 
                    model_data['predictions'],
                    model_data['target_names'],
                    model_name,
                    self.baseline_results['primary_metrics']
                )
    
    def compare_models(self, metrics=['rmse', 'mae', 'r2'], include_significance=True):
        """
        Compare all models across specified metrics.
        
        Args:
            metrics: List of metrics to compare
            include_significance: Whether to include statistical significance tests
            
        Returns:
            DataFrame with comparison results
        """
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {'Model': model_name}
            
            # Overall metrics
            for metric in metrics:
                if metric in results['primary_metrics']['overall']:
                    row[f'Overall_{metric.upper()}'] = results['primary_metrics']['overall'][metric]
            
            # Per-target metrics (example for key targets)
            key_targets = ['WVHT_1h', 'WVHT_6h', 'DPD_1h', 'DPD_6h']
            for target in key_targets:
                if target in results['primary_metrics']['by_target']:
                    for metric in metrics:
                        if metric in results['primary_metrics']['by_target'][target]:
                            row[f'{target}_{metric.upper()}'] = results['primary_metrics']['by_target'][target][metric]
            
            # Improvement metrics
            if 'comparative_metrics' in results:
                comp = results['comparative_metrics']['overall_improvement']
                row['Improvement_%'] = comp['improvement_pct']
                row['Meets_20%_Target'] = comp['meets_20pct_target']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add statistical significance tests
        if include_significance and self.baseline_results:
            self._add_significance_tests(comparison_df)
        
        return comparison_df
    
    def _add_significance_tests(self, comparison_df):
        """Add statistical significance test results to comparison DataFrame."""
        
        if not self.baseline_results:
            return
            
        baseline_name = self.baseline_results['model_name']
        baseline_predictions = None
        baseline_true = None
        
        # Find baseline predictions
        for model_name, model_data in self.models.items():
            if model_name == baseline_name:
                baseline_predictions = model_data['predictions']
                baseline_true = model_data['true_values']
                break
        
        if baseline_predictions is None:
            return
        
        significance_results = []
        
        for model_name, model_data in self.models.items():
            if model_name == baseline_name:
                significance_results.append(None)
                continue
                
            try:
                sig_test = statistical_significance_test(
                    baseline_true, baseline_predictions, model_data['predictions']
                )
                significance_results.append(sig_test['is_significant'])
            except Exception as e:
                warnings.warn(f"Could not perform significance test for {model_name}: {e}")
                significance_results.append(None)
        
        comparison_df['Statistically_Significant'] = significance_results
    
    def rank_models(self, primary_metric='rmse', ascending=True):
        """
        Rank models by a primary metric.
        
        Args:
            primary_metric: Metric to rank by
            ascending: Whether lower values are better
            
        Returns:
            DataFrame with ranked models
        """
        
        ranking_data = []
        
        for model_name, results in self.results.items():
            metric_value = results['primary_metrics']['overall'].get(primary_metric)
            if metric_value is not None:
                ranking_data.append({
                    'Rank': 0,  # Will be filled
                    'Model': model_name,
                    f'{primary_metric.upper()}': metric_value,
                    'Info': self.models[model_name]['info']
                })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values(f'{primary_metric.upper()}', ascending=ascending)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df[['Rank', 'Model', f'{primary_metric.upper()}', 'Info']]
    
    def get_best_model(self, metric='rmse', target=None):
        """
        Get the best performing model for a specific metric and target.
        
        Args:
            metric: Metric to optimize
            target: Specific target (if None, uses overall metric)
            
        Returns:
            Tuple of (best_model_name, best_metric_value)
        """
        
        best_model = None
        best_value = float('inf') if metric in ['rmse', 'mae'] else float('-inf')
        is_lower_better = metric in ['rmse', 'mae']
        
        for model_name, results in self.results.items():
            if target:
                if target in results['primary_metrics']['by_target']:
                    metric_value = results['primary_metrics']['by_target'][target][metric]
                else:
                    continue
            else:
                metric_value = results['primary_metrics']['overall'][metric]
            
            if is_lower_better:
                if metric_value < best_value:
                    best_value = metric_value
                    best_model = model_name
            else:
                if metric_value > best_value:
                    best_value = metric_value
                    best_model = model_name
        
        return best_model, best_value
    
    def create_comparison_report(self, save_path=None):
        """
        Generate a comprehensive comparison report.
        
        Args:
            save_path: Path to save the report (optional)
            
        Returns:
            Dictionary with complete comparison results
        """
        
        report = {
            'summary': {
                'n_models': len(self.models),
                'models': list(self.models.keys()),
                'baseline': self.baseline_results['model_name'] if self.baseline_results else None
            },
            'model_rankings': {},
            'best_performers': {},
            'comparison_table': None,
            'prd_compliance': {}
        }
        
        # Model rankings
        for metric in ['rmse', 'mae', 'r2']:
            ascending = metric != 'r2'
            ranking = self.rank_models(metric, ascending)
            report['model_rankings'][metric] = ranking.to_dict('records')
        
        # Best performers by target
        key_targets = ['WVHT_1h', 'WVHT_3h', 'WVHT_6h', 'DPD_1h', 'DPD_3h', 'DPD_6h']
        for target in key_targets:
            best_model, best_value = self.get_best_model('rmse', target)
            if best_model:
                report['best_performers'][target] = {
                    'model': best_model,
                    'rmse': best_value
                }
        
        # Comparison table
        comparison_df = self.compare_models()
        report['comparison_table'] = comparison_df.to_dict('records')
        
        # PRD compliance check
        from .metrics import check_prd_targets
        for model_name, results in self.results.items():
            model_data = self.models[model_name]
            compliance = check_prd_targets(results, model_data['target_names'])
            report['prd_compliance'][model_name] = compliance
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def print_comparison_summary(self):
        """Print a formatted summary of model comparison."""
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        print(f"Number of models compared: {len(self.models)}")
        if self.baseline_results:
            print(f"Baseline model: {self.baseline_results['model_name']}")
        
        # Overall performance ranking
        ranking = self.rank_models('rmse', ascending=True)
        print(f"\nOverall Performance Ranking (by RMSE):")
        print("-" * 50)
        for _, row in ranking.iterrows():
            print(f"{row['Rank']:2d}. {row['Model']:<20} RMSE: {row['RMSE']:.4f}")
        
        # Best performers by target
        print(f"\nBest Performers by Target:")
        print("-" * 50)
        key_targets = ['WVHT_1h', 'WVHT_6h', 'DPD_1h', 'DPD_6h']
        for target in key_targets:
            best_model, best_value = self.get_best_model('rmse', target)
            if best_model:
                print(f"{target:<10}: {best_model:<20} (RMSE: {best_value:.4f})")
        
        # PRD compliance
        print(f"\nPRD Target Compliance:")
        print("-" * 50)
        
        for model_name, results in self.results.items():
            model_data = self.models[model_name]
            
            # Check key targets
            wvht_1h_pass = False
            wvht_6h_pass = False
            improvement_pass = False
            
            # WVHT_1h target (≤0.30m)
            if 'WVHT_1h' in results['primary_metrics']['by_target']:
                rmse_1h = results['primary_metrics']['by_target']['WVHT_1h']['rmse']
                wvht_1h_pass = rmse_1h <= 0.30
            
            # WVHT_6h target (≤0.50m)  
            if 'WVHT_6h' in results['primary_metrics']['by_target']:
                rmse_6h = results['primary_metrics']['by_target']['WVHT_6h']['rmse']
                wvht_6h_pass = rmse_6h <= 0.50
            
            # Improvement target (≥20%)
            if 'comparative_metrics' in results:
                improvement_pct = results['comparative_metrics']['overall_improvement']['improvement_pct']
                improvement_pass = improvement_pct >= 20.0
            
            status_1h = "✅" if wvht_1h_pass else "❌"
            status_6h = "✅" if wvht_6h_pass else "❌" 
            status_imp = "✅" if improvement_pass else "❌"
            
            print(f"{model_name:<20}: 1h {status_1h} | 6h {status_6h} | Improvement {status_imp}")
    
    def visualize_comparison(self, save_dir=None, show_plots=True):
        """
        Create visualization plots for model comparison.
        
        Args:
            save_dir: Directory to save plots (optional)
            show_plots: Whether to display plots
            
        Returns:
            Dictionary of matplotlib figure objects
        """
        
        figures = {}
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Summary dashboard
        dashboard_path = save_dir / "model_comparison_dashboard.png" if save_dir else None
        fig_dashboard = create_model_summary_dashboard(
            self.results, save_path=dashboard_path, show_plot=show_plots
        )
        figures['dashboard'] = fig_dashboard
        
        # Horizon comparison plots
        for metric in ['rmse', 'mae', 'r2']:
            horizon_path = save_dir / f"horizon_comparison_{metric}.png" if save_dir else None
            fig_horizon = plot_horizon_comparison(
                self.results, metric=metric, save_path=horizon_path, show_plot=show_plots
            )
            figures[f'horizon_{metric}'] = fig_horizon
        
        return figures

def load_model_results(results_path):
    """
    Load model results from saved JSON file.
    
    Args:
        results_path: Path to JSON results file
        
    Returns:
        Dictionary with model results
    """
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results

def compare_saved_results(results_paths_dict, baseline_model=None):
    """
    Compare models from saved results files.
    
    Args:
        results_paths_dict: Dict mapping model names to results file paths
        baseline_model: Name of baseline model (optional)
        
    Returns:
        ModelComparison object with loaded results
    """
    
    comparison = ModelComparison()
    
    # Note: This function loads results only, not actual predictions
    # For full comparison with predictions, use add_model() method
    
    for model_name, results_path in results_paths_dict.items():
        results = load_model_results(results_path)
        comparison.results[model_name] = results
    
    if baseline_model and baseline_model in comparison.results:
        comparison.baseline_results = comparison.results[baseline_model]
    
    return comparison