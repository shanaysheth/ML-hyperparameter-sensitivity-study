from preprocessing import check_null_values, encode_data, standard_scale
from sensitivity_analysis import perform_sensitivity_analysis
from models.decision_tree import DecisionTreeModel
from models.knn import KNNModel
from models.logistic_regression import LogisticRegressionModel
from models.svm import SVMModel
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import sys
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Use an absolute results directory inside the project root to avoid permission issues
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

def load_data(file_path):
    """Load CSV file resolving common relative locations (src/, project root, project_root/data/).
    Raises FileNotFoundError with suggestions if not found.
    """
    candidates = []
    # If absolute path provided, try it first
    if os.path.isabs(file_path):
        candidates.append(file_path)
    else:
        # try relative to src (BASE_DIR)
        candidates.append(os.path.join(BASE_DIR, file_path))
        # try relative to project root
        candidates.append(os.path.join(PROJECT_ROOT, file_path))
        # try inside project_root/data/
        candidates.append(os.path.join(PROJECT_ROOT, 'data', os.path.basename(file_path)))

    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)

    # Helpful error if not found
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    if os.path.isdir(data_dir):
        available = os.listdir(data_dir)
        raise FileNotFoundError(
            f"File '{file_path}' not found. Searched paths: {candidates}\n"
            f"Available files in data/: {available}"
        )
    else:
        raise FileNotFoundError(
            f"File '{file_path}' not found. Searched paths: {candidates}\n"
            f"Data directory '{data_dir}' does not exist."
        )

def load_hyperparameters(model_name):
    param_file = os.path.join(PROJECT_ROOT, 'hyperparameters', f'{model_name}_params.json')
    if os.path.exists(param_file):
        with open(param_file) as f:
            return json.load(f)
    return {}

def get_param_combinations(hyperparams):
    """Generate all combinations of hyperparameters"""
    if not hyperparams:
        return [{}]
    
    param_names = list(hyperparams.keys())
    param_values = list(hyperparams.values())
    
    # Convert single values to lists
    param_values = [[v] if not isinstance(v, list) else v for v in param_values]
    
    # Generate all combinations
    combinations = []
    from itertools import product
    for combo in product(*param_values):
        combinations.append(dict(zip(param_names, combo)))
    
    return combinations

def save_results_to_csv(all_results, output_path=None):
    """Save all results to CSV file"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Resolve output path (use RESULTS_DIR for relative paths)
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, 'sensitivity_metrics.csv')
    elif not os.path.isabs(output_path):
        output_path = os.path.join(RESULTS_DIR, os.path.basename(output_path))

    rows = []
    for model_name, model_results in all_results.items():
        for i, result in enumerate(model_results):
            rows.append({
                'Model': model_name,
                'Run': i + 1,
                'Hyperparameters': str(result['hyperparameters']),
                'F1_Score': result['f1_score'],
                'AUC': result['auc'],
                'Accuracy': result['accuracy']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to {output_path}")

def save_variance_results(variance_results, output_path=None):
    """Save variance summary to CSV file"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, 'variance_summary.csv')
    elif not os.path.isabs(output_path):
        output_path = os.path.join(RESULTS_DIR, os.path.basename(output_path))

    rows = []
    for model_name, metrics in variance_results.items():
        rows.append({
            'Model': model_name,
            'F1_Score_Mean': metrics['f1_score']['mean'],
            'F1_Score_Variance': metrics['f1_score']['variance'],
            'F1_Score_Std': metrics['f1_score']['std'],
            'AUC_Mean': metrics['auc']['mean'],
            'AUC_Variance': metrics['auc']['variance'],
            'AUC_Std': metrics['auc']['std'],
            'Accuracy_Mean': metrics['accuracy']['mean'],
            'Accuracy_Variance': metrics['accuracy']['variance'],
            'Accuracy_Std': metrics['accuracy']['std']
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Variance summary saved to {output_path}")

def calculate_variance_stats(scores):
    """Calculate mean, variance, and standard deviation"""
    scores_array = np.array(scores)
    return {
        'mean': np.mean(scores_array),
        'variance': np.var(scores_array),
        'std': np.std(scores_array)
    }

def main():
    datasets = {
        'diabetes': r'data\diabetes_binary.csv',
    }

    # Load hyperparameters
    hyperparams_config = {
        'decision_tree': load_hyperparameters('decision_tree'),
        'knn': load_hyperparameters('knn'),
        'logistic_regression': load_hyperparameters('logistic_regression'),
        'svm': load_hyperparameters('svm')
    }

    all_results = {
        'decision_tree': [],
        'knn': [],
        'logistic_regression': [],
        'svm': []
    }

    for dataset_name, file_path in datasets.items():
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        data = load_data(file_path)
        print(f"Dataset shape: {data.shape}")
        
        # Check for null values
        null_check = check_null_values(data)
        if len(null_check) > 0:
            print(f"Null values found:\n{null_check}")
        else:
            print("No null values found.")
        
        # Separate target and features
        y = data.iloc[:, 0]
        X = data.iloc[:, 1:]
        print(f"Target shape: {y.shape}, Features shape: {X.shape}")
        
        # Preprocess data
        print("Encoding data...")
        encoded_data = encode_data(X)
        print("Scaling data...")
        scaled_data = standard_scale(encoded_data)

        # Test each model with different hyperparameters
        for model_name in ['decision_tree', 'knn', 'logistic_regression', 'svm']:
            print(f"\n{'-'*60}")
            print(f"Testing {model_name.upper()}")
            print(f"{'-'*60}")
            
            hyperparams = hyperparams_config[model_name]
            param_combinations = get_param_combinations(hyperparams)
            
            print(f"Number of hyperparameter combinations: {len(param_combinations)}")
            
            for idx, params in enumerate(param_combinations):
                print(f"\n  Run {idx + 1}/{len(param_combinations)}: {params}")
                
                try:
                    # Initialize model with current hyperparameters
                    if model_name == 'decision_tree':
                        model = DecisionTreeModel(
                            None,
                            max_depth=params.get('max_depth', 5),
                            min_samples_split=params.get('min_samples_split', 2)
                        )
                    elif model_name == 'knn':
                        model = KNNModel(
                            None,
                            n_neighbors=params.get('n_neighbors', 5)
                        )
                    elif model_name == 'logistic_regression':
                        model = LogisticRegressionModel(
                            None,
                            max_iter=params.get('max_iter', 2000),
                            C=params.get('C', 1.0)
                        )
                    elif model_name == 'svm':
                        model = SVMModel(
                            None,
                            C=params.get('C', 0.1)
                        )
                    
                    # Perform sensitivity analysis
                    results = perform_sensitivity_analysis(
                        model.model,
                        scaled_data,
                        y,
                        params
                    )
                    
                    # Store results
                    all_results[model_name].append({
                        'hyperparameters': params,
                        'f1_score': results['f1_score'],
                        'auc': results['auc'],
                        'accuracy': results['accuracy']
                    })
                    
                    print(f"    F1 Score: {results['f1_score']:.4f}")
                    print(f"    AUC: {results['auc']:.4f}")
                    print(f"    Accuracy: {results['accuracy']:.4f}")
                    
                except Exception as e:
                    print(f"    Error: {str(e)}")
                    all_results[model_name].append({
                        'hyperparameters': params,
                        'f1_score': 0,
                        'auc': 0,
                        'accuracy': 0
                    })

    # Save detailed results
    save_results_to_csv(all_results)

    # Calculate variance statistics
    print(f"\n{'='*60}")
    print("CALCULATING VARIANCE STATISTICS")
    print(f"{'='*60}")
    
    variance_results = {}
    for model_name, results in all_results.items():
        f1_scores = [r['f1_score'] for r in results]
        auc_scores = [r['auc'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        
        variance_results[model_name] = {
            'f1_score': calculate_variance_stats(f1_scores),
            'auc': calculate_variance_stats(auc_scores),
            'accuracy': calculate_variance_stats(accuracies)
        }
        
        print(f"\n{model_name.upper()}:")
        print(f"  F1 Score   - Mean: {variance_results[model_name]['f1_score']['mean']:.4f}, "
              f"Variance: {variance_results[model_name]['f1_score']['variance']:.4f}, "
              f"Std: {variance_results[model_name]['f1_score']['std']:.4f}")
        print(f"  AUC        - Mean: {variance_results[model_name]['auc']['mean']:.4f}, "
              f"Variance: {variance_results[model_name]['auc']['variance']:.4f}, "
              f"Std: {variance_results[model_name]['auc']['std']:.4f}")
        print(f"  Accuracy   - Mean: {variance_results[model_name]['accuracy']['mean']:.4f}, "
              f"Variance: {variance_results[model_name]['accuracy']['variance']:.4f}, "
              f"Std: {variance_results[model_name]['accuracy']['std']:.4f}")

    # Save variance results
    save_variance_results(variance_results)

    # Create visualization
    plot_variance_results(variance_results)

def plot_variance_results(variance_results):
    """Create visualization of variance results"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
     
    models = list(variance_results.keys())
    metrics = ['f1_score', 'auc', 'accuracy']
     
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
     
    for idx, metric in enumerate(metrics):
         means = [variance_results[m][metric]['mean'] for m in models]
         stds = [variance_results[m][metric]['std'] for m in models]
         
         axes[idx].bar(models, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
         axes[idx].set_title(f'{metric.upper()} (Mean ± Std Dev)')
         axes[idx].set_ylabel('Score')
         axes[idx].set_ylim([0, 1])
         axes[idx].grid(axis='y', alpha=0.3)
     
    plt.tight_layout()
    viz_path = os.path.join(RESULTS_DIR, 'variance_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Variance visualization saved to {viz_path}")
    plt.show()

if __name__ == "__main__":
    main()