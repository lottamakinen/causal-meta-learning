"""
Evaluation utilities for hierarchical meta-learning models.
"""

import sys
import pandas as pd

sys.path.insert(0, 'utils')
sys.path.insert(0, 'metrics')
sys.path.insert(0, 'method')

from accuracy import task_classification_metrics, avg_task_classification_metrics, compute_curves
from hierarchical_model import hierarchical_best_model


def evaluate_best_model(best_params, trainloader, taskloader, data, args):
    """
    Evaluate the best model on test set.
    
    Args:
        best_params: Best hyperparameters from tuning
        trainloader: DataLoader for training tasks
        taskloader: DataLoader for test tasks
        data: DataModule instance
        args: Command line arguments
    """
    # Get predictions on test set
    test_task_preds = hierarchical_best_model(best_params, trainloader, taskloader, data, args)

    # Compute metrics
    task_metrics = task_classification_metrics(test_task_preds)
    avg_metrics = avg_task_classification_metrics(task_metrics)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for task_label in task_metrics:
        print(f"\nTask {task_label}:")
        for metric, value in task_metrics[task_label].items():
            print(f"  {metric}: {value:.4f}")

    print("\n" + "-" * 60)
    print("Average across all tasks:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save metrics to CSV
    task_metrics_df = pd.DataFrame(task_metrics).T
    task_metrics_df['task'] = task_metrics_df.index
    task_metrics_df.reset_index(drop=True, inplace=True)

    avg_metrics_df = pd.DataFrame([avg_metrics])
    avg_metrics_df['task'] = 'average'

    final_metrics_df = pd.concat([task_metrics_df, avg_metrics_df], ignore_index=True)
    output_file = f'{args["outprefix"]}_task_classification_metrics.csv'
    final_metrics_df.to_csv(output_file, index=None)
    print(f"\n💾 Metrics saved to: {output_file}")

    # Generate ROC/PR curves
    compute_curves(test_task_preds, args)
