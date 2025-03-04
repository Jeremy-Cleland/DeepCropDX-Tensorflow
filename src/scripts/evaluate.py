#!/usr/bin/env python3
"""
Evaluate a trained model on a dataset
"""

import os
import argparse

import tensorflow as tf
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm


from src.config.config import get_paths
from src.config.config_loader import ConfigLoader
from src.preprocessing.data_loader import DataLoader
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_class_distribution,
    plot_misclassified_examples,
)
from src.utils.report_generator import ReportGenerator


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.h5 file)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the dataset directory for evaluation",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (overrides config)",
    )
    parser.add_argument(
        "--visualize_misclassified",
        action="store_true",
        help="Generate visualizations of misclassified samples",
    )
    args = parser.parse_args()

    # Get project paths
    paths = get_paths()

    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.get_config()

    # Override batch size if provided
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # If model path is inside trials directory, use its evaluation folder
        model_path = Path(args.model_path)
        model_dir = model_path.parent

        if "trials" in str(model_dir):
            output_dir = model_dir / "evaluation"
        else:
            # Default to a timestamp-based directory
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = paths.trials_dir / "evaluations" / f"eval_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation results will be saved to: {output_dir}")

    # Load model
    try:
        print(f"Loading model from {args.model_path}...")
        model = tf.keras.models.load_model(args.model_path)
        print(f"Model loaded successfully.")

        # Print model summary
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load data
    data_loader = DataLoader(config)
    _, _, test_data, class_names = data_loader.load_data(args.data_dir)

    if test_data is None:
        print("Error: No test data available for evaluation")
        return

    print(f"Evaluating model on {len(class_names)} classes")

    # Evaluate model
    print("Evaluating model...")
    metrics_path = output_dir / "metrics.json"
    metrics = evaluate_model(
        model,
        test_data,
        class_names=class_names,
        metrics_path=metrics_path,
        use_tqdm=True,
    )

    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  Loss: {metrics.get('loss', 0):.4f}")

    if "f1_macro" in metrics:
        print(f"  F1 Score (Macro): {metrics.get('f1_macro', 0):.4f}")

    if "precision_macro" in metrics:
        print(f"  Precision (Macro): {metrics.get('precision_macro', 0):.4f}")

    if "recall_macro" in metrics:
        print(f"  Recall (Macro): {metrics.get('recall_macro', 0):.4f}")

    # Generate plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get predictions for visualization
    print("Generating predictions for visualization...")
    all_x = []
    all_y_true = []
    all_y_pred_prob = []

    # Use tqdm for progress tracking
    for batch_idx, (x, y) in enumerate(tqdm(test_data, desc="Predicting")):
        # Get predictions for this batch
        y_pred = model.predict(x, verbose=0)
        all_y_pred_prob.append(y_pred)
        all_y_true.append(y)

        # For misclassified visualization, save the images too
        if args.visualize_misclassified:
            all_x.append(x)

    # Concatenate all batches
    y_pred_prob = np.vstack(all_y_pred_prob)
    y_true = np.vstack(all_y_true)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # For misclassified visualization, concatenate the images
    if args.visualize_misclassified:
        x_test = np.vstack(all_x)

    # Plot confusion matrix
    print("Generating confusion matrix...")
    cm_path = plots_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Plot normalized confusion matrix
    cm_norm_path = plots_dir / "confusion_matrix_normalized.png"
    plot_confusion_matrix(
        y_true, y_pred, class_names, save_path=cm_norm_path, normalize=True
    )
    print(f"Normalized confusion matrix saved to {cm_norm_path}")

    # Plot ROC curve
    print("Generating ROC curve...")
    roc_path = plots_dir / "roc_curve.png"
    plot_roc_curve(y_true, y_pred_prob, class_names, save_path=roc_path)
    print(f"ROC curve saved to {roc_path}")

    # Plot precision-recall curve
    print("Generating precision-recall curve...")
    pr_path = plots_dir / "precision_recall_curve.png"
    plot_precision_recall_curve(y_true, y_pred_prob, class_names, save_path=pr_path)
    print(f"Precision-recall curve saved to {pr_path}")

    # Plot class distribution
    print("Generating class distribution...")
    dist_path = plots_dir / "class_distribution.png"
    plot_class_distribution(y_true, class_names, save_path=dist_path)
    print(f"Class distribution saved to {dist_path}")

    # Plot misclassified examples if requested
    if args.visualize_misclassified:
        print("Generating misclassified examples visualization...")
        misclass_path = plots_dir / "misclassified_examples.png"
        plot_misclassified_examples(
            x_test,
            y_true,
            y_pred,
            class_names,
            num_examples=25,
            save_path=misclass_path,
        )
        print(f"Misclassified examples saved to {misclass_path}")

    # Generate HTML report
    print("Generating evaluation report...")
    report_generator = ReportGenerator(config)
    report_context = {
        "model_path": args.model_path,
        "metrics": metrics,
        "plots": {
            "confusion_matrix": str(cm_path),
            "confusion_matrix_normalized": str(cm_norm_path),
            "roc_curve": str(roc_path),
            "precision_recall_curve": str(pr_path),
            "class_distribution": str(dist_path),
        },
    }

    if args.visualize_misclassified:
        report_context["plots"]["misclassified_examples"] = str(misclass_path)

    report_path = output_dir / "evaluation_report.html"

    # Create a simple report template
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .metrics {{ margin: 20px 0; }}
            .metrics table {{ border-collapse: collapse; width: 100%; }}
            .metrics th, .metrics td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .metrics th {{ background-color: #f2f2f2; }}
            .plots {{ display: flex; flex-wrap: wrap; justify-content: center; }}
            .plot {{ margin: 10px; text-align: center; }}
            .plot img {{ max-width: 800px; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>
        <p>Model: {os.path.basename(args.model_path)}</p>
        
        <h2>Metrics</h2>
        <div class="metrics">
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Accuracy</td><td>{metrics.get('accuracy', 0):.4f}</td></tr>
                <tr><td>Loss</td><td>{metrics.get('loss', 0):.4f}</td></tr>
                <tr><td>F1 Score (Macro)</td><td>{metrics.get('f1_macro', 0):.4f}</td></tr>
                <tr><td>Precision (Macro)</td><td>{metrics.get('precision_macro', 0):.4f}</td></tr>
                <tr><td>Recall (Macro)</td><td>{metrics.get('recall_macro', 0):.4f}</td></tr>
            </table>
        </div>
        
        <h2>Visualizations</h2>
        <div class="plots">
            <div class="plot">
                <h3>Confusion Matrix</h3>
                <img src="{os.path.relpath(cm_path, output_dir)}" alt="Confusion Matrix">
            </div>
            
            <div class="plot">
                <h3>Normalized Confusion Matrix</h3>
                <img src="{os.path.relpath(cm_norm_path, output_dir)}" alt="Normalized Confusion Matrix">
            </div>
            
            <div class="plot">
                <h3>ROC Curve</h3>
                <img src="{os.path.relpath(roc_path, output_dir)}" alt="ROC Curve">
            </div>
            
            <div class="plot">
                <h3>Precision-Recall Curve</h3>
                <img src="{os.path.relpath(pr_path, output_dir)}" alt="Precision-Recall Curve">
            </div>
            
            <div class="plot">
                <h3>Class Distribution</h3>
                <img src="{os.path.relpath(dist_path, output_dir)}" alt="Class Distribution">
            </div>
            
            {f'<div class="plot"><h3>Misclassified Examples</h3><img src="{os.path.relpath(misclass_path, output_dir)}" alt="Misclassified Examples"></div>' if args.visualize_misclassified else ''}
        </div>
    </body>
    </html>
    """

    with open(report_path, "w") as f:
        f.write(report_html)

    print(f"Evaluation completed. Report generated at {report_path}")


if __name__ == "__main__":
    main()
