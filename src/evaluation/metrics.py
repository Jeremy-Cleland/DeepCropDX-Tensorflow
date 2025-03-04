import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import json
from pathlib import Path
from tqdm.auto import tqdm


def calculate_metrics(y_true, y_pred, y_pred_prob=None, class_names=None):
    """
    Calculate evaluation metrics for classification

    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        y_pred_prob: Predicted probabilities (optional)
        class_names: List of class names (optional)

    Returns:
        Dictionary with calculated metrics
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true

    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
    else:
        y_pred_indices = y_pred

    # Calculate basic metrics
    metrics = {
        "accuracy": accuracy_score(y_true_indices, y_pred_indices),
        "precision_macro": precision_score(
            y_true_indices, y_pred_indices, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true_indices, y_pred_indices, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true_indices, y_pred_indices, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true_indices, y_pred_indices, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true_indices, y_pred_indices, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true_indices, y_pred_indices, average="weighted", zero_division=0
        ),
    }

    # Calculate AUC if probabilities are provided
    if y_pred_prob is not None:
        # For multi-class classification
        if y_pred_prob.shape[1] > 2:
            try:
                # Convert y_true to one-hot encoding if it's not already
                if y_true.ndim == 1 or y_true.shape[1] == 1:
                    n_classes = y_pred_prob.shape[1]
                    y_true_onehot = np.zeros((len(y_true_indices), n_classes))
                    y_true_onehot[np.arange(len(y_true_indices)), y_true_indices] = 1
                else:
                    y_true_onehot = y_true

                metrics["roc_auc_macro"] = roc_auc_score(
                    y_true_onehot, y_pred_prob, average="macro", multi_class="ovr"
                )
                metrics["roc_auc_weighted"] = roc_auc_score(
                    y_true_onehot, y_pred_prob, average="weighted", multi_class="ovr"
                )
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC: {e}")
        # For binary classification
        elif y_pred_prob.shape[1] == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true_indices, y_pred_prob[:, 1])
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC: {e}")

    # Add per-class metrics if class names are provided
    if class_names is not None:
        # Get report as dictionary
        report = classification_report(
            y_true_indices,
            y_pred_indices,
            output_dict=True,
            target_names=(
                class_names
                if isinstance(class_names, list)
                else list(class_names.values())
            ),
        )

        # Add per-class metrics to the main metrics dictionary
        metrics["per_class"] = {}
        for class_name in report:
            if class_name not in ["accuracy", "macro avg", "weighted avg"]:
                metrics["per_class"][class_name] = {
                    "precision": report[class_name]["precision"],
                    "recall": report[class_name]["recall"],
                    "f1-score": report[class_name]["f1-score"],
                    "support": report[class_name]["support"],
                }

    # Calculate confusion matrix but don't include in returned metrics
    # (it's not JSON serializable)
    cm = confusion_matrix(y_true_indices, y_pred_indices)

    return metrics


def evaluate_model(
    model, test_data, class_names=None, metrics_path=None, use_tqdm=True
):
    """
    Evaluate a model on test data with progress tracking

    Args:
        model: TensorFlow model to evaluate
        test_data: Test dataset
        class_names: List of class names (optional)
        metrics_path: Path to save metrics (optional)
        use_tqdm: Whether to use tqdm progress bar

    Returns:
        Dictionary with evaluation metrics
    """
    # Create predictions with progress bar
    print("Generating predictions...")

    if use_tqdm:
        # Get number of batches
        n_batches = len(test_data)

        # Initialize lists for predictions and true labels
        all_y_pred = []
        all_y_true = []

        # Use tqdm for progress tracking
        for batch_idx, (x, y) in enumerate(
            tqdm(test_data, desc="Predicting", total=n_batches)
        ):
            # Get predictions for this batch
            y_pred = model.predict(x, verbose=0)
            all_y_pred.append(y_pred)
            all_y_true.append(y)

        # Concatenate all batches
        y_pred_prob = np.vstack(all_y_pred)
        y_true = np.vstack(all_y_true)
    else:
        # Standard evaluation without progress tracking
        y_pred_prob = model.predict(test_data)
        y_true = np.concatenate([y for x, y in test_data], axis=0)

    # Convert probabilities to class predictions
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_prob, class_names)

    # Add standard evaluation metrics
    if hasattr(model, "evaluate"):
        results = model.evaluate(test_data, verbose=1)
        for i, metric_name in enumerate(model.metrics_names):
            metrics[metric_name] = results[i]

    # Save metrics if path is provided
    if metrics_path:
        # Convert path to Path object if it's a string
        if isinstance(metrics_path, str):
            metrics_path = Path(metrics_path)

        # Create parent directories if they don't exist
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python native types for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if key == "per_class":
                metrics_json[key] = {}
                for class_name, class_metrics in value.items():
                    metrics_json[key][class_name] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in class_metrics.items()
                    }
            else:
                metrics_json[key] = (
                    float(value)
                    if isinstance(value, (np.float32, np.float64))
                    else value
                )

        # Save to file
        with open(metrics_path, "w") as f:
            json.dump(metrics_json, f, indent=4)

    return metrics
