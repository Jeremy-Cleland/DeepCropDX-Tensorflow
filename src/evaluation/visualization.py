import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from pathlib import Path


def plot_training_history(history, save_path=None):
    """Plot training and validation metrics over epochs"""
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    if "accuracy" in history.history:
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        if "val_accuracy" in history.history:
            plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
    elif "acc" in history.history:  # For compatibility with older TF versions
        plt.plot(history.history["acc"], label="Training Accuracy")
        if "val_acc" in history.history:
            plt.plot(history.history["val_acc"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true, y_pred, class_names=None, save_path=None, figsize=(10, 8), normalize=False
):
    """
    Plot confusion matrix

    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        class_names: List or dictionary of class names
        save_path: Path to save the plot
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix if requested
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    # Process class names
    if class_names is None:
        # Use indices as class names
        labels = [str(i) for i in range(cm.shape[0])]
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, convert it to a list of names
        labels = [class_names[i] for i in range(cm.shape[0])]
    else:
        labels = class_names

    # Truncate long class names
    max_length = 20
    labels = [
        label[:max_length] + "..." if len(label) > max_length else label
        for label in labels
    ]

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Add counts to plot title
    plt.figtext(0.5, 0.01, f"Total samples: {len(y_true)}", ha="center")

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true, y_pred_prob, class_names=None, save_path=None, figsize=(10, 8)
):
    """
    Plot ROC curve for multi-class classification

    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred_prob: Predicted probabilities
        class_names: List or dictionary of class names
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
        y_true_onehot = y_true
    else:
        y_true_indices = y_true
        # Convert to one-hot encoding
        n_classes = y_pred_prob.shape[1]
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1

    # Calculate ROC curve and ROC area for each class
    n_classes = y_pred_prob.shape[1]

    # Process class names
    if class_names is None:
        # Use indices as class names
        labels = [str(i) for i in range(n_classes)]
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, convert it to a list of names
        labels = [class_names[i] for i in range(n_classes)]
    else:
        labels = class_names

    plt.figure(figsize=figsize)

    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{labels[i]} (AUC = {roc_auc:.2f})")

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(
    y_true, y_pred_prob, class_names=None, save_path=None, figsize=(10, 8)
):
    """
    Plot precision-recall curve for multi-class classification

    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred_prob: Predicted probabilities
        class_names: List or dictionary of class names
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
        y_true_onehot = y_true
    else:
        y_true_indices = y_true
        # Convert to one-hot encoding
        n_classes = y_pred_prob.shape[1]
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1

    # Calculate Precision-Recall curve for each class
    n_classes = y_pred_prob.shape[1]

    # Process class names
    if class_names is None:
        # Use indices as class names
        labels = [str(i) for i in range(n_classes)]
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, convert it to a list of names
        labels = [class_names[i] for i in range(n_classes)]
    else:
        labels = class_names

    plt.figure(figsize=figsize)

    # Plot Precision-Recall curve for each class
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(
            y_true_onehot[:, i], y_pred_prob[:, i]
        )
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"{labels[i]} (AUC = {pr_auc:.2f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_class_distribution(y_true, class_names=None, save_path=None, figsize=(12, 6)):
    """
    Plot class distribution

    Args:
        y_true: True labels (one-hot encoded or class indices)
        class_names: List or dictionary of class names
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    # Count occurrences of each class
    unique_classes, counts = np.unique(y_true, return_counts=True)

    # Process class names
    if class_names is None:
        # Use indices as class names
        labels = [str(i) for i in unique_classes]
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, convert it to a list of names
        labels = [class_names[i] for i in unique_classes]
    else:
        labels = [class_names[i] for i in unique_classes]

    # Sort by frequency
    idx = np.argsort(counts)[::-1]
    counts = counts[idx]
    labels = [labels[i] for i in idx]

    # Plot
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(counts)), counts, align="center")
    plt.xticks(range(len(counts)), labels, rotation=45, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")

    # Add counts on top of bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 5,
            str(counts[i]),
            ha="center",
        )

    plt.tight_layout()

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_misclassified_examples(
    x_test,
    y_true,
    y_pred,
    class_names=None,
    num_examples=9,
    save_path=None,
    figsize=(15, 15),
):
    """
    Plot misclassified examples

    Args:
        x_test: Test images
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        class_names: List or dictionary of class names
        num_examples: Number of examples to plot
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Find misclassified examples
    misclassified = np.where(y_true != y_pred)[0]

    if len(misclassified) == 0:
        print("No misclassified examples found!")
        return

    # Process class names
    if class_names is None:
        # Use indices as class names
        get_class_name = lambda idx: str(idx)
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, use it directly
        get_class_name = lambda idx: class_names[idx]
    else:
        # If class_names is a list, use indices
        get_class_name = lambda idx: class_names[idx]

    # Select random misclassified examples
    indices = np.random.choice(
        misclassified, size=min(num_examples, len(misclassified)), replace=False
    )

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(len(indices))))

    # Plot
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        # Get the image
        img = x_test[idx]

        # For greyscale images
        if img.shape[-1] == 1:
            img = img.reshape(img.shape[:-1])

        # Normalize image if needed
        if img.max() > 1.0:
            img = img / 255.0

        # Plot the image
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {get_class_name(y_true[idx])}\nPred: {get_class_name(y_pred[idx])}"
        )
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
