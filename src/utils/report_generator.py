import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from jinja2 import Template


from src.config.config import get_paths


class ReportGenerator:
    def __init__(self, config=None):
        self.config = config or {}
        self.paths = get_paths()

    def generate_model_report(
        self, model_name, run_dir, metrics, history=None, class_names=None
    ):
        """Generate an HTML report for a model run"""
        # Convert run_dir to Path if it's a string
        if isinstance(run_dir, str):
            run_dir = Path(run_dir)

        # Load metrics
        if isinstance(metrics, str) and os.path.exists(metrics):
            with open(metrics, "r") as f:
                metrics = json.load(f)

        # Load history if path provided
        if isinstance(history, str) and os.path.exists(history):
            history_df = pd.read_csv(history)
            history_dict = {col: history_df[col].tolist() for col in history_df.columns}
            history = type("obj", (object,), {"history": history_dict})

        # Create plots directory if needed
        plots_dir = run_dir / "training" / "plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Generate plots if history is available
        if history is not None and hasattr(history, "history"):
            # Import here to avoid circular imports
            from src.evaluation.visualization import plot_training_history

            history_plot_path = plots_dir / "training_history.png"
            plot_training_history(history, save_path=history_plot_path)

            # Create additional plots if enabled in config
            if self.config.get("reporting", {}).get("generate_plots", True):
                # Generate learning rate plot if available
                if "lr" in history.history:
                    self._plot_learning_rate(history, plots_dir / "learning_rate.png")

                # Generate loss and metrics comparison plots
                metrics_to_plot = [
                    k
                    for k in history.history.keys()
                    if not k.startswith("lr") and not k.startswith("val_")
                ]

                for metric in metrics_to_plot:
                    val_metric = f"val_{metric}"
                    if val_metric in history.history:
                        self._plot_metric_comparison(
                            history,
                            metric,
                            val_metric,
                            plots_dir / f"{metric}_comparison.png",
                        )

        # Create report context
        context = {
            "model_name": model_name,
            "run_dir": str(run_dir),
            "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "has_history": history is not None and hasattr(history, "history"),
            "class_names": class_names,
            "project_name": self.config.get("project", {}).get(
                "name", "Plant Disease Detection"
            ),
            "project_version": self.config.get("project", {}).get("version", "1.0.0"),
        }

        # Generate HTML report using template
        html = self._render_report_template(context)

        # Save report to file
        report_path = run_dir / "report.html"
        with open(report_path, "w") as f:
            f.write(html)

        return report_path

    def _plot_learning_rate(self, history, save_path):
        """Plot learning rate over epochs"""
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["lr"])
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def _plot_metric_comparison(self, history, train_metric, val_metric, save_path):
        """Plot comparison between training and validation metrics"""
        plt.figure(figsize=(10, 5))
        plt.plot(history.history[train_metric], label=f"Training {train_metric}")
        plt.plot(history.history[val_metric], label=f"Validation {val_metric}")
        plt.title(f"{train_metric} Comparison")
        plt.xlabel("Epoch")
        plt.ylabel(train_metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def _render_report_template(self, context):
        """Render HTML report using Jinja2 template"""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ model_name }} Training Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                    background-color: #f8f9fa;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .card {
                    background: #fff;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 20px;
                    margin-bottom: 20px;
                }
                .header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .header h1 {
                    color: white;
                    margin: 0;
                }
                .header p {
                    margin: 5px 0 0 0;
                    opacity: 0.8;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f8f9fa;
                }
                .metric-value {
                    font-weight: bold;
                }
                .good-metric {
                    color: #28a745;
                }
                .average-metric {
                    color: #fd7e14;
                }
                .poor-metric {
                    color: #dc3545;
                }
                .plot-container {
                    text-align: center;
                    margin: 20px 0;
                }
                .plot-container img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .plot-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .footer {
                    margin-top: 30px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 0.9em;
                    padding: 20px;
                    border-top: 1px solid #eee;
                }
                .summary {
                    font-size: 1.2em;
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f1f8ff;
                    border-left: 4px solid #4285f4;
                    border-radius: 3px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{{ model_name }} Training Report</h1>
                    <p>{{ project_name }} v{{ project_version }} | Generated on {{ generation_time }}</p>
                </div>
                
                <div class="card">
                    <h2>Model Information</h2>
                    <table>
                        <tr>
                            <th>Model Name</th>
                            <td>{{ model_name }}</td>
                        </tr>
                        <tr>
                            <th>Run Directory</th>
                            <td>{{ run_dir }}</td>
                        </tr>
                        {% if "training_time" in metrics %}
                        <tr>
                            <th>Training Time</th>
                            <td>{{ "%.2f"|format(metrics["training_time"]) }} seconds ({{ "%.2f"|format(metrics["training_time"]/60) }} minutes)</td>
                        </tr>
                        {% endif %}
                        {% if class_names %}
                        <tr>
                            <th>Classes</th>
                            <td>{{ class_names|length }} classes
                                {% if class_names|length <= 20 %}
                                    <br><small>{{ class_names|join(', ') }}</small>
                                {% endif %}
                            </td>
                        </tr>
                        {% endif %}
                    </table>
                </div>
                
                {% if "test_accuracy" in metrics or "val_accuracy" in metrics %}
                <div class="summary">
                    Model Performance Summary: 
                    {% if "test_accuracy" in metrics %}
                        Test Accuracy: <span class="metric-value 
                            {% if metrics["test_accuracy"] > 0.9 %}good-metric
                            {% elif metrics["test_accuracy"] > 0.8 %}average-metric
                            {% else %}poor-metric{% endif %}">
                            {{ "%.2f"|format(metrics["test_accuracy"] * 100) }}%
                        </span>
                    {% elif "val_accuracy" in metrics %}
                        Validation Accuracy: <span class="metric-value 
                            {% if metrics["val_accuracy"] > 0.9 %}good-metric
                            {% elif metrics["val_accuracy"] > 0.8 %}average-metric
                            {% else %}poor-metric{% endif %}">
                            {{ "%.2f"|format(metrics["val_accuracy"] * 100) }}%
                        </span>
                    {% endif %}
                </div>
                {% endif %}
                
                <div class="card">
                    <h2>Performance Metrics</h2>
                    <table>
                        {% for key, value in metrics.items() %}
                        <tr>
                            <th>{{ key }}</th>
                            <td class="metric-value">
                                {% if value is number %}
                                    {{ "%.4f"|format(value) }}
                                    {% if "accuracy" in key or "precision" in key or "recall" in key or "f1" in key or "auc" in key %}
                                        ({{ "%.2f"|format(value * 100) }}%)
                                    {% endif %}
                                {% else %}
                                    {{ value }}
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                {% if has_history %}
                <div class="card">
                    <h2>Training Visualization</h2>
                    <div class="plot-container">
                        <img src="training/plots/training_history.png" alt="Training History">
                    </div>
                    
                    <h3>Additional Plots</h3>
                    <div class="plot-grid">
                        {% if metrics.get("loss") %}
                        <div class="plot-container">
                            <img src="training/plots/loss_comparison.png" alt="Loss Comparison">
                        </div>
                        {% endif %}
                        
                        {% if metrics.get("accuracy") %}
                        <div class="plot-container">
                            <img src="training/plots/accuracy_comparison.png" alt="Accuracy Comparison">
                        </div>
                        {% endif %}
                        
                        {% if metrics.get("lr") %}
                        <div class="plot-container">
                            <img src="training/plots/learning_rate.png" alt="Learning Rate">
                        </div>
                        {% endif %}
                    </div>
                </div>
                {% endif %}
                
                <div class="footer">
                    <p>{{ project_name }} | Deep Learning for Plant Disease Detection</p>
                </div>
            </div>
        </body>
        </html>
        """

        template = Template(template_str)
        return template.render(**context)

    def generate_comparison_report(self, models_data, output_path=None):
        """Generate a comparison report for multiple models

        Args:
            models_data: List of dictionaries with model results
            output_path: Path to save the report
        """
        # Convert string metrics to numeric
        for model in models_data:
            for key, value in model["metrics"].items():
                if isinstance(value, str):
                    try:
                        if value.lower() == "n/a":
                            model["metrics"][key] = 0.0
                        else:
                            model["metrics"][key] = float(value)
                    except (ValueError, TypeError):
                        model["metrics"][key] = 0.0

        if output_path is None:
            output_path = self.paths.trials_dir / "model_comparison.html"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create comparison plots
        plots_dir = Path(os.path.dirname(output_path) / "comparison_plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Generate accuracy comparison plot
        accuracy_plot_path = Path(plots_dir / "accuracy_comparison.png")
        self._plot_model_comparison(
            models_data, "test_accuracy", "Test Accuracy", accuracy_plot_path
        )

        # Generate other comparison plots if data is available
        metrics_to_compare = [
            ("test_loss", "Test Loss"),
            ("precision_macro", "Precision (Macro)"),
            ("recall_macro", "Recall (Macro)"),
            ("f1_macro", "F1 Score (Macro)"),
            ("training_time", "Training Time (s)"),
        ]

        plot_paths = {"accuracy": accuracy_plot_path}

        for metric_key, metric_name in metrics_to_compare:
            if any(metric_key in model["metrics"] for model in models_data):
                plot_path = Path(plots_dir / f"{metric_key}_comparison.png")
                self._plot_model_comparison(
                    models_data, metric_key, metric_name, plot_path
                )
                plot_paths[metric_key] = plot_path

        # Create comparison context
        context = {
            "models_data": models_data,
            "plot_paths": plot_paths,
            "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "project_name": self.config.get("project", {}).get(
                "name", "Plant Disease Detection"
            ),
            "project_version": self.config.get("project", {}).get("version", "1.0.0"),
        }

        # Generate HTML report
        html = self._render_comparison_template(context)

        # Save report
        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    def _plot_model_comparison(self, models_data, metric_key, metric_name, save_path):
        """Create a bar chart comparing models based on a metric"""
        # Extract data
        model_names = []
        metric_values = []

        for model in models_data:
            model_names.append(model["name"])
            if metric_key in model["metrics"]:
                metric_values.append(model["metrics"][metric_key])
            else:
                metric_values.append(0)

        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, metric_values)

        # Add value annotations
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            if (
                "accuracy" in metric_key
                or "precision" in metric_key
                or "recall" in metric_key
                or "f1" in metric_key
            ):
                text = f"{value:.2%}"
            else:
                text = f"{value:.2f}"
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                text,
                ha="center",
                va="bottom",
                rotation=0,
            )

        plt.title(f"Model Comparison - {metric_name}")
        plt.xlabel("Model")
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save the plot
        plt.savefig(save_path)
        plt.close()

    def _render_comparison_template(self, context):
        """
        Prepare and render the comparison template.

        Args:
            context: The template context data

        Returns:
            str: Rendered HTML content
        """
        # Prepare models_data to ensure metrics are numeric
        for model in context["models_data"]:
            for key, value in model["metrics"].items():
                # Convert string metrics to numbers
                if isinstance(value, str):
                    try:
                        if value.lower() == "n/a":
                            model["metrics"][key] = 0.0
                        else:
                            model["metrics"][key] = float(value)
                    except (ValueError, TypeError):
                        # If conversion fails, set to 0
                        model["metrics"][key] = 0.0
                elif value is None:
                    model["metrics"][key] = 0.0

        # Continue with template rendering...
        # Existing rendering code...
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                /* CSS styles from the model report template, plus comparison-specific styles */
                /* ... add styles as in the previous template ... */
                .comparison-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 30px;
                }
                .comparison-table th, .comparison-table td {
                    padding: 12px 15px;
                    text-align: center;
                    border: 1px solid #ddd;
                }
                .comparison-table th {
                    background-color: #f8f9fa;
                }
                .comparison-table tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
                .best-value {
                    font-weight: bold;
                    color: #28a745;
                }
                .second-best {
                    color: #17a2b8;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Model Comparison Report</h1>
                    <p>{{ project_name }} v{{ project_version }} | Generated on {{ generation_time }}</p>
                </div>
                
                <div class="card">
                    <h2>Models Compared</h2>
                    <p>Comparison of {{ models_data|length }} models</p>
                    
                    <div class="comparison-table-container">
                        <table class="comparison-table">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Test Accuracy</th>
                                    <th>Test Loss</th>
                                    <th>F1 Score</th>
                                    <th>Training Time</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for model in models_data %}
                                <tr>
                                    <td>{{ model.name }}</td>
                                    <td>{{ "%.2f"|format(model.metrics.get("test_accuracy", 0) * 100) }}%</td>
                                    <td>{{ "%.4f"|format(model.metrics.get("test_loss", 0)) }}</td>
                                    <td>{{ "%.2f"|format(model.metrics.get("f1_macro", 0) * 100) }}%</td>
                                    <td>{{ "%.2f"|format(model.metrics.get("training_time", 0)) }}s</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Visual Comparison</h2>
                    
                    {% if plot_paths.accuracy %}
                    <div class="plot-container">
                        <h3>Accuracy Comparison</h3>
                        <img src="{{ plot_paths.accuracy }}" alt="Accuracy Comparison">
                    </div>
                    {% endif %}
                    
                    {% if plot_paths.test_loss %}
                    <div class="plot-container">
                        <h3>Loss Comparison</h3>
                        <img src="{{ plot_paths.test_loss }}" alt="Loss Comparison">
                    </div>
                    {% endif %}
                    
                    {% if plot_paths.f1_macro %}
                    <div class="plot-container">
                        <h3>F1 Score Comparison</h3>
                        <img src="{{ plot_paths.f1_macro }}" alt="F1 Score Comparison">
                    </div>
                    {% endif %}
                    
                    {% if plot_paths.training_time %}
                    <div class="plot-container">
                        <h3>Training Time Comparison</h3>
                        <img src="{{ plot_paths.training_time }}" alt="Training Time Comparison">
                    </div>
                    {% endif %}
                </div>
                
                <div class="summary">
                    <h2>Recommendation</h2>
                    {% set best_model = {'name': '', 'accuracy': 0} %}
                    {% for model in models_data %}
                        {% if model.metrics.get("test_accuracy", 0) > best_model.accuracy %}
                            {% set _ = best_model.update({'name': model.name, 'accuracy': model.metrics.get("test_accuracy", 0)}) %}
                        {% endif %}
                    {% endfor %}
                    
                    <p>Based on the comparison, <strong>{{ best_model.name }}</strong> performs best with an accuracy of {{ "%.2f"|format(best_model.accuracy * 100) }}%.</p>
                </div>
                
                <div class="footer">
                    <p>{{ project_name }} | Deep Learning for Plant Disease Detection</p>
                </div>
            </div>
        </body>
        </html>
        """

        template = Template(template_str)
        return template.render(**context)
