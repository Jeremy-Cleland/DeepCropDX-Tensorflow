#!/usr/bin/env python3
"""
Command line interface for the model registry.
This script allows you to manage the model registry from the command line.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from tabulate import tabulate

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.model_registry.registry_manager import ModelRegistryManager


def list_models(registry, args):
    """List all models in the registry"""
    models = registry.list_models()

    if not models:
        print("No models found in registry")
        return

    # Collect details for each model
    model_details = []
    for model_name in models:
        model_info = registry._registry["models"][model_name]
        best_run_id = model_info.get("best_run")

        if best_run_id:
            best_run = model_info["runs"][best_run_id]
            accuracy = best_run.get("metrics", {}).get("test_accuracy", 0)
            loss = best_run.get("metrics", {}).get("test_loss", 0)
        else:
            accuracy = 0
            loss = 0

        model_details.append(
            {
                "Model": model_name,
                "Total Runs": model_info.get("total_runs", 0),
                "Best Run": best_run_id,
                "Best Accuracy": f"{accuracy:.4f}",
                "Best Loss": f"{loss:.4f}",
            }
        )

    # Create DataFrame and display as table
    df = pd.DataFrame(model_details)
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))


def list_runs(registry, args):
    """List all runs for a specific model"""
    model_name = args.model

    if model_name not in registry._registry["models"]:
        print(f"Model {model_name} not found in registry")
        return

    runs = registry.list_runs(model_name)
    model_info = registry._registry["models"][model_name]
    best_run_id = model_info.get("best_run")

    if not runs:
        print(f"No runs found for model {model_name}")
        return

    # Collect details for each run
    run_details = []
    for run_id in runs:
        run_info = model_info["runs"][run_id]
        accuracy = run_info.get("metrics", {}).get("test_accuracy", 0)
        loss = run_info.get("metrics", {}).get("test_loss", 0)
        timestamp = run_info.get("timestamp", "")
        status = run_info.get("status", "unknown")
        is_best = run_id == best_run_id

        run_details.append(
            {
                "Run ID": run_id,
                "Timestamp": timestamp,
                "Accuracy": f"{accuracy:.4f}",
                "Loss": f"{loss:.4f}",
                "Status": status,
                "Best": "✓" if is_best else "",
            }
        )

    # Sort by timestamp (newest first)
    run_details.sort(key=lambda x: x["Timestamp"], reverse=True)

    # Create DataFrame and display as table
    df = pd.DataFrame(run_details)
    print(f"\nRuns for model: {model_name}")
    print(f"Total runs: {len(runs)}")
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))


def show_details(registry, args):
    """Show detailed information about a model or run"""
    model_name = args.model
    run_id = args.run

    if model_name not in registry._registry["models"]:
        print(f"Model {model_name} not found in registry")
        return

    model_info = registry._registry["models"][model_name]

    if run_id:
        # Show run details
        if run_id not in model_info["runs"]:
            print(f"Run {run_id} not found for model {model_name}")
            return

        run_info = model_info["runs"][run_id]
        print(f"\nDetails for {model_name} run {run_id}:")
        print(f"  Path: {run_info.get('path')}")
        print(f"  Timestamp: {run_info.get('timestamp')}")
        print(f"  Status: {run_info.get('status')}")
        print(f"  Model file: {run_info.get('model_path')}")
        print(f"  Has checkpoints: {run_info.get('has_checkpoints')}")
        print(f"  Has TensorBoard logs: {run_info.get('has_tensorboard')}")

        print("\nMetrics:")
        metrics = run_info.get("metrics", {})
        for key, value in metrics.items():
            # Format the value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = str(value)
            print(f"  {key}: {formatted_value}")
    else:
        # Show model details
        print(f"\nDetails for model {model_name}:")
        print(f"  Total runs: {model_info.get('total_runs')}")
        print(f"  Best run: {model_info.get('best_run')}")
        print(f"  Last run: {model_info.get('last_run')}")

        if model_info.get("best_run"):
            best_run = model_info["runs"][model_info.get("best_run")]
            print("\nBest run metrics:")
            metrics = best_run.get("metrics", {})
            for key, value in metrics.items():
                if key.startswith(("test_", "val_")) and isinstance(
                    value, (int, float)
                ):
                    print(f"  {key}: {value:.6f}")


def scan_trials(registry, args):
    """Scan trials directory for new models and runs"""
    new_runs = registry.scan_trials(rescan=args.rescan)
    if new_runs > 0:
        print(f"Added {new_runs} new runs to registry")
    else:
        print("No new runs found")


def compare_models(registry, args):
    """Compare multiple models"""
    if args.models:
        model_names = args.models
    else:
        # Use top N models if no names provided
        top_models = registry.get_best_models(top_n=args.top)
        model_names = [model["name"] for model in top_models]

    if not model_names:
        print("No models to compare")
        return

    print(f"Comparing models: {', '.join(model_names)}")

    # Get metrics to compare
    if args.metrics:
        metrics = args.metrics
    else:
        metrics = ["test_accuracy", "test_loss", "training_time"]

    # Compare models
    comparison_df = registry.compare_models(
        model_names=model_names, metrics=metrics, plot=True, output_dir=args.output_dir
    )

    if comparison_df.empty:
        print("No data available for comparison")
        return

    # Display comparison table
    print("\nModel Comparison:")
    print(tabulate(comparison_df, headers="keys", tablefmt="pretty", showindex=False))

    # Print path to generated plots
    if args.output_dir:
        print(f"\nComparison plots saved to: {args.output_dir}")
    else:
        print(
            f"\nComparison plots saved to: {registry.paths.trials_dir / 'comparisons'}"
        )


def export_registry(registry, args):
    """Export the registry to a file"""
    output_path = args.output
    path = registry.export_registry(output_path)
    print(f"Registry exported to {path}")


def import_registry(registry, args):
    """Import a registry from a file"""
    input_path = args.input
    success = registry.import_registry(input_path, merge=not args.replace)
    if success:
        print(f"Registry imported from {input_path}")
    else:
        print(f"Failed to import registry from {input_path}")


def generate_report(registry, args):
    """Generate an HTML report of the registry"""
    output_path = args.output
    path = registry.generate_registry_report(output_path)
    print(f"Registry report generated at {path}")


def delete_run(registry, args):
    """Delete a run from the registry"""
    model_name = args.model
    run_id = args.run

    if not args.force:
        confirmation = input(
            f"Are you sure you want to delete run {run_id} of model {model_name}? (y/n): "
        )
        if confirmation.lower() != "y":
            print("Deletion cancelled")
            return

    success = registry.delete_run(model_name, run_id, delete_files=args.delete_files)
    if success:
        print(f"Run {run_id} of model {model_name} deleted from registry")
        if args.delete_files:
            print("Run files were also deleted from disk")
    else:
        print(f"Failed to delete run {run_id} of model {model_name}")


def main():
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Model Registry CLI - Manage trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  registry_cli.py list                            # List all models
  registry_cli.py runs --model ResNet50           # List all runs for ResNet50
  registry_cli.py details --model ResNet50        # Show details for ResNet50
  registry_cli.py details --model ResNet50 --run run_20250304_123456_001  # Show run details
  registry_cli.py scan                            # Scan for new models and runs
  registry_cli.py compare --models ResNet50 MobileNetV2  # Compare models
  registry_cli.py report                          # Generate HTML report
        """,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List models command
    list_parser = subparsers.add_parser("list", help="List all models in the registry")

    # List runs command
    runs_parser = subparsers.add_parser(
        "runs", help="List all runs for a specific model"
    )
    runs_parser.add_argument("--model", required=True, help="Name of the model")

    # Show details command
    details_parser = subparsers.add_parser(
        "details", help="Show detailed information about a model or run"
    )
    details_parser.add_argument("--model", required=True, help="Name of the model")
    details_parser.add_argument("--run", help="ID of the run (optional)")

    # Scan trials command
    scan_parser = subparsers.add_parser(
        "scan", help="Scan trials directory for new models and runs"
    )
    scan_parser.add_argument(
        "--rescan",
        action="store_true",
        help="Rescan all trials, even if already in registry",
    )

    # Compare models command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument(
        "--models", nargs="+", help="Names of models to compare"
    )
    compare_parser.add_argument("--metrics", nargs="+", help="Metrics to compare")
    compare_parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top models to compare if no models specified",
    )
    compare_parser.add_argument(
        "--output-dir", help="Directory to save comparison plots"
    )

    # Export registry command
    export_parser = subparsers.add_parser(
        "export", help="Export the registry to a file"
    )
    export_parser.add_argument("--output", help="Path to export the registry")

    # Import registry command
    import_parser = subparsers.add_parser(
        "import", help="Import a registry from a file"
    )
    import_parser.add_argument(
        "--input", required=True, help="Path to the registry file"
    )
    import_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing registry instead of merging",
    )

    # Generate report command
    report_parser = subparsers.add_parser(
        "report", help="Generate an HTML report of the registry"
    )
    report_parser.add_argument("--output", help="Path to save the report")

    # Delete run command
    delete_parser = subparsers.add_parser(
        "delete", help="Delete a run from the registry"
    )
    delete_parser.add_argument("--model", required=True, help="Name of the model")
    delete_parser.add_argument("--run", required=True, help="ID of the run")
    delete_parser.add_argument(
        "--delete-files", action="store_true", help="Also delete run files from disk"
    )
    delete_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )

    # Parse arguments
    args = parser.parse_args()

    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return

    # Initialize registry manager
    registry = ModelRegistryManager()

    # Execute the appropriate command
    commands = {
        "list": list_models,
        "runs": list_runs,
        "details": show_details,
        "scan": scan_trials,
        "compare": compare_models,
        "export": export_registry,
        "import": import_registry,
        "report": generate_report,
        "delete": delete_run,
    }

    if args.command in commands:
        commands[args.command](registry, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
