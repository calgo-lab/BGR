"""
Ensemble summary utilities for BGR seed ensemble experiments.

This module provides utilities for aggregating metrics across multiple seed runs,
creating ensemble summaries, and generating visualization plots with confidence bands.
"""

import json
import csv
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


def save_ensemble_summary(
    aggregated: dict,
    output_dir: str,
    all_results: List[dict],
    experiment_type: str,
    n_seeds: int
) -> tuple[Path, Path]:
    """
    Save ensemble summary as JSON and CSV files.

    Args:
        aggregated: Aggregated metrics (mean, std for each key)
        output_dir: Directory to save summary files
        all_results: List of per-seed result dictionaries
        experiment_type: Name of the experiment
        n_seeds: Number of seeds in ensemble

    Returns:
        Tuple of (JSON path, CSV path)
    """
    output_dir = Path(output_dir)

    summary_data = {
        'experiment_type': experiment_type,
        'n_seeds': n_seeds,
        'seeds': [r['seed'] for r in all_results],
        'metrics': _prepare_metrics_for_json(aggregated)
    }

    json_path = output_dir / 'ensemble_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=_json_serializer)

    flat_data = _flatten_metrics_for_csv(aggregated)
    flat_data['experiment_type'] = experiment_type
    flat_data['n_seeds'] = n_seeds
    flat_data['seeds'] = ','.join(str(r['seed']) for r in all_results)

    csv_path = output_dir / 'ensemble_summary.csv'
    df_rows = []
    for key, value in flat_data.items():
        if isinstance(value, dict):
            for subkey, subval in value.items():
                df_rows.append({'metric': key, 'stat': subkey, 'value': subval})
        elif not isinstance(value, (list, dict)):
            df_rows.append({'metric': key, 'stat': 'value', 'value': value})
    
    if df_rows:
        import pandas as pd
        df = pd.DataFrame(df_rows)
        df.to_csv(csv_path, index=False)

    seed_details_dir = output_dir / 'seed_details'
    seed_details_dir.mkdir(exist_ok=True)
    for result in all_results:
        seed_path = seed_details_dir / f'seed_{result["seed"]}.json'
        with open(seed_path, 'w') as f:
            json.dump(_prepare_metrics_for_json(result['final_epoch_metrics']), f, indent=2, default=_json_serializer)

    return json_path, csv_path


def _prepare_metrics_for_json(metrics: dict) -> dict:
    """Prepare metrics dict for JSON serialization."""
    result = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            result[key] = float(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, list):
            result[key] = [_prepare_metrics_for_json(item) if isinstance(item, dict) else float(v) if isinstance(v, (np.integer, np.floating)) else v for item in value]
        elif isinstance(value, dict):
            result[key] = _prepare_metrics_for_json(value)
        elif value is None or isinstance(value, (str, int, float, bool)):
            result[key] = value
    return result


def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [_json_serializer(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _json_serializer(v) for k, v in obj.items()}
    raise TypeError(f"Type {type(obj)} not serializable")


def _flatten_metrics_for_csv(metrics: dict) -> dict:
    """Flatten nested metrics dict for CSV export."""
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subval in value.items():
                flat[f'{key}_{subkey}'] = subval if not isinstance(subval, (np.integer, np.floating)) else float(subval)
        elif isinstance(value, (np.integer, np.floating)):
            flat[key] = float(value)
        elif value is None or isinstance(value, (str, int, float, bool)):
            flat[key] = value
    return flat


def extract_backward_compat_metrics(aggregated: dict) -> dict:
    """
    Extract ensemble means for backward compatibility with existing code.

    This strips the '_mean' suffix from metric keys so that the returned
    dict has the same format as a single-run result dict.

    Args:
        aggregated: Aggregated metrics dict

    Returns:
        Dict with mean values, keys without '_mean' suffix
    """
    compat = {}
    for key, value in aggregated.items():
        if key.endswith('_mean') and isinstance(value, (int, float, np.integer, np.floating)):
            base_key = key[:-5]
            compat[base_key] = float(value)
    return compat


def plot_ensemble_training_curves(
    aggregated_history: List[dict],
    output_dir: str,
    wandb_log: bool = True,
    group_name: Optional[str] = None
) -> Path:
    """
    Plot training and validation curves with confidence bands (mean +/- std).

    Creates PDF plots showing the training trajectory across all seeds,
    with shaded regions indicating the standard deviation.

    Args:
        aggregated_history: List of per-epoch aggregated metrics
        output_dir: Directory to save the plot
        wandb_log: Whether to log the plot to wandb
        group_name: Name for the plot title

    Returns:
        Path to the saved PDF file
    """
    if not aggregated_history:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h.get('epoch', i+1) for i, h in enumerate(aggregated_history)]

    key_metrics = [
        ('train_loss', 'Train Loss'),
        ('val_loss', 'Validation Loss'),
        ('train_Horizon_accuracy', 'Train Horizon Accuracy'),
        ('val_Horizon_accuracy', 'Validation Horizon Accuracy'),
        ('val_Depth_IoU', 'Validation Depth IoU'),
        ('test_Horizon_accuracy', 'Test Horizon Accuracy'),
    ]

    valid_metrics = []
    for key, title in key_metrics:
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'
        if mean_key in aggregated_history[0] and std_key in aggregated_history[0]:
            means = [h.get(mean_key) for h in aggregated_history]
            stds = [h.get(std_key, 0) for h in aggregated_history]
            if any(m is not None for m in means):
                valid_metrics.append((key, title, mean_key, std_key))

    if not valid_metrics:
        return None

    n_metrics = len(valid_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes

    for idx, (key, title, mean_key, std_key) in enumerate(valid_metrics):
        ax = axes[idx]
        means = []
        stds = []
        valid_epochs = []
        for i, h in enumerate(aggregated_history):
            mean_val = h.get(mean_key)
            std_val = h.get(std_key, 0)
            if mean_val is not None:
                means.append(mean_val)
                stds.append(std_val if std_val is not None else 0)
                valid_epochs.append(epochs[i])

        if not means:
            ax.text(0.5, 0.5, f'No data for {title}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        means = np.array(means)
        stds = np.array(stds)
        valid_epochs = np.array(valid_epochs)

        ax.plot(valid_epochs, means, 'b-', linewidth=2, label='Mean')
        ax.fill_between(valid_epochs, means - stds, means + stds, alpha=0.3, color='blue', label='+/- Std')

        final_mean = means[-1] if len(means) > 0 else 0
        final_std = stds[-1] if len(stds) > 0 else 0
        ax.set_title(f'{title}\nFinal: {final_mean:.4f} +/- {final_std:.4f}', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title.split()[-1] if len(title.split()) > 1 else title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(valid_metrics), len(axes)):
        axes[idx].axis('off')

    title_suffix = f' ({group_name})' if group_name else ''
    plt.suptitle(f'Ensemble Training Curves{title_suffix}', fontsize=14, y=1.02)
    plt.tight_layout()

    pdf_path = output_dir / 'ensemble_training_curves.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    plt.close(fig)

    if wandb_log:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({'ensemble_training_curves': wandb.Image(str(pdf_path))})
        except Exception:
            pass

    return pdf_path


def create_ensemble_summary_figure(
    aggregated: dict,
    experiment_type: str,
    n_seeds: int
) -> plt.Figure:
    """
    Create a summary figure showing key metrics with error bars.

    Args:
        aggregated: Aggregated metrics dict
        experiment_type: Name of the experiment
        n_seeds: Number of seeds

    Returns:
        Matplotlib Figure
    """
    key_metrics = [
        ('test_Horizon_accuracy', 'Horizon Accuracy'),
        ('test_Horizon_topk_accuracy', 'Horizon Top-5 Acc'),
        ('test_Depth_IoU', 'Depth IoU'),
        ('test_Bodenart_accuracy', 'Bodenart Acc'),
        ('val_loss', 'Val Loss'),
    ]

    valid_metrics = []
    for key, title in key_metrics:
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'
        if mean_key in aggregated and std_key in aggregated:
            mean_val = aggregated.get(mean_key)
            std_val = aggregated.get(std_key, 0)
            if mean_val is not None and isinstance(mean_val, (int, float, np.integer, np.floating)):
                valid_metrics.append((key, title, mean_val, std_val))

    if not valid_metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No key metrics available', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(max(6, len(valid_metrics) * 2), 5))

    x_pos = range(len(valid_metrics))
    labels = [m[1] for m in valid_metrics]
    means = [m[2] for m in valid_metrics]
    stds = [m[3] for m in valid_metrics]

    bars = ax.bar(x_pos, means, yerr=stds, capsize=8, color='steelblue', alpha=0.7, edgecolor='black')

    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}\n+/- {std:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(f'{experiment_type} Ensemble (n={n_seeds})', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig