"""
Utility functions for Motor Imagery training pipeline

This module provides helper functions for logging, visualization, metadata handling,
and other common tasks used throughout the training process.
"""

import json
import logging
import os
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def setup_logging(debug: bool = False) -> None:
    """
    Configure logging for the training pipeline
    
    Good logging is essential for debugging and monitoring training progress.
    We use different log levels to control verbosity:
    - INFO: Progress updates, results, important events
    - DEBUG: Detailed internal state, parameter values
    - WARNING: Potential issues that don't stop execution
    - ERROR: Serious problems that may cause failure
    
    Args:
        debug: If True, enable DEBUG level logging
    """
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Configure logging format
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Reduce verbosity of some third-party libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    
    if debug:
        logging.info("Debug logging enabled")
    else:
        logging.info("Logging configured (INFO level)")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    out_path: str,
    normalize: bool = True
) -> None:
    """
    Create and save a confusion matrix plot
    
    Confusion matrices show the detailed classification performance by
    displaying true vs predicted labels. They reveal:
    - Overall accuracy (diagonal elements)
    - Class-specific errors (off-diagonal elements)
    - Class imbalances in predictions
    
    Args:
        y_true: True labels [n_samples]
        y_pred: Predicted labels [n_samples]
        labels: Class names for display
        out_path: Path to save the plot
        normalize: If True, normalize by true class counts
    """
    logging.info(f"Creating confusion matrix plot: {out_path}")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        # Normalize by true class counts (rows sum to 1)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot confusion matrix using seaborn for better aesthetics
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add performance metrics as text
    accuracy = np.trace(cm) / np.sum(cm) if not normalize else np.mean(y_true == y_pred)
    plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.3f}', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Confusion matrix saved to: {out_path}")


def save_metadata(meta: Dict[str, Any], out_path: str) -> None:
    """
    Save training metadata to JSON file
    
    Metadata preservation is crucial for reproducibility and model deployment.
    We save all the important parameters and results that allow someone to:
    - Understand how the model was trained
    - Reproduce the results
    - Apply the model correctly to new data
    
    Args:
        meta: Dictionary containing metadata to save
        out_path: Path to save JSON file
    """
    logging.info(f"Saving metadata to: {out_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Convert numpy types to JSON-serializable types
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert metadata
    json_meta = convert_numpy_types(meta)
    
    # Save to JSON file
    try:
        with open(out_path, 'w') as f:
            json.dump(json_meta, f, indent=2, sort_keys=True)
        logging.info("Metadata saved successfully")
    except Exception as e:
        logging.error(f"Failed to save metadata: {e}")
        raise


def class_distribution(labels: np.ndarray) -> Dict[str, int]:
    """
    Compute class distribution statistics
    
    Understanding class balance is important for:
    - Identifying potential bias in the dataset
    - Choosing appropriate evaluation metrics
    - Deciding on balancing strategies
    
    Args:
        labels: Array of class labels (0=LEFT, 1=RIGHT)
        
    Returns:
        Dictionary with class counts and statistics
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Map numeric labels to names
    label_names = {0: 'LEFT', 1: 'RIGHT'}
    
    distribution = {}
    total_samples = len(labels)
    
    for label, count in zip(unique_labels, counts):
        name = label_names.get(label, f'Class_{label}')
        distribution[name] = {
            'count': int(count),
            'percentage': float(count / total_samples * 100)
        }
    
    # Add summary statistics
    distribution['total'] = total_samples
    distribution['n_classes'] = len(unique_labels)
    
    # Calculate imbalance ratio (majority/minority)
    if len(counts) == 2:
        imbalance_ratio = max(counts) / min(counts)
        distribution['imbalance_ratio'] = float(imbalance_ratio)
    
    return distribution


def fix_label_case(s: str) -> str:
    """
    Normalize label strings to standard case
    
    Data files often have inconsistent capitalization for labels.
    This function standardizes them to uppercase for consistency.
    
    Args:
        s: Input label string
        
    Returns:
        Normalized label string ('LEFT' or 'RIGHT')
    """
    if not isinstance(s, str):
        s = str(s)
    
    s_upper = s.strip().upper()
    
    # Handle common variations
    if s_upper in ['L', 'LEFT', 'LEFT_HAND']:
        return 'LEFT'
    elif s_upper in ['R', 'RIGHT', 'RIGHT_HAND']:
        return 'RIGHT'
    else:
        return s_upper


def validate_data_shapes(
    epochs: np.ndarray,
    labels: np.ndarray,
    expected_n_channels: int = None
) -> None:
    """
    Validate that data arrays have expected shapes and properties
    
    Early validation prevents cryptic errors later in the pipeline.
    This function checks common issues with EEG data formatting.
    
    Args:
        epochs: Epoch data array
        labels: Label array
        expected_n_channels: Expected number of EEG channels
        
    Raises:
        ValueError: If data shapes or properties are invalid
    """
    # Check basic array properties
    if not isinstance(epochs, np.ndarray):
        raise ValueError("Epochs must be a numpy array")
    
    if not isinstance(labels, np.ndarray):
        raise ValueError("Labels must be a numpy array")
    
    # Check dimensions
    if epochs.ndim != 3:
        raise ValueError(f"Epochs must be 3D (n_epochs x n_channels x n_samples), got {epochs.ndim}D")
    
    if labels.ndim != 1:
        raise ValueError(f"Labels must be 1D, got {labels.ndim}D")
    
    # Check matching sample counts
    if epochs.shape[0] != labels.shape[0]:
        raise ValueError(f"Epoch count ({epochs.shape[0]}) != label count ({labels.shape[0]})")
    
    # Check for empty data
    if epochs.shape[0] == 0:
        raise ValueError("No epochs provided")
    
    if epochs.shape[1] == 0:
        raise ValueError("No channels in epoch data")
    
    if epochs.shape[2] == 0:
        raise ValueError("No time samples in epoch data")
    
    # Check channel count if specified
    if expected_n_channels is not None and epochs.shape[1] != expected_n_channels:
        raise ValueError(f"Expected {expected_n_channels} channels, got {epochs.shape[1]}")
    
    # Check for valid labels
    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError(f"Expected 2 classes, found {len(unique_labels)}: {unique_labels}")
    
    if not np.array_equal(unique_labels, [0, 1]):
        raise ValueError(f"Labels must be 0 (LEFT) and 1 (RIGHT), got: {unique_labels}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(epochs)):
        raise ValueError("Epochs contain NaN values")
    
    if np.any(np.isinf(epochs)):
        raise ValueError("Epochs contain infinite values")
    
    logging.info(f"✓ Data validation passed: {epochs.shape[0]} epochs, {epochs.shape[1]} channels, {epochs.shape[2]} samples")


def print_training_summary(
    config,
    data_info: Dict[str, Any],
    cv_scores: List[float],
    model_info: Dict[str, Any]
) -> None:
    """
    Print a comprehensive training summary
    
    This provides a human-readable summary of the entire training process,
    making it easy to understand what was done and how well it worked.
    
    Args:
        config: Training configuration object
        data_info: Information about the loaded data
        cv_scores: Cross-validation accuracy scores
        model_info: Information about the trained model
    """
    print("\n" + "="*80)
    print("MOTOR IMAGERY TRAINING SUMMARY")
    print("="*80)
    
    # Data information
    print(f"\n📊 DATA INFORMATION:")
    print(f"   Source: {data_info.get('source', 'Unknown')}")
    print(f"   Total epochs: {data_info.get('n_epochs', 'Unknown')}")
    print(f"   Channels: {data_info.get('n_channels', 'Unknown')}")
    print(f"   Sampling rate: {config.fs} Hz")
    print(f"   Epoch duration: {data_info.get('epoch_duration', 'Unknown')}s")
    
    # Class distribution
    if 'class_distribution' in data_info:
        dist = data_info['class_distribution']
        print(f"   Class balance: LEFT={dist.get('LEFT', {}).get('count', '?')}, RIGHT={dist.get('RIGHT', {}).get('count', '?')}")
    
    # Preprocessing
    print(f"\n⚙️  PREPROCESSING:")
    print(f"   Notch filter: {config.notch_hz} Hz")
    print(f"   Band-pass: {config.bp_low}-{config.bp_high} Hz")
    print(f"   Artifact rejection: PTP < {config.ptp_uV} µV, Z-score < {config.z_thresh}")
    
    # Model configuration
    print(f"\n🤖 MODEL CONFIGURATION:")
    print(f"   Algorithm: {config.classifier.upper()}")
    print(f"   Feature extraction: {'FBCSP' if len(config.fbcsp_bands) > 1 else 'CSP'}")
    print(f"   CSP components: {config.csp_components}")
    if len(config.fbcsp_bands) > 1:
        print(f"   Frequency bands: {config.fbcsp_bands}")
    
    # Cross-validation results
    print(f"\n📈 CROSS-VALIDATION RESULTS:")
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    print(f"   Mean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"   Fold scores: {[f'{score:.3f}' for score in cv_scores]}")
    print(f"   Best fold: {max(cv_scores):.3f}")
    print(f"   Worst fold: {min(cv_scores):.3f}")
    
    # Performance interpretation
    if mean_acc >= 0.8:
        performance = "Excellent! 🎉"
    elif mean_acc >= 0.7:
        performance = "Good 👍"
    elif mean_acc >= 0.6:
        performance = "Fair 🤔"
    else:
        performance = "Poor - consider more data or different approach 😕"
    
    print(f"   Performance: {performance}")
    
    # Output files
    print(f"\n💾 OUTPUT FILES:")
    print(f"   Model: {config.out_model}")
    print(f"   Metadata: {config.out_meta}")
    print(f"   Confusion matrix: {config.out_confmat}")
    
    print("\n" + "="*80)
    print("Training complete! 🚀")
    print("="*80)
