#!/usr/bin/env python3
"""
Motor Imagery Classifier Training Script

This is the main entry point for training motor imagery classifiers.
It orchestrates the entire pipeline from data loading to model saving.

Usage Examples:
    # Train on synthetic data (for testing)
    python train.py --fake --user demo

    # Train on CSV data
    python train.py --csv data/motor_imagery.csv --user alice

    # Train on XDF data
    python train.py --xdf data/session1.xdf --user bob

    # Use custom parameters
    python train.py --csv data.csv --user test --classifier svm --csp-components 8 --cv-folds 10

Author: AI Assistant (Claude)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

import numpy as np

# Import our training modules
from mi_training.config import Config, ensure_output_dirs, validate_config
from mi_training.data_io import load_csv, load_xdf, resolve_motor_channels
from mi_training.preprocessing import apply_filters, reject_artifacts_ptp, reject_artifacts_zscore
from mi_training.epoching import build_epochs, baseline_correct, balance_classes
from mi_training.models import (
    build_pipeline_csp_lda, build_pipeline_csp_svm,
    build_pipeline_fbcsp_lda, build_pipeline_fbcsp_svm,
    cross_validate, fit_and_save
)
from mi_training.utils import (
    setup_logging, plot_confusion_matrix, save_metadata,
    class_distribution, validate_data_shapes, print_training_summary
)
from mi_training.fake_data import synthesize_mi_trials


def load_data(config: Config, data_source: str, **kwargs) -> tuple:
    """
    Load EEG data from specified source
    
    This function handles the different data loading pathways and returns
    a consistent format regardless of the input source.
    
    Args:
        config: Training configuration
        data_source: Type of data source ('csv', 'xdf', 'fake')
        **kwargs: Additional arguments for data loading
        
    Returns:
        Tuple of (data, markers, fs, ch_names)
    """
    logging.info(f"Loading data from {data_source} source")
    
    if data_source == 'fake':
        logging.info("Generating synthetic motor imagery data")
        return synthesize_mi_trials(
            n_trials_per_class=kwargs.get('n_trials', 60),
            fs=config.fs,
            trial_dur=4.0,
            seed=config.seed
        )
    
    elif data_source == 'csv':
        csv_path = kwargs.get('csv_path')
        if not csv_path:
            raise ValueError("CSV path must be specified for CSV data source")
        
        return load_csv(
            path=csv_path,
            fs=config.fs,
            ch_names=None,  # Auto-detect
            marker_col=kwargs.get('marker_col', 'marker')
        )
    
    elif data_source == 'xdf':
        xdf_path = kwargs.get('xdf_path')
        if not xdf_path:
            raise ValueError("XDF path must be specified for XDF data source")
        
        return load_xdf(
            path=xdf_path,
            stream_name=kwargs.get('stream_name', 'EEG'),
            marker_stream=kwargs.get('marker_stream', 'Markers')
        )
    
    else:
        raise ValueError(f"Unknown data source: {data_source}")


def preprocess_data(config: Config, data: np.ndarray, fs: int) -> np.ndarray:
    """
    Apply preprocessing pipeline to raw EEG data
    
    Args:
        config: Training configuration
        data: Raw EEG data [channels x samples]
        fs: Sampling frequency
        
    Returns:
        Preprocessed EEG data
    """
    logging.info("Starting EEG preprocessing")
    
    # Apply digital filters
    filtered_data = apply_filters(
        data=data,
        fs=fs,
        notch=config.notch_hz,
        bp_low=config.bp_low,
        bp_high=config.bp_high,
        resample=config.resample
    )
    
    # Update sampling rate if resampling was applied
    if config.resample and config.resample != fs:
        fs = config.resample
    
    logging.info("Preprocessing complete")
    return filtered_data


def extract_epochs(config: Config, data: np.ndarray, markers: list, fs: int) -> tuple:
    """
    Extract and preprocess epochs around motor imagery events
    
    Args:
        config: Training configuration
        data: Preprocessed EEG data [channels x samples]
        markers: List of (sample_index, label) tuples
        fs: Sampling frequency
        
    Returns:
        Tuple of (clean_epochs, clean_labels, time_vector)
    """
    logging.info("Extracting epochs around motor imagery markers")
    
    # Build epochs
    epochs, labels, time_vector = build_epochs(
        data=data,
        markers=markers,
        fs=fs,
        tmin=-0.2,  # 200ms baseline
        tmax=4.0    # 4s motor imagery period
    )
    
    # Apply baseline correction
    epochs = baseline_correct(
        epochs=epochs,
        fs=fs,
        tmin=-0.2,
        tmax=4.0,
        baseline=(-0.2, 0.0)
    )
    
    # Artifact rejection - PTP first
    epochs, labels, ptp_mask = reject_artifacts_ptp(
        epochs=epochs,
        labels=labels,
        fs=fs,
        ptp_uV=config.ptp_uV
    )
    
    # Then z-score rejection
    epochs, labels, zscore_mask = reject_artifacts_zscore(
        epochs=epochs,
        labels=labels,
        z_thresh=config.z_thresh
    )
    
    # Balance classes if needed
    class_dist = class_distribution(labels)
    if 'imbalance_ratio' in class_dist and class_dist['imbalance_ratio'] > 1.5:
        logging.info(f"Class imbalance detected (ratio: {class_dist['imbalance_ratio']:.2f}), applying balancing")
        epochs, labels = balance_classes(
            epochs=epochs,
            labels=labels,
            method="undersample",  # Conservative choice
            random_state=config.seed
        )
    
    # Validate final data
    validate_data_shapes(epochs, labels)
    
    logging.info(f"Final dataset: {epochs.shape[0]} epochs, {epochs.shape[1]} channels")
    return epochs, labels, time_vector


def select_motor_channels(config: Config, epochs: np.ndarray, ch_names: list) -> tuple:
    """
    Select motor cortex channels for analysis
    
    Args:
        config: Training configuration
        epochs: Epoch data [n_epochs x n_channels x n_samples]
        ch_names: List of channel names
        
    Returns:
        Tuple of (motor_epochs, motor_ch_names)
    """
    logging.info("Selecting motor cortex channels")
    
    # Resolve motor channel indices
    motor_indices, motor_names = resolve_motor_channels(
        ch_names=ch_names,
        motor_ch_names=config.motor_ch_names
    )
    
    # Extract motor channels
    motor_epochs = epochs[:, motor_indices, :]
    
    logging.info(f"Selected {len(motor_indices)} motor channels: {motor_names}")
    return motor_epochs, motor_names


def build_model_pipeline(config: Config, fs: int):
    """
    Build the appropriate model pipeline based on configuration
    
    Args:
        config: Training configuration
        fs: Sampling frequency
        
    Returns:
        Configured sklearn Pipeline
    """
    logging.info(f"Building {config.classifier.upper()} pipeline")
    
    # Determine if we should use FBCSP (multiple frequency bands)
    use_fbcsp = len(config.fbcsp_bands) > 1
    
    if use_fbcsp:
        logging.info(f"Using FBCSP with {len(config.fbcsp_bands)} frequency bands")
        if config.classifier == 'lda':
            pipeline = build_pipeline_fbcsp_lda(
                bands=config.fbcsp_bands,
                n_components=config.csp_components,
                fs=fs,
                random_state=config.seed
            )
        elif config.classifier == 'svm':
            pipeline = build_pipeline_fbcsp_svm(
                bands=config.fbcsp_bands,
                n_components=config.csp_components,
                fs=fs,
                C=config.svm_c,
                random_state=config.seed
            )
        else:
            raise ValueError(f"Unknown classifier: {config.classifier}")
    else:
        logging.info("Using standard CSP")
        if config.classifier == 'lda':
            pipeline = build_pipeline_csp_lda(
                n_components=config.csp_components,
                random_state=config.seed
            )
        elif config.classifier == 'svm':
            pipeline = build_pipeline_csp_svm(
                n_components=config.csp_components,
                C=config.svm_c,
                random_state=config.seed
            )
        else:
            raise ValueError(f"Unknown classifier: {config.classifier}")
    
    return pipeline


def train_and_evaluate(config: Config, pipeline, epochs: np.ndarray, labels: np.ndarray) -> dict:
    """
    Train model and perform cross-validation evaluation
    
    Args:
        config: Training configuration
        pipeline: Model pipeline to train
        epochs: Training epochs
        labels: Training labels
        
    Returns:
        Dictionary with training results
    """
    logging.info("Starting model training and evaluation")
    
    # Perform cross-validation
    cv_scores, y_true, y_pred = cross_validate(
        pipeline=pipeline,
        X=epochs,
        y=labels,
        cv=config.cv_folds,
        seed=config.seed
    )
    
    # Create confusion matrix plot
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=['LEFT', 'RIGHT'],
        out_path=config.out_confmat
    )
    
    # Fit final model on all data
    fit_and_save(
        pipeline=pipeline,
        X=epochs,
        y=labels,
        model_path=config.out_model
    )
    
    # Compile results
    results = {
        'cv_scores': cv_scores,
        'mean_accuracy': float(np.mean(cv_scores)),
        'std_accuracy': float(np.std(cv_scores)),
        'best_fold': float(max(cv_scores)),
        'worst_fold': float(min(cv_scores)),
        'n_folds': config.cv_folds
    }
    
    logging.info(f"Training complete. Mean CV accuracy: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")
    return results


def save_training_metadata(config: Config, data_info: dict, results: dict, 
                         motor_ch_names: list, training_time: float):
    """
    Save comprehensive training metadata
    
    Args:
        config: Training configuration
        data_info: Information about the training data
        results: Training results
        motor_ch_names: Names of motor channels used
        training_time: Total training time in seconds
    """
    logging.info("Saving training metadata")
    
    metadata = {
        # Training configuration
        'config': {
            'fs': config.fs,
            'notch_hz': config.notch_hz,
            'bp_low': config.bp_low,
            'bp_high': config.bp_high,
            'resample': config.resample,
            'motor_ch_names': config.motor_ch_names,
            'fbcsp_bands': config.fbcsp_bands,
            'csp_components': config.csp_components,
            'cv_folds': config.cv_folds,
            'classifier': config.classifier,
            'svm_c': config.svm_c,
            'ptp_uV': config.ptp_uV,
            'z_thresh': config.z_thresh,
            'seed': config.seed
        },
        
        # Data information
        'data': data_info,
        
        # Training results
        'results': results,
        
        # Technical details
        'motor_channels_used': motor_ch_names,
        'training_time_seconds': training_time,
        'training_date': datetime.now().isoformat(),
        'python_version': sys.version,
        
        # File paths
        'model_path': config.out_model,
        'confusion_matrix_path': config.out_confmat
    }
    
    save_metadata(metadata, config.out_meta)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Train Motor Imagery EEG Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on synthetic data (for testing)
  python train.py --fake --user demo
  
  # Train on CSV data with custom parameters
  python train.py --csv data/mi_data.csv --user alice --classifier svm --csp-components 8
  
  # Train on XDF data with debugging
  python train.py --xdf recording.xdf --user bob --debug
  
  # Use Filter Bank CSP with custom frequency bands
  python train.py --csv data.csv --user test --fbcsp-bands 8,12 12,16 16,20 20,26 26,30
        """
    )
    
    # Data source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--fake', action='store_true',
                             help='Use synthetic motor imagery data')
    source_group.add_argument('--csv', type=str,
                             help='Path to CSV file with EEG data and markers')
    source_group.add_argument('--xdf', type=str,
                             help='Path to XDF file with EEG and marker streams')
    
    # Required parameters
    parser.add_argument('--user', type=str, required=True,
                       help='User identifier for model naming')
    
    # Data loading parameters
    parser.add_argument('--marker-col', type=str, default='marker',
                       help='Name of marker column in CSV (default: marker)')
    parser.add_argument('--stream-name', type=str, default='EEG',
                       help='Name of EEG stream in XDF (default: EEG)')
    parser.add_argument('--marker-stream', type=str, default='Markers',
                       help='Name of marker stream in XDF (default: Markers)')
    parser.add_argument('--n-trials', type=int, default=60,
                       help='Number of trials per class for synthetic data (default: 60)')
    
    # Signal processing parameters
    parser.add_argument('--fs', type=int, default=250,
                       help='Sampling frequency in Hz (default: 250)')
    parser.add_argument('--notch', type=int, choices=[50, 60], default=60,
                       help='Notch filter frequency (default: 60)')
    parser.add_argument('--bp-low', type=float, default=8.0,
                       help='Band-pass low cutoff in Hz (default: 8.0)')
    parser.add_argument('--bp-high', type=float, default=30.0,
                       help='Band-pass high cutoff in Hz (default: 30.0)')
    parser.add_argument('--resample', type=int,
                       help='Resample to this frequency (optional)')
    
    # Channel selection
    parser.add_argument('--motor-channels', type=str, nargs='+', default=['C3', 'Cz', 'C4'],
                       help='Motor cortex channel names (default: C3 Cz C4)')
    
    # Feature extraction parameters
    parser.add_argument('--csp-components', type=int, default=6,
                       help='Number of CSP components (default: 6)')
    parser.add_argument('--fbcsp-bands', type=str, nargs='+',
                       help='FBCSP frequency bands as "low,high" pairs (e.g., "8,12" "12,16")')
    
    # Classification parameters
    parser.add_argument('--classifier', choices=['lda', 'svm'], default='lda',
                       help='Classifier type (default: lda)')
    parser.add_argument('--svm-c', type=float, default=1.0,
                       help='SVM regularization parameter (default: 1.0)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Cross-validation folds (default: 5)')
    
    # Artifact rejection parameters
    parser.add_argument('--ptp-threshold', type=float, default=200.0,
                       help='Peak-to-peak threshold in µV (default: 200.0)')
    parser.add_argument('--z-threshold', type=float, default=5.0,
                       help='Z-score threshold for outlier rejection (default: 5.0)')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models (default: models)')
    parser.add_argument('--model-name', type=str,
                       help='Custom model filename (default: auto-generated)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser


def parse_fbcsp_bands(bands_str: list) -> list:
    """
    Parse FBCSP frequency bands from command line arguments
    
    Args:
        bands_str: List of "low,high" strings
        
    Returns:
        List of (low, high) tuples
    """
    if not bands_str:
        return [(8, 12), (12, 16), (16, 20), (20, 26), (26, 30)]  # Default bands
    
    bands = []
    for band_str in bands_str:
        try:
            low, high = map(float, band_str.split(','))
            bands.append((low, high))
        except ValueError:
            raise ValueError(f"Invalid frequency band format: {band_str}. Use 'low,high' format.")
    
    return bands


def main():
    """Main training function"""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    
    logging.info("Starting Motor Imagery classifier training")
    logging.info(f"User: {args.user}")
    
    start_time = time.time()
    
    try:
        # Create configuration
        config = Config(
            fs=args.fs,
            notch_hz=args.notch,
            bp_low=args.bp_low,
            bp_high=args.bp_high,
            resample=args.resample,
            motor_ch_names=args.motor_channels,
            fbcsp_bands=parse_fbcsp_bands(args.fbcsp_bands),
            csp_components=args.csp_components,
            cv_folds=args.cv_folds,
            classifier=args.classifier,
            svm_c=args.svm_c,
            ptp_uV=args.ptp_threshold,
            z_thresh=args.z_threshold,
            seed=args.seed
        )
        
        # Set output paths
        if args.model_name:
            model_filename = args.model_name
        else:
            model_filename = f"mi_{config.classifier}_{args.user}.joblib"
        
        config.out_model = os.path.join(args.output_dir, model_filename)
        config.out_meta = os.path.join(args.output_dir, f"mi_meta_{args.user}.json")
        config.out_confmat = os.path.join(args.output_dir, f"mi_confusion_{args.user}.png")
        
        # Validate configuration and create output directories
        validate_config(config)
        ensure_output_dirs(config)
        
        # Load data
        if args.fake:
            data_source = 'fake'
            data, markers, fs, ch_names = load_data(
                config, data_source, n_trials=args.n_trials
            )
        elif args.csv:
            data_source = 'csv'
            data, markers, fs, ch_names = load_data(
                config, data_source, csv_path=args.csv, 
                marker_col=args.marker_col
            )
        elif args.xdf:
            data_source = 'xdf'
            data, markers, fs, ch_names = load_data(
                config, data_source, xdf_path=args.xdf,
                stream_name=args.stream_name, marker_stream=args.marker_stream
            )
        
        # Store data information
        data_info = {
            'source': data_source,
            'original_fs': fs,
            'n_channels_original': len(ch_names),
            'channel_names': ch_names,
            'n_markers': len(markers),
            'data_duration_seconds': data.shape[1] / fs,
            'class_distribution': class_distribution([label for _, label in markers])
        }
        
        # Preprocess data
        if fs != config.fs:
            logging.warning(f"Data sampling rate ({fs}) differs from config ({config.fs})")
            config.fs = fs  # Use actual data sampling rate
        
        preprocessed_data = preprocess_data(config, data, fs)
        
        # Extract epochs
        epochs, labels, time_vector = extract_epochs(config, preprocessed_data, markers, fs)
        
        # Update data info with epoch information
        data_info.update({
            'n_epochs': epochs.shape[0],
            'epoch_duration': (time_vector[-1] - time_vector[0]),
            'n_samples_per_epoch': epochs.shape[2],
            'final_class_distribution': class_distribution(labels)
        })
        
        # Select motor channels
        motor_epochs, motor_ch_names = select_motor_channels(config, epochs, ch_names)
        
        # Build model pipeline
        pipeline = build_model_pipeline(config, fs)
        
        # Train and evaluate
        results = train_and_evaluate(config, pipeline, motor_epochs, labels)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Save metadata
        save_training_metadata(config, data_info, results, motor_ch_names, training_time)
        
        # Print summary
        print_training_summary(config, data_info, results['cv_scores'], results)
        
        logging.info(f"Training completed successfully in {training_time:.1f} seconds")
        return 0
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Training failed: {e}")
        if args.debug:
            raise  # Re-raise with full traceback in debug mode
        return 1


if __name__ == '__main__':
    sys.exit(main())
