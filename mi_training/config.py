"""
Configuration settings for Motor Imagery training pipeline

This module defines the main configuration dataclass that controls all aspects
of the training pipeline, from data loading to model saving.
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class Config:
    """
    Configuration for Motor Imagery classifier training
    
    This dataclass contains all the parameters needed to control the training
    pipeline. Each parameter has a sensible default that works well for most
    motor imagery experiments.
    
    Hardware/Data Parameters:
    - fs: Sampling frequency in Hz (250 Hz is standard for OpenBCI)
    - notch_hz: Power line frequency for notch filtering (60 Hz US, 50 Hz EU)
    - resample: Target sampling rate if downsampling is needed
    
    Preprocessing Parameters:
    - bp_low/bp_high: Band-pass filter limits focusing on mu/beta rhythms
    - motor_ch_names: EEG channels over motor cortex (C3=left, C4=right motor)
    
    Feature Extraction:
    - fbcsp_bands: Frequency sub-bands for Filter Bank CSP
    - csp_components: Number of CSP spatial filters (more = more features)
    
    Classification:
    - classifier: "lda" (faster) or "svm" (potentially more accurate)
    - cv_folds: Cross-validation folds for performance estimation
    
    Artifact Rejection:
    - ptp_uV: Peak-to-peak amplitude threshold in microvolts
    - z_thresh: Z-score threshold for statistical outlier detection
    """
    
    # Sampling and filtering
    fs: int = 250                           # Sampling frequency (Hz)
    notch_hz: int = 60                      # Power line frequency (50 for EU, 60 for US)
    bp_low: float = 8.0                     # Band-pass low cut (Hz) - start of mu rhythm
    bp_high: float = 30.0                   # Band-pass high cut (Hz) - end of beta rhythm
    resample: Optional[int] = None          # Downsample to this rate if specified
    
    # Channel selection
    motor_ch_names: List[str] = None        # Motor cortex channels
    
    # Filter Bank CSP parameters
    fbcsp_bands: List[Tuple[float, float]] = None  # Frequency sub-bands
    csp_components: int = 6                 # Number of CSP components per band
    
    # Classification
    cv_folds: int = 5                       # Cross-validation folds
    classifier: str = "lda"                 # "lda" or "svm"
    svm_c: float = 1.0                      # SVM regularization parameter
    
    # Artifact rejection thresholds
    ptp_uV: float = 200.0                   # Peak-to-peak amplitude threshold (µV)
    z_thresh: float = 5.0                   # Z-score threshold for outlier detection
    
    # Reproducibility
    seed: int = 42                          # Random seed for reproducible results
    
    # Output paths
    out_model: str = "models/mi_csp_lda.joblib"     # Trained model file
    out_meta: str = "models/mi_meta.json"           # Training metadata
    out_confmat: str = "models/mi_confusion.png"    # Confusion matrix plot
    
    def __post_init__(self):
        """
        Initialize default values that depend on other parameters
        
        This runs after the dataclass is created to set up defaults that
        reference other fields or need computation.
        """
        # Set default motor channels if not specified
        if self.motor_ch_names is None:
            self.motor_ch_names = ["C3", "Cz", "C4"]
        
        # Set default FBCSP frequency bands if not specified
        # These cover the mu (8-12 Hz) and beta (13-30 Hz) rhythms
        # with overlapping bands to capture spectral transitions
        if self.fbcsp_bands is None:
            self.fbcsp_bands = [
                (8, 12),    # Classical mu rhythm
                (12, 16),   # Low beta
                (16, 20),   # Mid beta  
                (20, 26),   # High beta
                (26, 30)    # Very high beta
            ]


def ensure_output_dirs(config: Config) -> None:
    """
    Create output directories if they don't exist
    
    This ensures that all the output paths specified in the config
    have their parent directories created. Essential for saving
    models and results without path errors.
    
    Args:
        config: Configuration object with output paths
    """
    output_files = [config.out_model, config.out_meta, config.out_confmat]
    
    for filepath in output_files:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created output directory: {directory}")


def validate_config(config: Config) -> None:
    """
    Validate configuration parameters for common mistakes
    
    This catches configuration errors early before training starts,
    saving time and preventing confusing error messages later.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration parameters are invalid
    """
    # Validate sampling rate
    if config.fs <= 0:
        raise ValueError(f"Sampling rate must be positive, got {config.fs}")
    
    # Validate band-pass frequencies
    if config.bp_low >= config.bp_high:
        raise ValueError(f"Band-pass low ({config.bp_low}) must be < high ({config.bp_high})")
    
    if config.bp_high > config.fs / 2:
        raise ValueError(f"Band-pass high ({config.bp_high}) exceeds Nyquist ({config.fs/2})")
    
    # Validate CSP components
    if config.csp_components <= 0:
        raise ValueError(f"CSP components must be positive, got {config.csp_components}")
    
    # Validate classifier choice
    if config.classifier not in ["lda", "svm"]:
        raise ValueError(f"Classifier must be 'lda' or 'svm', got '{config.classifier}'")
    
    # Validate cross-validation folds
    if config.cv_folds < 2:
        raise ValueError(f"CV folds must be >= 2, got {config.cv_folds}")
    
    # Validate artifact thresholds
    if config.ptp_uV <= 0:
        raise ValueError(f"PTP threshold must be positive, got {config.ptp_uV}")
    
    if config.z_thresh <= 0:
        raise ValueError(f"Z-score threshold must be positive, got {config.z_thresh}")
    
    print("✓ Configuration validation passed")
