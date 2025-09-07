"""
Epoching functions for Motor Imagery data

This module handles the extraction of time-locked epochs around motor imagery events.
Proper epoching is crucial because it defines the time windows used for classification.
"""

import logging
from typing import Tuple, Optional
import numpy as np


def build_epochs(
    data: np.ndarray,
    markers: list,
    fs: int,
    tmin: float = -0.2,
    tmax: float = 4.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract epochs around motor imagery markers
    
    This function creates time-locked epochs around LEFT and RIGHT motor imagery
    events. The epoch timing is critical for motor imagery classification:
    
    - Pre-stimulus period (tmin < 0): Captures baseline brain activity
    - Motor imagery period (0 to ~3-4s): When the actual imagery occurs
    - Post-imagery period: May contain movement-related activity
    
    Why these default timings?
    - tmin = -0.2s: Short baseline for artifact detection and normalization
    - tmax = 4.0s: Typical motor imagery trial duration in experiments
    - Total 4.2s epochs capture the full imagery process
    
    Args:
        data: Continuous EEG data [channels x samples]
        markers: List of (sample_index, label) tuples from data loading
        fs: Sampling frequency in Hz
        tmin: Start time relative to marker (seconds, negative = before)
        tmax: End time relative to marker (seconds, positive = after)
        
    Returns:
        Tuple of:
        - epochs: Epoched data [n_epochs x n_channels x n_samples]
        - labels: Numeric labels [n_epochs] (0=LEFT, 1=RIGHT)
        - time_vector: Time points in seconds [n_samples]
        
    Raises:
        ValueError: If epoch parameters are invalid or no valid epochs found
    """
    if tmin >= tmax:
        raise ValueError(f"tmin ({tmin}) must be < tmax ({tmax})")
    
    if len(markers) == 0:
        raise ValueError("No markers provided for epoching")
    
    logging.info(f"Building epochs from {len(markers)} markers")
    logging.info(f"Epoch window: {tmin:.2f} to {tmax:.2f} seconds ({tmax-tmin:.2f}s duration)")
    
    # Calculate epoch parameters
    n_samples_epoch = int((tmax - tmin) * fs)
    samples_before = int(-tmin * fs)  # Number of samples before marker (positive)
    samples_after = int(tmax * fs)    # Number of samples after marker
    
    logging.info(f"Epoch length: {n_samples_epoch} samples ({n_samples_epoch/fs:.2f}s)")
    
    # Create time vector for epochs
    time_vector = np.linspace(tmin, tmax, n_samples_epoch)
    
    # Prepare lists for valid epochs and labels
    valid_epochs = []
    valid_labels = []
    n_channels = data.shape[0]
    n_samples_total = data.shape[1]
    
    # Process each marker
    for marker_sample, marker_label in markers:
        # Calculate epoch boundaries
        epoch_start = marker_sample - samples_before
        epoch_end = marker_sample + samples_after
        
        # Check if epoch is within data bounds
        if epoch_start < 0:
            logging.warning(f"Epoch starting at sample {epoch_start} is before data start, skipping")
            continue
        
        if epoch_end > n_samples_total:
            logging.warning(f"Epoch ending at sample {epoch_end} exceeds data length {n_samples_total}, skipping")
            continue
        
        # Extract epoch
        epoch_data = data[:, epoch_start:epoch_end]
        
        # Ensure epoch has correct length (handle rounding errors)
        if epoch_data.shape[1] != n_samples_epoch:
            # Pad or truncate to exact length
            if epoch_data.shape[1] < n_samples_epoch:
                # Pad with zeros
                padding = np.zeros((n_channels, n_samples_epoch - epoch_data.shape[1]))
                epoch_data = np.concatenate([epoch_data, padding], axis=1)
            else:
                # Truncate
                epoch_data = epoch_data[:, :n_samples_epoch]
        
        valid_epochs.append(epoch_data)
        
        # Convert string labels to numeric
        if marker_label.upper() == "LEFT":
            valid_labels.append(0)
        elif marker_label.upper() == "RIGHT":
            valid_labels.append(1)
        else:
            logging.warning(f"Unknown marker label: {marker_label}, skipping")
            valid_epochs.pop()  # Remove the epoch we just added
            continue
    
    if len(valid_epochs) == 0:
        raise ValueError("No valid epochs could be extracted. Check marker timing and data length.")
    
    # Convert to numpy arrays
    epochs = np.array(valid_epochs)  # [n_epochs x n_channels x n_samples]
    labels = np.array(valid_labels)  # [n_epochs]
    
    # Log epoch statistics
    n_left = np.sum(labels == 0)
    n_right = np.sum(labels == 1)
    logging.info(f"Extracted {len(epochs)} valid epochs:")
    logging.info(f"  LEFT: {n_left} epochs ({n_left/len(epochs)*100:.1f}%)")
    logging.info(f"  RIGHT: {n_right} epochs ({n_right/len(epochs)*100:.1f}%)")
    logging.info(f"  Epoch shape: {epochs.shape}")
    
    return epochs, labels, time_vector


def baseline_correct(
    epochs: np.ndarray,
    fs: int,
    tmin: float,
    tmax: float,
    baseline: Tuple[float, float] = (-0.2, 0.0)
) -> np.ndarray:
    """
    Apply baseline correction to epochs
    
    Baseline correction removes slow drifts and DC offsets by subtracting
    the mean activity during a reference period (usually pre-stimulus).
    This is essential for motor imagery because:
    
    1. It removes slow drifts that can bias classification
    2. It normalizes starting conditions across trials
    3. It highlights changes relative to the resting state
    
    Mathematical operation:
    corrected_signal = original_signal - mean(baseline_period)
    
    Args:
        epochs: Epoched data [n_epochs x n_channels x n_samples]
        fs: Sampling frequency in Hz
        tmin: Epoch start time (for time vector calculation)
        tmax: Epoch end time (for time vector calculation)
        baseline: Tuple of (start, end) times for baseline period in seconds
        
    Returns:
        Baseline-corrected epochs with same shape as input
        
    Raises:
        ValueError: If baseline period is invalid or outside epoch bounds
    """
    baseline_start, baseline_end = baseline
    
    if baseline_start >= baseline_end:
        raise ValueError(f"Baseline start ({baseline_start}) must be < end ({baseline_end})")
    
    if baseline_start < tmin or baseline_end > tmax:
        raise ValueError(f"Baseline period {baseline} must be within epoch bounds [{tmin}, {tmax}]")
    
    logging.info(f"Applying baseline correction using period {baseline_start:.2f} to {baseline_end:.2f}s")
    
    n_epochs, n_channels, n_samples = epochs.shape
    
    # Create time vector for the epoch
    time_vector = np.linspace(tmin, tmax, n_samples)
    
    # Find baseline sample indices
    baseline_mask = (time_vector >= baseline_start) & (time_vector <= baseline_end)
    baseline_samples = np.where(baseline_mask)[0]
    
    if len(baseline_samples) == 0:
        raise ValueError(f"No samples found in baseline period {baseline}")
    
    logging.info(f"Using {len(baseline_samples)} samples for baseline correction")
    
    # Apply baseline correction
    corrected_epochs = epochs.copy()
    
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            # Calculate baseline mean for this epoch and channel
            baseline_mean = np.mean(epochs[epoch_idx, ch_idx, baseline_samples])
            
            # Subtract baseline from entire epoch
            corrected_epochs[epoch_idx, ch_idx, :] -= baseline_mean
    
    logging.info("Baseline correction complete")
    return corrected_epochs


def balance_classes(
    epochs: np.ndarray,
    labels: np.ndarray,
    method: str = "undersample",
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance the number of epochs between LEFT and RIGHT classes
    
    Class imbalance can bias machine learning classifiers toward the majority class.
    This function provides two strategies to address imbalance:
    
    1. Undersampling: Randomly remove epochs from the majority class
    2. Oversampling: Randomly duplicate epochs from the minority class
    
    Why balance classes?
    - Prevents classifier bias toward majority class
    - Improves generalization to both classes
    - Standard practice in BCI research
    
    Args:
        epochs: Epoched data [n_epochs x n_channels x n_samples]
        labels: Class labels [n_epochs]
        method: "undersample" or "oversample"
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (balanced_epochs, balanced_labels)
        
    Raises:
        ValueError: If method is invalid or no epochs for a class
    """
    if method not in ["undersample", "oversample"]:
        raise ValueError(f"Method must be 'undersample' or 'oversample', got '{method}'")
    
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Count samples per class
    n_left = np.sum(labels == 0)
    n_right = np.sum(labels == 1)
    
    logging.info(f"Original class distribution: LEFT={n_left}, RIGHT={n_right}")
    
    if n_left == 0 or n_right == 0:
        raise ValueError("Cannot balance classes: one class has no samples")
    
    if n_left == n_right:
        logging.info("Classes already balanced, no action needed")
        return epochs, labels
    
    # Get indices for each class
    left_indices = np.where(labels == 0)[0]
    right_indices = np.where(labels == 1)[0]
    
    if method == "undersample":
        # Reduce majority class to match minority class
        target_size = min(n_left, n_right)
        
        if n_left > target_size:
            # Randomly select subset of LEFT epochs
            selected_left = np.random.choice(left_indices, size=target_size, replace=False)
            left_indices = selected_left
        
        if n_right > target_size:
            # Randomly select subset of RIGHT epochs
            selected_right = np.random.choice(right_indices, size=target_size, replace=False)
            right_indices = selected_right
        
        logging.info(f"Undersampled to {target_size} epochs per class")
    
    elif method == "oversample":
        # Increase minority class to match majority class
        target_size = max(n_left, n_right)
        
        if n_left < target_size:
            # Randomly duplicate LEFT epochs
            additional_needed = target_size - n_left
            additional_left = np.random.choice(left_indices, size=additional_needed, replace=True)
            left_indices = np.concatenate([left_indices, additional_left])
        
        if n_right < target_size:
            # Randomly duplicate RIGHT epochs
            additional_needed = target_size - n_right
            additional_right = np.random.choice(right_indices, size=additional_needed, replace=True)
            right_indices = np.concatenate([right_indices, additional_right])
        
        logging.info(f"Oversampled to {target_size} epochs per class")
    
    # Combine balanced indices
    balanced_indices = np.concatenate([left_indices, right_indices])
    
    # Shuffle to randomize order
    np.random.shuffle(balanced_indices)
    
    # Extract balanced data
    balanced_epochs = epochs[balanced_indices]
    balanced_labels = labels[balanced_indices]
    
    # Verify balance
    final_left = np.sum(balanced_labels == 0)
    final_right = np.sum(balanced_labels == 1)
    logging.info(f"Final balanced distribution: LEFT={final_left}, RIGHT={final_right}")
    
    return balanced_epochs, balanced_labels
