"""
EEG preprocessing functions for Motor Imagery training

This module handles the essential preprocessing steps that prepare raw EEG data
for feature extraction. Good preprocessing is critical for motor imagery classification
because we need to isolate the relevant frequency bands and remove artifacts.
"""

import logging
from typing import Optional, Tuple
import numpy as np
from scipy import signal
import mne

# Configure MNE to reduce verbose output
mne.set_log_level('WARNING')


def apply_filters(
    data: np.ndarray,
    fs: int,
    notch: Optional[int] = None,
    bp_low: Optional[float] = None,
    bp_high: Optional[float] = None,
    resample: Optional[int] = None
) -> np.ndarray:
    """
    Apply digital filters to EEG data
    
    This function applies the standard preprocessing pipeline for motor imagery:
    1. Notch filter to remove power line interference (50/60 Hz)
    2. Band-pass filter to focus on mu/beta rhythms (typically 8-30 Hz)
    3. Optional resampling to reduce computational load
    
    Why these filters?
    - Notch: Power lines create strong artifacts at exactly 50 or 60 Hz
    - Band-pass: Motor imagery primarily affects mu (8-12 Hz) and beta (13-30 Hz) rhythms
    - Resampling: Lower sampling rates reduce file sizes and computation time
    
    Technical choices:
    - Zero-phase filtering (filtfilt) to avoid phase distortion
    - Butterworth filters for smooth frequency response
    - FIR filters for notch to avoid ringing artifacts
    
    Args:
        data: EEG data [channels x samples]
        fs: Sampling frequency in Hz
        notch: Notch filter frequency (50 or 60 Hz), None to skip
        bp_low: Band-pass low cutoff in Hz, None to skip
        bp_high: Band-pass high cutoff in Hz, None to skip
        resample: Target sampling rate for downsampling, None to skip
        
    Returns:
        Filtered EEG data with same shape as input (unless resampled)
    """
    filtered_data = data.copy()
    
    logging.info(f"Applying filters to data shape {data.shape}")
    
    # Apply notch filter to remove power line interference
    if notch is not None:
        logging.info(f"Applying notch filter at {notch} Hz")
        
        # Design notch filter using scipy
        # Q factor of 30 gives a narrow notch (~2 Hz wide at -3dB)
        nyquist = fs / 2
        w0 = notch / nyquist
        b, a = signal.iirnotch(w0, Q=30)
        
        # Apply to each channel separately
        for ch in range(filtered_data.shape[0]):
            filtered_data[ch, :] = signal.filtfilt(b, a, filtered_data[ch, :])
    
    # Apply band-pass filter to focus on motor imagery frequencies
    if bp_low is not None and bp_high is not None:
        logging.info(f"Applying band-pass filter {bp_low}-{bp_high} Hz")
        
        # Validate frequency range
        nyquist = fs / 2
        if bp_high >= nyquist:
            raise ValueError(f"Band-pass high ({bp_high}) must be < Nyquist ({nyquist})")
        
        if bp_low >= bp_high:
            raise ValueError(f"Band-pass low ({bp_low}) must be < high ({bp_high})")
        
        # Design 4th-order Butterworth band-pass filter
        # Butterworth filters have maximally flat passband response
        low = bp_low / nyquist
        high = bp_high / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply to each channel separately
        for ch in range(filtered_data.shape[0]):
            filtered_data[ch, :] = signal.filtfilt(b, a, filtered_data[ch, :])
    
    # Resample to reduce data size and computational load
    if resample is not None and resample != fs:
        if resample > fs:
            logging.warning(f"Target sampling rate ({resample}) > original ({fs}), no resampling applied")
        else:
            logging.info(f"Resampling from {fs} Hz to {resample} Hz")
            
            # Calculate resampling parameters
            downsample_factor = fs // resample
            
            if fs % resample == 0:
                # Simple integer downsampling
                filtered_data = filtered_data[:, ::downsample_factor]
            else:
                # Use scipy's resample for non-integer factors
                n_samples_new = int(filtered_data.shape[1] * resample / fs)
                resampled = np.zeros((filtered_data.shape[0], n_samples_new))
                
                for ch in range(filtered_data.shape[0]):
                    resampled[ch, :] = signal.resample(filtered_data[ch, :], n_samples_new)
                
                filtered_data = resampled
    
    logging.info(f"Filtering complete. Output shape: {filtered_data.shape}")
    return filtered_data


def reject_artifacts_ptp(
    epochs: np.ndarray,
    labels: np.ndarray,
    fs: int,
    ptp_uV: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reject epochs based on peak-to-peak amplitude threshold
    
    Peak-to-peak (PTP) rejection is the most common artifact rejection method.
    It catches large amplitude artifacts like eye blinks, muscle activity,
    and electrode movement. The threshold is typically 100-200 µV.
    
    Why PTP rejection?
    - Simple and effective for catching obvious artifacts
    - Physiologically meaningful (normal EEG is typically < 100 µV)
    - Fast computation compared to more sophisticated methods
    
    Args:
        epochs: Epoch data [n_epochs x n_channels x n_samples]
        labels: Epoch labels [n_epochs]
        fs: Sampling frequency (for logging only)
        ptp_uV: Peak-to-peak threshold in microvolts
        
    Returns:
        Tuple of (clean_epochs, clean_labels, kept_mask)
    """
    n_epochs = epochs.shape[0]
    logging.info(f"Applying PTP artifact rejection (threshold: {ptp_uV} µV) to {n_epochs} epochs")
    
    # Calculate peak-to-peak amplitude for each epoch and channel
    ptp_amplitudes = np.ptp(epochs, axis=2)  # [n_epochs x n_channels]
    
    # Find epochs where ANY channel exceeds the threshold
    max_ptp_per_epoch = np.max(ptp_amplitudes, axis=1)  # [n_epochs]
    kept_mask = max_ptp_per_epoch <= ptp_uV
    
    # Apply the mask
    clean_epochs = epochs[kept_mask]
    clean_labels = labels[kept_mask]
    
    # Count rejections by class
    n_rejected = n_epochs - np.sum(kept_mask)
    n_rejected_left = np.sum((labels == 0) & ~kept_mask)  # Assuming LEFT=0
    n_rejected_right = np.sum((labels == 1) & ~kept_mask)  # Assuming RIGHT=1
    
    logging.info(f"PTP rejection: {n_rejected}/{n_epochs} epochs rejected ({n_rejected/n_epochs*100:.1f}%)")
    logging.info(f"  LEFT: {n_rejected_left} rejected, RIGHT: {n_rejected_right} rejected")
    logging.info(f"Remaining epochs: {clean_epochs.shape[0]}")
    
    if clean_epochs.shape[0] == 0:
        raise ValueError(f"All epochs rejected with PTP threshold {ptp_uV} µV. Consider increasing threshold.")
    
    return clean_epochs, clean_labels, kept_mask


def reject_artifacts_zscore(
    epochs: np.ndarray,
    labels: np.ndarray,
    z_thresh: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reject epochs based on z-score statistical outlier detection
    
    Z-score rejection identifies epochs that are statistical outliers
    compared to the overall distribution. It's more sophisticated than
    PTP rejection and can catch subtler artifacts.
    
    Why z-score rejection?
    - Adapts to the data's natural amplitude distribution
    - Can catch artifacts that aren't just high amplitude
    - Complementary to PTP rejection for comprehensive cleaning
    
    The method:
    1. Calculate RMS amplitude for each epoch/channel
    2. Compute z-scores relative to the overall distribution
    3. Reject epochs where any channel has |z-score| > threshold
    
    Args:
        epochs: Epoch data [n_epochs x n_channels x n_samples]
        labels: Epoch labels [n_epochs]
        z_thresh: Z-score threshold (typically 3-5)
        
    Returns:
        Tuple of (clean_epochs, clean_labels, kept_mask)
    """
    n_epochs = epochs.shape[0]
    logging.info(f"Applying z-score artifact rejection (threshold: {z_thresh}) to {n_epochs} epochs")
    
    # Calculate RMS amplitude for each epoch and channel
    rms_amplitudes = np.sqrt(np.mean(epochs**2, axis=2))  # [n_epochs x n_channels]
    
    # Calculate z-scores for each channel separately
    # This accounts for different baseline amplitudes across channels
    z_scores = np.zeros_like(rms_amplitudes)
    for ch in range(rms_amplitudes.shape[1]):
        ch_rms = rms_amplitudes[:, ch]
        z_scores[:, ch] = np.abs((ch_rms - np.mean(ch_rms)) / (np.std(ch_rms) + 1e-10))
    
    # Find epochs where ANY channel exceeds the z-score threshold
    max_zscore_per_epoch = np.max(z_scores, axis=1)  # [n_epochs]
    kept_mask = max_zscore_per_epoch <= z_thresh
    
    # Apply the mask
    clean_epochs = epochs[kept_mask]
    clean_labels = labels[kept_mask]
    
    # Count rejections by class
    n_rejected = n_epochs - np.sum(kept_mask)
    n_rejected_left = np.sum((labels == 0) & ~kept_mask)  # Assuming LEFT=0
    n_rejected_right = np.sum((labels == 1) & ~kept_mask)  # Assuming RIGHT=1
    
    logging.info(f"Z-score rejection: {n_rejected}/{n_epochs} epochs rejected ({n_rejected/n_epochs*100:.1f}%)")
    logging.info(f"  LEFT: {n_rejected_left} rejected, RIGHT: {n_rejected_right} rejected")
    logging.info(f"Remaining epochs: {clean_epochs.shape[0]}")
    
    if clean_epochs.shape[0] == 0:
        raise ValueError(f"All epochs rejected with z-score threshold {z_thresh}. Consider increasing threshold.")
    
    return clean_epochs, clean_labels, kept_mask


def apply_car_filter(epochs: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR) spatial filter
    
    CAR filtering subtracts the average of all channels from each channel.
    This removes common-mode artifacts (like power line interference that
    affects all channels equally) while preserving local brain activity.
    
    Why CAR?
    - Reduces common artifacts affecting all channels
    - Improves spatial specificity of signals
    - Standard preprocessing for many BCI applications
    - Simple and computationally efficient
    
    Mathematical operation:
    CAR_signal[ch] = original_signal[ch] - mean(all_channels)
    
    Args:
        epochs: Epoch data [n_epochs x n_channels x n_samples]
        
    Returns:
        CAR-filtered epochs with same shape
    """
    logging.info(f"Applying Common Average Reference (CAR) filter to {epochs.shape[0]} epochs")
    
    # Calculate the average across all channels for each time point
    # Shape: [n_epochs x 1 x n_samples]
    channel_average = np.mean(epochs, axis=1, keepdims=True)
    
    # Subtract the average from each channel
    car_epochs = epochs - channel_average
    
    logging.info("CAR filtering complete")
    return car_epochs


def apply_laplacian_filter(epochs: np.ndarray, channel_names: list) -> np.ndarray:
    """
    Apply Laplacian spatial filter to motor channels
    
    The Laplacian filter enhances local activity by subtracting the average
    of surrounding channels. For motor imagery, we typically apply it to
    C3 and C4 using their immediate neighbors.
    
    Why Laplacian filtering?
    - Enhances local brain activity
    - Reduces volume conduction from distant sources
    - Improves signal-to-noise ratio for focal activities
    - Particularly effective for sensorimotor rhythms
    
    Standard motor Laplacian:
    C3_Lap = C3 - (FC1 + FC5 + CP1 + CP5) / 4
    C4_Lap = C4 - (FC2 + FC6 + CP2 + CP6) / 4
    
    Args:
        epochs: Epoch data [n_epochs x n_channels x n_samples]
        channel_names: List of channel names corresponding to the channel dimension
        
    Returns:
        Laplacian-filtered epochs (only motor channels are modified)
    """
    logging.info("Applying Laplacian spatial filter to motor channels")
    
    # Define Laplacian neighborhoods for common motor channels
    laplacian_neighborhoods = {
        'C3': ['FC1', 'FC5', 'CP1', 'CP5'],
        'C4': ['FC2', 'FC6', 'CP2', 'CP6'],
        'Cz': ['FCz', 'C1', 'C2', 'CPz']
    }
    
    filtered_epochs = epochs.copy()
    
    for center_ch, neighbor_names in laplacian_neighborhoods.items():
        try:
            # Find indices for center and neighbor channels
            center_idx = channel_names.index(center_ch)
            neighbor_indices = []
            
            for neighbor in neighbor_names:
                try:
                    neighbor_indices.append(channel_names.index(neighbor))
                except ValueError:
                    continue  # Skip missing neighbors
            
            if len(neighbor_indices) >= 2:  # Need at least 2 neighbors for meaningful Laplacian
                # Calculate Laplacian: center - mean(neighbors)
                neighbor_average = np.mean(epochs[:, neighbor_indices, :], axis=1)
                filtered_epochs[:, center_idx, :] = epochs[:, center_idx, :] - neighbor_average
                
                logging.info(f"Applied Laplacian to {center_ch} using {len(neighbor_indices)} neighbors")
            else:
                logging.warning(f"Insufficient neighbors for {center_ch} Laplacian filter")
                
        except ValueError:
            logging.info(f"Channel {center_ch} not found, skipping Laplacian filter")
    
    return filtered_epochs
