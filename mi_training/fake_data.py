"""
Synthetic Motor Imagery data generation

This module creates realistic synthetic EEG data for testing the training pipeline
without requiring real EEG recordings. The synthetic data mimics the key
characteristics of motor imagery signals.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np


def synthesize_mi_trials(
    n_trials_per_class: int = 60,
    fs: int = 250,
    trial_dur: float = 4.0,
    seed: int = 42,
    ch_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[Tuple[int, str]], int, List[str]]:
    """
    Generate synthetic motor imagery EEG data
    
    This function creates realistic synthetic EEG that mimics the key features
    of motor imagery signals:
    
    1. Event-Related Desynchronization (ERD) in mu rhythm (8-12 Hz)
    2. Spatial specificity: LEFT imagery affects C3, RIGHT imagery affects C4
    3. Realistic 1/f background noise
    4. Occasional artifacts to test rejection algorithms
    
    Why synthetic data?
    - Test pipeline without needing real EEG recordings
    - Known ground truth for validation
    - Controlled signal properties for debugging
    - Reproducible results for development
    
    Physiological basis:
    - Motor imagery causes ERD (power decrease) in sensorimotor rhythms
    - Left hand imagery primarily affects right motor cortex (C4)
    - Right hand imagery primarily affects left motor cortex (C3)
    - ERD typically occurs 1-3 seconds after imagery onset
    
    Args:
        n_trials_per_class: Number of LEFT and RIGHT trials to generate
        fs: Sampling frequency in Hz
        trial_dur: Duration of each trial in seconds
        seed: Random seed for reproducibility
        ch_names: Channel names to generate (default: standard 10-20 layout)
        
    Returns:
        Tuple matching data_io.load_csv format:
        - data: Continuous EEG [channels x samples]
        - markers: List of (sample_index, label) tuples
        - fs: Sampling frequency
        - ch_names: List of channel names
    """
    if ch_names is None:
        # Standard motor imagery montage based on 10-20 system
        ch_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'P7', 'P3', 'Pz', 'P4', 'P8',
            'O1', 'O2'
        ]
    
    np.random.seed(seed)
    logging.info(f"Generating synthetic MI data: {n_trials_per_class} trials per class")
    logging.info(f"Parameters: fs={fs}Hz, trial_dur={trial_dur}s, channels={len(ch_names)}")
    
    n_channels = len(ch_names)
    n_total_trials = n_trials_per_class * 2
    
    # Calculate timing parameters
    samples_per_trial = int(trial_dur * fs)
    inter_trial_interval = int(2.0 * fs)  # 2 second gap between trials
    total_samples = n_total_trials * (samples_per_trial + inter_trial_interval)
    
    # Initialize continuous data array
    data = np.zeros((n_channels, total_samples))
    markers = []
    
    # Find motor channel indices for ERD simulation
    motor_channels = {}
    for ch_name in ['C3', 'Cz', 'C4']:
        if ch_name in ch_names:
            motor_channels[ch_name] = ch_names.index(ch_name)
    
    logging.info(f"Motor channels found: {motor_channels}")
    
    # Generate trials
    current_sample = 0
    
    for trial_idx in range(n_total_trials):
        # Determine trial type (alternate LEFT/RIGHT)
        is_left_trial = trial_idx % 2 == 0
        label = "LEFT" if is_left_trial else "RIGHT"
        
        # Record marker at trial onset
        markers.append((current_sample, label))
        
        # Generate trial data
        trial_data = _generate_single_trial(
            n_channels=n_channels,
            samples_per_trial=samples_per_trial,
            fs=fs,
            is_left_trial=is_left_trial,
            motor_channels=motor_channels,
            ch_names=ch_names
        )
        
        # Insert trial into continuous data
        data[:, current_sample:current_sample + samples_per_trial] = trial_data
        current_sample += samples_per_trial
        
        # Add inter-trial interval with baseline activity
        if current_sample < total_samples - inter_trial_interval:
            baseline_data = _generate_baseline_activity(n_channels, inter_trial_interval, fs)
            data[:, current_sample:current_sample + inter_trial_interval] = baseline_data
            current_sample += inter_trial_interval
    
    # Trim data to actual length used
    data = data[:, :current_sample]
    
    logging.info(f"Generated {len(markers)} trials, total duration: {current_sample/fs:.1f}s")
    logging.info(f"Data shape: {data.shape}")
    
    return data, markers, fs, ch_names


def _generate_single_trial(
    n_channels: int,
    samples_per_trial: int,
    fs: int,
    is_left_trial: bool,
    motor_channels: dict,
    ch_names: List[str]
) -> np.ndarray:
    """
    Generate a single motor imagery trial
    
    This function creates the EEG activity for one trial, including:
    - Baseline activity (first 0.5s)
    - Motor imagery period with ERD (1.0-3.0s)
    - Recovery period (3.0-4.0s)
    
    Args:
        n_channels: Number of EEG channels
        samples_per_trial: Number of time samples in trial
        fs: Sampling frequency
        is_left_trial: True for LEFT imagery, False for RIGHT
        motor_channels: Dict mapping channel names to indices
        ch_names: List of channel names
        
    Returns:
        Trial EEG data [n_channels x samples_per_trial]
    """
    trial_data = np.zeros((n_channels, samples_per_trial))
    time_vector = np.arange(samples_per_trial) / fs
    
    # Generate base activity for all channels
    for ch_idx in range(n_channels):
        # 1/f background noise (pink noise approximation)
        white_noise = np.random.randn(samples_per_trial)
        # Simple 1/f approximation using cumulative sum
        pink_noise = np.cumsum(white_noise) / np.sqrt(samples_per_trial)
        pink_noise = pink_noise - np.mean(pink_noise)  # Remove DC
        
        # Scale to realistic EEG amplitudes (10-50 µV)
        base_amplitude = 20 + np.random.randn() * 5
        trial_data[ch_idx, :] = pink_noise * base_amplitude
        
        # Add alpha rhythm (8-12 Hz) - stronger in occipital
        if any(occ in ch_names[ch_idx] for occ in ['O1', 'O2', 'Pz']):
            alpha_freq = 10 + np.random.randn() * 1  # Individual alpha frequency
            alpha_phase = np.random.rand() * 2 * np.pi
            alpha_amplitude = 15 + np.random.randn() * 3
            alpha_wave = alpha_amplitude * np.sin(2 * np.pi * alpha_freq * time_vector + alpha_phase)
            trial_data[ch_idx, :] += alpha_wave
        
        # Add beta rhythm (15-25 Hz) - present in all channels
        beta_freq = 20 + np.random.randn() * 2
        beta_phase = np.random.rand() * 2 * np.pi
        beta_amplitude = 8 + np.random.randn() * 2
        beta_wave = beta_amplitude * np.sin(2 * np.pi * beta_freq * time_vector + beta_phase)
        trial_data[ch_idx, :] += beta_wave
    
    # Add motor imagery specific ERD
    _add_motor_imagery_erd(
        trial_data, time_vector, fs, is_left_trial, motor_channels, ch_names
    )
    
    # Occasionally add artifacts
    if np.random.rand() < 0.1:  # 10% chance of artifact
        _add_artifact(trial_data, time_vector, fs)
    
    return trial_data


def _add_motor_imagery_erd(
    trial_data: np.ndarray,
    time_vector: np.ndarray,
    fs: int,
    is_left_trial: bool,
    motor_channels: dict,
    ch_names: List[str]
) -> None:
    """
    Add Event-Related Desynchronization (ERD) to motor channels
    
    ERD is the key signature of motor imagery - a decrease in mu rhythm power
    over the contralateral motor cortex during imagery.
    
    Args:
        trial_data: Trial EEG data to modify in-place
        time_vector: Time points for the trial
        fs: Sampling frequency
        is_left_trial: True for LEFT imagery, False for RIGHT
        motor_channels: Dict of motor channel indices
        ch_names: List of channel names
    """
    # ERD timing parameters
    erd_onset = 1.0  # ERD starts 1s after trial onset
    erd_duration = 2.0  # ERD lasts 2 seconds
    erd_offset = erd_onset + erd_duration
    
    # Create ERD envelope (gradual onset and offset)
    erd_envelope = np.zeros_like(time_vector)
    erd_mask = (time_vector >= erd_onset) & (time_vector <= erd_offset)
    
    # Smooth ERD envelope with cosine ramps
    for i, t in enumerate(time_vector):
        if erd_onset <= t <= erd_offset:
            # Ramp up/down with cosine
            if t <= erd_onset + 0.5:  # Ramp up
                ramp_progress = (t - erd_onset) / 0.5
                erd_envelope[i] = 0.5 * (1 - np.cos(np.pi * ramp_progress))
            elif t >= erd_offset - 0.5:  # Ramp down
                ramp_progress = (erd_offset - t) / 0.5
                erd_envelope[i] = 0.5 * (1 - np.cos(np.pi * ramp_progress))
            else:  # Sustained ERD
                erd_envelope[i] = 1.0
    
    # Apply ERD to appropriate motor channel
    if is_left_trial and 'C4' in motor_channels:
        # Left imagery affects right motor cortex (C4)
        _apply_erd_to_channel(trial_data, motor_channels['C4'], time_vector, erd_envelope)
        logging.debug("Applied LEFT imagery ERD to C4")
        
    elif not is_left_trial and 'C3' in motor_channels:
        # Right imagery affects left motor cortex (C3)
        _apply_erd_to_channel(trial_data, motor_channels['C3'], time_vector, erd_envelope)
        logging.debug("Applied RIGHT imagery ERD to C3")
    
    # Add weaker ERD to central channel (Cz) for both conditions
    if 'Cz' in motor_channels:
        weak_envelope = erd_envelope * 0.3  # 30% of main ERD
        _apply_erd_to_channel(trial_data, motor_channels['Cz'], time_vector, weak_envelope)


def _apply_erd_to_channel(
    trial_data: np.ndarray,
    ch_idx: int,
    time_vector: np.ndarray,
    erd_envelope: np.ndarray
) -> None:
    """
    Apply ERD (power reduction) to a specific channel
    
    ERD is implemented as amplitude modulation of the mu rhythm (8-12 Hz).
    During ERD, the mu rhythm amplitude is reduced.
    
    Args:
        trial_data: Trial data to modify in-place
        ch_idx: Channel index to apply ERD to
        time_vector: Time vector
        erd_envelope: ERD strength envelope (0-1)
    """
    # Generate mu rhythm component
    mu_freq = 10 + np.random.randn() * 0.5  # Individual mu frequency
    mu_phase = np.random.rand() * 2 * np.pi
    mu_amplitude = 12 + np.random.randn() * 2
    
    mu_rhythm = mu_amplitude * np.sin(2 * np.pi * mu_freq * time_vector + mu_phase)
    
    # Apply ERD as amplitude reduction
    erd_strength = 0.6  # 60% power reduction during ERD
    amplitude_modulation = 1.0 - (erd_strength * erd_envelope)
    
    # Modulate the mu rhythm
    modulated_mu = mu_rhythm * amplitude_modulation
    
    # Add to existing signal (this replaces part of the baseline mu activity)
    trial_data[ch_idx, :] += modulated_mu


def _add_artifact(trial_data: np.ndarray, time_vector: np.ndarray, fs: int) -> None:
    """
    Add realistic artifacts to trial data
    
    Artifacts are common in EEG and the pipeline should handle them robustly.
    This function adds typical artifacts like eye blinks and muscle activity.
    
    Args:
        trial_data: Trial data to modify in-place
        time_vector: Time vector
        fs: Sampling frequency
    """
    artifact_type = np.random.choice(['blink', 'muscle', 'electrode'])
    
    if artifact_type == 'blink':
        # Eye blink artifact - affects frontal channels
        blink_time = np.random.uniform(0.5, 3.5)  # Random time during trial
        blink_duration = 0.2  # 200ms blink
        
        # Gaussian blink shape
        blink_center_idx = int(blink_time * fs)
        blink_width = int(blink_duration * fs / 2)
        
        blink_indices = np.arange(
            max(0, blink_center_idx - blink_width),
            min(len(time_vector), blink_center_idx + blink_width)
        )
        
        if len(blink_indices) > 0:
            # Gaussian blink artifact
            blink_times = time_vector[blink_indices] - blink_time
            blink_amplitude = 100 + np.random.randn() * 20  # Large amplitude
            blink_artifact = blink_amplitude * np.exp(-0.5 * (blink_times / 0.05)**2)
            
            # Apply to frontal channels (first few channels typically)
            for ch_idx in range(min(4, trial_data.shape[0])):
                trial_data[ch_idx, blink_indices] += blink_artifact
    
    elif artifact_type == 'muscle':
        # Muscle artifact - high frequency, random channels
        muscle_start = np.random.uniform(0, 3)
        muscle_duration = np.random.uniform(0.5, 1.5)
        muscle_end = min(muscle_start + muscle_duration, time_vector[-1])
        
        muscle_mask = (time_vector >= muscle_start) & (time_vector <= muscle_end)
        muscle_indices = np.where(muscle_mask)[0]
        
        if len(muscle_indices) > 0:
            # High-frequency muscle activity
            muscle_freq = np.random.uniform(40, 80)  # High frequency
            muscle_amplitude = 50 + np.random.randn() * 15
            muscle_noise = muscle_amplitude * np.random.randn(len(muscle_indices))
            
            # Apply to random subset of channels
            affected_channels = np.random.choice(
                trial_data.shape[0], 
                size=min(3, trial_data.shape[0]), 
                replace=False
            )
            
            for ch_idx in affected_channels:
                trial_data[ch_idx, muscle_indices] += muscle_noise
    
    elif artifact_type == 'electrode':
        # Electrode movement - sudden DC shift
        shift_time = np.random.uniform(0.5, 3.5)
        shift_idx = int(shift_time * fs)
        
        if shift_idx < trial_data.shape[1]:
            # Sudden DC shift in one channel
            affected_channel = np.random.randint(0, trial_data.shape[0])
            shift_amplitude = np.random.uniform(30, 80) * np.random.choice([-1, 1])
            
            # Apply shift from this point onward
            trial_data[affected_channel, shift_idx:] += shift_amplitude


def _generate_baseline_activity(n_channels: int, n_samples: int, fs: int) -> np.ndarray:
    """
    Generate baseline EEG activity for inter-trial intervals
    
    This creates realistic resting-state EEG without motor imagery components.
    
    Args:
        n_channels: Number of EEG channels
        n_samples: Number of time samples
        fs: Sampling frequency
        
    Returns:
        Baseline EEG data [n_channels x n_samples]
    """
    baseline_data = np.zeros((n_channels, n_samples))
    time_vector = np.arange(n_samples) / fs
    
    for ch_idx in range(n_channels):
        # Pink noise background
        white_noise = np.random.randn(n_samples)
        pink_noise = np.cumsum(white_noise) / np.sqrt(n_samples)
        pink_noise = pink_noise - np.mean(pink_noise)
        
        # Scale to EEG amplitudes
        baseline_amplitude = 15 + np.random.randn() * 3
        baseline_data[ch_idx, :] = pink_noise * baseline_amplitude
        
        # Add spontaneous alpha rhythm
        alpha_freq = 10 + np.random.randn() * 1
        alpha_amplitude = 10 + np.random.randn() * 2
        alpha_phase = np.random.rand() * 2 * np.pi
        alpha_wave = alpha_amplitude * np.sin(2 * np.pi * alpha_freq * time_vector + alpha_phase)
        baseline_data[ch_idx, :] += alpha_wave
    
    return baseline_data
