#!/usr/bin/env python3
"""
EEG Bridge - Production-quality BCI processing for Unity integration

This script connects to OpenBCI Cyton+Daisy or LSL streams, processes EEG signals,
detects mental states (relax/focus/motor imagery), and streams results to Unity via UDP.

Author: AI Assistant (Claude)
Python: 3.10+
Dependencies: brainflow, numpy, scipy, scikit-learn, joblib, mne, pylsl
"""

import argparse
import json
import logging
import os
import socket
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from threading import Event
import signal

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import zscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Optional imports with fallbacks
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    logging.warning("BrainFlow not available - use --source lsl or --fake instead")

try:
    import pylsl
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    logging.warning("pylsl not available - BrainFlow and --fake modes only")

try:
    import mne
    from mne.decoding import CSP
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    logging.warning("MNE not available - using basic CSP implementation")

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURATION SECTION - User should edit these values for their setup
# ============================================================================

# Hardware Configuration
SERIAL_PORT = "COM3"              # TODO: Change to your actual port (Windows: COMx, Linux: /dev/ttyUSBx)
FS_EXPECTED = 250                 # Cyton+Daisy sampling rate (Hz)
NOTCH_HZ = 60                     # Power line frequency (50 Hz for EU, 60 Hz for US)
BANDPASS = (1.0, 45.0)           # Band-pass filter range (Hz)

# Processing Configuration
WINDOW_SEC = 1.0                  # Analysis window duration (seconds)
WINDOW_OVERLAP = 0.5              # Window overlap fraction (0.5 = 50%)
EMA_TAU_SEC = 4.0                 # Exponential moving average time constant
HOLD_MS = 800                     # Minimum hold time for state changes (ms)

# Communication Configuration
UDP_HOST = "127.0.0.1"            # Unity UDP host
UDP_PORT = 5005                   # Unity UDP port

# File Paths
PROFILE_DIR = "profiles"          # User calibration profiles directory
MODEL_DIR = "models"              # ML models directory
MI_MODEL_PATH = "models/mi_csp_lda.joblib"  # Motor imagery model file

# Channel Mapping - TODO: Update these indices based on your actual BrainFlow setup
# To find correct indices, run: python -c "from brainflow import *; print(BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD))"
CHANNELS = {
    # Occipital channels for alpha (relaxation)
    "O1": 0,  # TODO: Replace with actual BrainFlow channel index
    "O2": 1,  # TODO: Replace with actual BrainFlow channel index
    "Pz": 2,  # TODO: Replace with actual BrainFlow channel index
    
    # Frontal channels for beta (focus)
    "Fz": 3,  # TODO: Replace with actual BrainFlow channel index
    "F3": 4,  # TODO: Replace with actual BrainFlow channel index
    "F4": 5,  # TODO: Replace with actual BrainFlow channel index
    
    # Central channels for motor imagery
    "C3": 6,  # TODO: Replace with actual BrainFlow channel index
    "C4": 7,  # TODO: Replace with actual BrainFlow channel index
    "Cz": 8,  # TODO: Replace with actual BrainFlow channel index
}

# Frequency Bands (Hz)
FREQ_BANDS = {
    "theta": (4, 7),     # Theta rhythm
    "alpha": (8, 12),    # Alpha rhythm (relaxation, eyes closed)
    "beta": (13, 30),    # Beta rhythm (focus, attention)
    "mu": (8, 12),       # Mu rhythm (motor imagery, same as alpha but over motor cortex)
}

# Electrode Groups for Different Analyses
ELECTRODE_GROUPS = {
    "alpha": ["O1", "O2", "Pz"],      # Occipital for relaxation
    "beta": ["Fz", "F3", "F4"],       # Frontal for focus
    "mi": ["C3", "C4", "Cz"],         # Central for motor imagery
}

# Artifact Detection Thresholds
ARTIFACT_ZSCORE_THRESH = 5.0      # Z-score threshold for artifact detection
ARTIFACT_CHANNELS = ["F3", "F4", "Fz"]  # Frontal channels for blink detection

# ============================================================================
# CORE DATA CLASSES AND PROCESSING COMPONENTS
# ============================================================================

@dataclass
class EEGWindow:
    """Container for a single EEG analysis window"""
    data: np.ndarray          # Shape: (n_channels, n_samples)
    timestamp: float          # Unix timestamp
    fs: float                 # Sampling frequency
    is_artifact: bool = False # Artifact flag

@dataclass
class BandPowers:
    """Container for frequency band powers"""
    theta: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    mu_c3: float = 0.0
    mu_c4: float = 0.0

@dataclass
class MentalState:
    """Container for detected mental state"""
    timestamp: float
    relax_index: float
    focus_index: float
    state: str  # "relax", "focus", "neutral"
    mi_prob_left: float
    mi_prob_right: float
    band_powers: BandPowers
    artifact: bool = False

class BrainSource:
    """
    Unified interface for EEG data acquisition from BrainFlow or LSL
    
    This class handles the complexity of different data sources and provides
    a consistent interface for the main processing loop.
    """
    
    def __init__(self, source_type: str = "brainflow", serial_port: str = SERIAL_PORT, 
                 lsl_stream_name: str = "EEG"):
        self.source_type = source_type
        self.serial_port = serial_port
        self.lsl_stream_name = lsl_stream_name
        self.board = None
        self.lsl_inlet = None
        self.fs = FS_EXPECTED
        self.eeg_channels = []
        self.is_connected = False
        
    def connect(self) -> bool:
        """
        Establish connection to the EEG data source
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.source_type == "brainflow":
                return self._connect_brainflow()
            elif self.source_type == "lsl":
                return self._connect_lsl()
            else:
                logging.error(f"Unknown source type: {self.source_type}")
                return False
        except Exception as e:
            logging.error(f"Failed to connect to {self.source_type}: {e}")
            return False
    
    def _connect_brainflow(self) -> bool:
        """Connect to OpenBCI via BrainFlow"""
        if not BRAINFLOW_AVAILABLE:
            logging.error("BrainFlow not available. Install with: pip install brainflow")
            return False
            
        try:
            # Configure BrainFlow parameters
            params = BrainFlowInputParams()
            params.serial_port = self.serial_port
            
            # Use Cyton+Daisy board (16 channels)
            board_id = BoardIds.CYTON_DAISY_BOARD
            self.board = BoardShim(board_id, params)
            
            # Get board info
            self.eeg_channels = BoardShim.get_eeg_channels(board_id)
            self.fs = BoardShim.get_sampling_rate(board_id)
            
            logging.info(f"BrainFlow EEG channels: {self.eeg_channels}")
            logging.info(f"Sampling rate: {self.fs} Hz")
            
            # Prepare and start session
            self.board.prepare_session()
            self.board.start_stream()
            
            self.is_connected = True
            logging.info(f"Connected to OpenBCI on {self.serial_port}")
            return True
            
        except Exception as e:
            logging.error(f"BrainFlow connection failed: {e}")
            logging.error("Hint: Check COM port, ensure board is on, and no other software is using it")
            return False
    
    def _connect_lsl(self) -> bool:
        """Connect to LSL EEG stream"""
        if not LSL_AVAILABLE:
            logging.error("pylsl not available. Install with: pip install pylsl")
            return False
            
        try:
            # Look for EEG streams
            logging.info(f"Looking for LSL stream: {self.lsl_stream_name}")
            streams = pylsl.resolve_stream('name', self.lsl_stream_name)
            
            if not streams:
                # Try generic EEG type
                streams = pylsl.resolve_stream('type', 'EEG')
                
            if not streams:
                logging.error("No LSL EEG streams found")
                return False
                
            # Connect to first available stream
            stream_info = streams[0]
            self.lsl_inlet = pylsl.StreamInlet(stream_info)
            
            # Get stream info
            self.fs = stream_info.nominal_srate()
            n_channels = stream_info.channel_count()
            self.eeg_channels = list(range(n_channels))
            
            logging.info(f"Connected to LSL stream: {stream_info.name()}")
            logging.info(f"Channels: {n_channels}, Sample rate: {self.fs} Hz")
            
            self.is_connected = True
            return True
            
        except Exception as e:
            logging.error(f"LSL connection failed: {e}")
            return False
    
    def get_data(self, duration_sec: float) -> Optional[np.ndarray]:
        """
        Get EEG data for specified duration
        
        Args:
            duration_sec: Duration of data to retrieve (seconds)
            
        Returns:
            np.ndarray: EEG data (channels x samples) or None if failed
        """
        if not self.is_connected:
            return None
            
        try:
            if self.source_type == "brainflow":
                return self._get_brainflow_data(duration_sec)
            elif self.source_type == "lsl":
                return self._get_lsl_data(duration_sec)
        except Exception as e:
            logging.error(f"Failed to get data: {e}")
            return None
    
    def _get_brainflow_data(self, duration_sec: float) -> Optional[np.ndarray]:
        """Get data from BrainFlow"""
        n_samples = int(duration_sec * self.fs)
        
        # Wait for enough data to accumulate
        time.sleep(duration_sec)
        
        # Get all available data
        data = self.board.get_board_data()
        
        if data.shape[1] < n_samples:
            logging.warning(f"Insufficient data: got {data.shape[1]}, needed {n_samples}")
            return None
            
        # Extract EEG channels and most recent samples
        eeg_data = data[self.eeg_channels, -n_samples:]
        return eeg_data
    
    def _get_lsl_data(self, duration_sec: float) -> Optional[np.ndarray]:
        """Get data from LSL"""
        n_samples = int(duration_sec * self.fs)
        
        # Collect samples
        samples = []
        for _ in range(n_samples):
            sample, _ = self.lsl_inlet.pull_sample(timeout=1.0)
            if sample is None:
                logging.warning("LSL timeout - no data received")
                return None
            samples.append(sample)
        
        # Convert to numpy array and transpose to (channels x samples)
        eeg_data = np.array(samples).T
        return eeg_data
    
    def disconnect(self):
        """Clean disconnect from data source"""
        try:
            if self.board is not None:
                self.board.stop_stream()
                self.board.release_session()
                logging.info("BrainFlow disconnected")
                
            if self.lsl_inlet is not None:
                self.lsl_inlet.close_stream()
                logging.info("LSL disconnected")
                
        except Exception as e:
            logging.error(f"Disconnect error: {e}")
        finally:
            self.is_connected = False

class Preprocessor:
    """
    EEG signal preprocessing pipeline
    
    Handles filtering, artifact detection, and windowing of raw EEG signals.
    This is crucial for clean feature extraction.
    """
    
    def __init__(self, fs: float, notch_freq: float = NOTCH_HZ, 
                 bandpass: Tuple[float, float] = BANDPASS):
        self.fs = fs
        self.notch_freq = notch_freq
        self.bandpass = bandpass
        
        # Design filters
        self._design_filters()
        
    def _design_filters(self):
        """Design digital filters for preprocessing"""
        nyquist = self.fs / 2
        
        # Notch filter for power line interference
        Q = 30  # Quality factor
        w0 = self.notch_freq / nyquist
        self.notch_b, self.notch_a = sp_signal.iirnotch(w0, Q)
        
        # Band-pass filter
        low = self.bandpass[0] / nyquist
        high = self.bandpass[1] / nyquist
        self.bp_b, self.bp_a = sp_signal.butter(4, [low, high], btype='band')
        
        logging.info(f"Filters designed: Notch {self.notch_freq}Hz, BP {self.bandpass}Hz")
    
    def filter_data(self, data: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing filters to EEG data
        
        Args:
            data: Raw EEG data (channels x samples)
            
        Returns:
            np.ndarray: Filtered EEG data
        """
        filtered = data.copy()
        
        # Apply filters channel by channel
        for ch in range(filtered.shape[0]):
            # Notch filter
            filtered[ch, :] = sp_signal.filtfilt(self.notch_b, self.notch_a, filtered[ch, :])
            # Band-pass filter
            filtered[ch, :] = sp_signal.filtfilt(self.bp_b, self.bp_a, filtered[ch, :])
            
        return filtered
    
    def detect_artifacts(self, data: np.ndarray, channel_map: Dict[str, int]) -> bool:
        """
        Simple artifact detection based on amplitude thresholds
        
        Args:
            data: Filtered EEG data (channels x samples)
            channel_map: Mapping of channel names to indices
            
        Returns:
            bool: True if artifacts detected
        """
        try:
            # Check frontal channels for blinks/muscle artifacts
            for ch_name in ARTIFACT_CHANNELS:
                if ch_name in channel_map:
                    ch_idx = channel_map[ch_name]
                    if ch_idx < data.shape[0]:
                        # Compute z-score for this channel
                        z_scores = np.abs(zscore(data[ch_idx, :]))
                        if np.any(z_scores > ARTIFACT_ZSCORE_THRESH):
                            return True
            return False
        except Exception:
            # If artifact detection fails, assume clean data
            return False
    
    def create_window(self, data: np.ndarray, timestamp: float, 
                     channel_map: Dict[str, int]) -> EEGWindow:
        """
        Create an EEG analysis window with preprocessing
        
        Args:
            data: Raw EEG data (channels x samples)
            timestamp: Window timestamp
            channel_map: Channel name to index mapping
            
        Returns:
            EEGWindow: Processed window ready for analysis
        """
        # Apply filters
        filtered_data = self.filter_data(data)
        
        # Detect artifacts
        is_artifact = self.detect_artifacts(filtered_data, channel_map)
        
        return EEGWindow(
            data=filtered_data,
            timestamp=timestamp,
            fs=self.fs,
            is_artifact=is_artifact
        )

class FeatureExtractor:
    """
    Extract frequency domain features from EEG windows
    
    This class computes power spectral density using Welch's method
    and extracts band powers for different frequency bands.
    """
    
    def __init__(self, fs: float, freq_bands: Dict[str, Tuple[float, float]] = FREQ_BANDS):
        self.fs = fs
        self.freq_bands = freq_bands
        
    def compute_welch_psd(self, data: np.ndarray, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density using Welch's method
        
        Args:
            data: EEG data for single channel (samples,)
            nperseg: Length of each segment for Welch's method
            
        Returns:
            Tuple[frequencies, power]: Frequency bins and power values
        """
        if nperseg is None:
            nperseg = min(int(self.fs), len(data))
            
        freqs, psd = sp_signal.welch(data, fs=self.fs, nperseg=nperseg, 
                                   noverlap=nperseg//2, window='hann')
        return freqs, psd
    
    def extract_band_power(self, data: np.ndarray, freq_range: Tuple[float, float]) -> float:
        """
        Extract power in specific frequency band
        
        Args:
            data: EEG data for single channel (samples,)
            freq_range: (low_freq, high_freq) in Hz
            
        Returns:
            float: Average power in the frequency band
        """
        freqs, psd = self.compute_welch_psd(data)
        
        # Find frequency indices
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        
        if not np.any(freq_mask):
            return 0.0
            
        # Return average power in band
        return np.mean(psd[freq_mask])
    
    def extract_features(self, window: EEGWindow, channel_map: Dict[str, int]) -> BandPowers:
        """
        Extract all band power features from an EEG window
        
        Args:
            window: EEG analysis window
            channel_map: Mapping of channel names to indices
            
        Returns:
            BandPowers: Extracted frequency band powers
        """
        powers = BandPowers()
        
        try:
            # Extract theta, alpha, beta from appropriate electrode groups
            for band_name, (low, high) in self.freq_bands.items():
                if band_name == "theta":
                    # Use all available channels for theta (general activity)
                    all_powers = []
                    for ch_idx in range(window.data.shape[0]):
                        power = self.extract_band_power(window.data[ch_idx, :], (low, high))
                        all_powers.append(power)
                    powers.theta = np.mean(all_powers) if all_powers else 0.0
                    
                elif band_name == "alpha":
                    # Use occipital channels for alpha (relaxation)
                    alpha_powers = []
                    for ch_name in ELECTRODE_GROUPS["alpha"]:
                        if ch_name in channel_map:
                            ch_idx = channel_map[ch_name]
                            if ch_idx < window.data.shape[0]:
                                power = self.extract_band_power(window.data[ch_idx, :], (low, high))
                                alpha_powers.append(power)
                    powers.alpha = np.mean(alpha_powers) if alpha_powers else 0.0
                    
                elif band_name == "beta":
                    # Use frontal channels for beta (focus)
                    beta_powers = []
                    for ch_name in ELECTRODE_GROUPS["beta"]:
                        if ch_name in channel_map:
                            ch_idx = channel_map[ch_name]
                            if ch_idx < window.data.shape[0]:
                                power = self.extract_band_power(window.data[ch_idx, :], (low, high))
                                beta_powers.append(power)
                    powers.beta = np.mean(beta_powers) if beta_powers else 0.0
                    
                elif band_name == "mu":
                    # Extract mu power from C3 and C4 separately for motor imagery
                    if "C3" in channel_map:
                        ch_idx = channel_map["C3"]
                        if ch_idx < window.data.shape[0]:
                            powers.mu_c3 = self.extract_band_power(window.data[ch_idx, :], (low, high))
                    
                    if "C4" in channel_map:
                        ch_idx = channel_map["C4"]
                        if ch_idx < window.data.shape[0]:
                            powers.mu_c4 = self.extract_band_power(window.data[ch_idx, :], (low, high))
                            
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            
        return powers

# TODO: Continue with RelaxFocusDetector and other classes...
# This is getting quite long, so I'll implement the remaining classes in the next section

class RelaxFocusDetector:
    """
    Detect relaxation vs focus states using calibrated thresholds
    
    This detector uses personalized thresholds learned during calibration
    to classify mental states with hysteresis to prevent rapid switching.
    """
    
    def __init__(self, profile_path: Optional[str] = None):
        self.profile_path = profile_path
        self.profile = None
        self.last_state = "neutral"
        self.last_state_time = 0.0
        self.ema_relax = 0.0
        self.ema_focus = 0.0
        self.ema_alpha = 0.9  # EMA smoothing factor
        
        # Load profile if available
        if profile_path and os.path.exists(profile_path):
            self.load_profile(profile_path)
    
    def load_profile(self, profile_path: str) -> bool:
        """Load user calibration profile"""
        try:
            with open(profile_path, 'r') as f:
                self.profile = json.load(f)
            logging.info(f"Loaded profile: {profile_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load profile: {e}")
            return False
    
    def compute_indices(self, powers: BandPowers) -> Tuple[float, float]:
        """
        Compute RelaxIndex and FocusIndex from band powers
        
        Args:
            powers: Extracted band powers
            
        Returns:
            Tuple[relax_index, focus_index]
        """
        # Prevent division by zero
        epsilon = 1e-6
        
        # RelaxIndex = Alpha / (Theta + Beta)
        relax_index = powers.alpha / (powers.theta + powers.beta + epsilon)
        
        # FocusIndex = Beta / Alpha  
        focus_index = powers.beta / (powers.alpha + epsilon)
        
        return relax_index, focus_index
    
    def update_ema(self, relax_index: float, focus_index: float):
        """Update exponential moving averages"""
        if self.ema_relax == 0.0:  # First update
            self.ema_relax = relax_index
            self.ema_focus = focus_index
        else:
            self.ema_relax = self.ema_alpha * self.ema_relax + (1 - self.ema_alpha) * relax_index
            self.ema_focus = self.ema_alpha * self.ema_focus + (1 - self.ema_alpha) * focus_index
    
    def detect_state(self, powers: BandPowers, timestamp: float) -> str:
        """
        Detect mental state with hysteresis
        
        Args:
            powers: Band powers from current window
            timestamp: Current timestamp
            
        Returns:
            str: Detected state ("relax", "focus", "neutral")
        """
        relax_index, focus_index = self.compute_indices(powers)
        self.update_ema(relax_index, focus_index)
        
        # Use default thresholds if no profile loaded
        if self.profile is None:
            relax_thresh = 1.2  # Default threshold
            focus_thresh = 0.8  # Default threshold
        else:
            relax_thresh = self.profile.get("relax", {}).get("threshold", 1.2)
            focus_thresh = self.profile.get("focus", {}).get("threshold", 0.8)
        
        # State detection with hysteresis
        current_time = timestamp
        hold_time = HOLD_MS / 1000.0  # Convert to seconds
        
        new_state = "neutral"
        if self.ema_relax > relax_thresh:
            new_state = "relax"
        elif self.ema_focus > focus_thresh:
            new_state = "focus"
        
        # Apply minimum hold time to prevent rapid switching
        if new_state != self.last_state:
            if current_time - self.last_state_time < hold_time:
                new_state = self.last_state  # Keep previous state
            else:
                self.last_state = new_state
                self.last_state_time = current_time
        
        return new_state

class MotorImageryDetector:
    """
    Motor imagery detection using trained CSP+LDA model or heuristic fallback
    
    This detector can use a pre-trained machine learning model for accurate
    classification, or fall back to a simple mu rhythm ERD heuristic.
    """
    
    def __init__(self, model_path: str = MI_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.use_model = False
        self.ema_left = 0.5
        self.ema_right = 0.5
        self.ema_alpha = 0.8  # Smoothing factor
        
        # Try to load trained model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load trained CSP+LDA model if available"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.use_model = True
                logging.info(f"Loaded MI model: {self.model_path}")
                return True
            else:
                logging.info("No MI model found - using heuristic fallback")
                return False
        except Exception as e:
            logging.error(f"Failed to load MI model: {e}")
            return False
    
    def _heuristic_detection(self, powers: BandPowers) -> Tuple[float, float]:
        """
        Heuristic motor imagery detection using mu ERD
        
        Right-hand imagery typically causes mu ERD at C3 (contralateral)
        Left-hand imagery typically causes mu ERD at C4 (contralateral)
        
        Args:
            powers: Band powers including mu_c3 and mu_c4
            
        Returns:
            Tuple[prob_left, prob_right]: Pseudo-probabilities
        """
        # Prevent division by zero
        epsilon = 1e-6
        total_mu = powers.mu_c3 + powers.mu_c4 + epsilon
        
        # ERD score: positive means C4 > C3 (right hand imagery)
        erd_score = (powers.mu_c4 - powers.mu_c3) / total_mu
        
        # Convert to pseudo-probabilities with sigmoid and dead zone
        dead_zone = 0.1
        if abs(erd_score) < dead_zone:
            # Neutral zone
            prob_left = 0.5
            prob_right = 0.5
        else:
            # Sigmoid mapping
            sigmoid_input = (erd_score - dead_zone) * 5  # Scale factor
            sigmoid_val = 1 / (1 + np.exp(-sigmoid_input))
            
            prob_right = sigmoid_val
            prob_left = 1 - sigmoid_val
        
        return prob_left, prob_right
    
    def _model_detection(self, window: EEGWindow, channel_map: Dict[str, int]) -> Tuple[float, float]:
        """
        Model-based motor imagery detection
        
        Args:
            window: EEG window data
            channel_map: Channel mapping
            
        Returns:
            Tuple[prob_left, prob_right]: Class probabilities
        """
        try:
            # Extract motor channels (C3, C4, Cz)
            mi_channels = []
            mi_indices = []
            for ch_name in ELECTRODE_GROUPS["mi"]:
                if ch_name in channel_map:
                    ch_idx = channel_map[ch_name]
                    if ch_idx < window.data.shape[0]:
                        mi_channels.append(window.data[ch_idx, :])
                        mi_indices.append(ch_idx)
            
            if len(mi_channels) < 2:
                # Fallback to heuristic if insufficient channels
                return 0.5, 0.5
            
            # Prepare data for model (reshape to match training format)
            X = np.array(mi_channels).reshape(1, -1)  # Flatten for sklearn
            
            # Get predictions
            proba = self.model.predict_proba(X)[0]
            
            # Assume model classes are [LEFT, RIGHT]
            prob_left = proba[0] if len(proba) > 0 else 0.5
            prob_right = proba[1] if len(proba) > 1 else 0.5
            
            return prob_left, prob_right
            
        except Exception as e:
            logging.error(f"Model prediction failed: {e}")
            return 0.5, 0.5
    
    def detect(self, window: EEGWindow, powers: BandPowers, 
              channel_map: Dict[str, int]) -> Tuple[float, float]:
        """
        Detect motor imagery with smoothing
        
        Args:
            window: EEG window
            powers: Band powers
            channel_map: Channel mapping
            
        Returns:
            Tuple[prob_left, prob_right]: Smoothed probabilities
        """
        if self.use_model and self.model is not None:
            prob_left, prob_right = self._model_detection(window, channel_map)
        else:
            prob_left, prob_right = self._heuristic_detection(powers)
        
        # Apply EMA smoothing
        self.ema_left = self.ema_alpha * self.ema_left + (1 - self.ema_alpha) * prob_left
        self.ema_right = self.ema_alpha * self.ema_right + (1 - self.ema_alpha) * prob_right
        
        return self.ema_left, self.ema_right

class UnitySender:
    """
    Send mental state updates to Unity via UDP JSON messages
    
    This class handles the communication with Unity, formatting the EEG
    analysis results into JSON and sending them over UDP.
    """
    
    def __init__(self, host: str = UDP_HOST, port: int = UDP_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self._setup_socket()
    
    def _setup_socket(self):
        """Setup UDP socket for communication"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            logging.info(f"UDP sender initialized: {self.host}:{self.port}")
        except Exception as e:
            logging.error(f"Failed to setup UDP socket: {e}")
    
    def send_state(self, state: MentalState) -> bool:
        """
        Send mental state to Unity
        
        Args:
            state: Current mental state
            
        Returns:
            bool: True if sent successfully
        """
        if self.socket is None:
            return False
            
        try:
            # Create JSON message
            message = {
                "t": state.timestamp,
                "alpha": float(state.band_powers.alpha),
                "beta": float(state.band_powers.beta),
                "theta": float(state.band_powers.theta),
                "relax_index": float(state.relax_index),
                "focus_index": float(state.focus_index),
                "state": state.state,
                "mi_prob_left": float(state.mi_prob_left),
                "mi_prob_right": float(state.mi_prob_right),
                "artifact": state.artifact
            }
            
            # Send JSON over UDP
            json_str = json.dumps(message)
            self.socket.sendto(json_str.encode('utf-8'), (self.host, self.port))
            return True
            
        except Exception as e:
            logging.error(f"Failed to send UDP message: {e}")
            return False
    
    def close(self):
        """Close UDP socket"""
        if self.socket:
            self.socket.close()

# ============================================================================
# SYNTHETIC DATA GENERATOR FOR TESTING
# ============================================================================

class FakeEEGSource:
    """
    Generate synthetic EEG data for testing Unity integration
    
    This creates realistic-looking EEG with controllable alpha/beta patterns
    and motor imagery signatures for development and testing.
    """
    
    def __init__(self, fs: float = FS_EXPECTED, n_channels: int = 16):
        self.fs = fs
        self.n_channels = n_channels
        self.time = 0.0
        self.state_cycle_time = 10.0  # Switch states every 10 seconds
        self.mi_cycle_time = 8.0      # Switch MI every 8 seconds
        
    def generate_window(self, duration_sec: float) -> np.ndarray:
        """
        Generate synthetic EEG window
        
        Args:
            duration_sec: Duration of data to generate
            
        Returns:
            np.ndarray: Synthetic EEG data (channels x samples)
        """
        n_samples = int(duration_sec * self.fs)
        t = np.linspace(self.time, self.time + duration_sec, n_samples)
        
        # Base EEG with 1/f noise
        data = np.random.randn(self.n_channels, n_samples) * 10
        
        # Add frequency components
        for ch in range(self.n_channels):
            # Add alpha rhythm (8-12 Hz) - stronger in occipital
            if ch in [0, 1, 2]:  # O1, O2, Pz
                alpha_amp = 15 + 10 * np.sin(2 * np.pi * self.time / self.state_cycle_time)
                data[ch, :] += alpha_amp * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
            
            # Add beta rhythm (13-30 Hz) - stronger in frontal
            if ch in [3, 4, 5]:  # Fz, F3, F4
                beta_amp = 8 + 6 * np.cos(2 * np.pi * self.time / self.state_cycle_time)
                data[ch, :] += beta_amp * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
            
            # Add mu rhythm modulation for motor imagery (C3, C4)
            if ch in [6, 7]:  # C3, C4
                mu_phase = 2 * np.pi * self.time / self.mi_cycle_time
                if ch == 6:  # C3 - left motor cortex
                    mu_amp = 12 - 4 * np.sin(mu_phase)  # ERD during right imagery
                else:  # C4 - right motor cortex  
                    mu_amp = 12 - 4 * np.cos(mu_phase)  # ERD during left imagery
                data[ch, :] += mu_amp * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
        
        self.time += duration_sec
        return data

# ============================================================================
# CALIBRATION AND TRAINING FUNCTIONS
# ============================================================================

def run_calibration(user_id: str, brain_source: BrainSource, 
                   preprocessor: Preprocessor, feature_extractor: FeatureExtractor) -> bool:
    """
    Run calibration procedure to determine user-specific thresholds
    
    This guides the user through relaxation and focus tasks while collecting
    EEG data to compute personalized detection thresholds.
    """
    logging.info("Starting calibration procedure...")
    
    # Create profile directory
    os.makedirs(PROFILE_DIR, exist_ok=True)
    profile_path = os.path.join(PROFILE_DIR, f"{user_id}.json")
    
    # Data collection lists
    relax_indices = []
    focus_indices = []
    
    try:
        # Phase 1: Relaxation (90 seconds)
        print("\n" + "="*60)
        print("CALIBRATION PHASE 1: RELAXATION (90 seconds)")
        print("Please sit comfortably, close your eyes, and relax.")
        print("Try to clear your mind and focus on your breathing.")
        print("Press ENTER when ready...")
        input()
        
        print("Starting relaxation recording...")
        start_time = time.time()
        
        while time.time() - start_time < 90:
            # Get EEG window
            raw_data = brain_source.get_data(WINDOW_SEC)
            if raw_data is None:
                continue
                
            # Process window
            window = preprocessor.create_window(raw_data, time.time(), CHANNELS)
            if window.is_artifact:
                continue  # Skip artifacted windows
                
            # Extract features
            powers = feature_extractor.extract_features(window, CHANNELS)
            
            # Compute indices
            relax_index = powers.alpha / (powers.theta + powers.beta + 1e-6)
            focus_index = powers.beta / (powers.alpha + 1e-6)
            
            relax_indices.append(relax_index)
            
            # Progress indicator
            elapsed = time.time() - start_time
            print(f"\rRelaxation: {elapsed:.1f}/90.0s - RelaxIndex: {relax_index:.2f}", end="")
        
        print(f"\nRelaxation phase complete. Collected {len(relax_indices)} samples.")
        
        # Phase 2: Focus (90 seconds)
        print("\n" + "="*60)
        print("CALIBRATION PHASE 2: FOCUS (90 seconds)")
        print("Please open your eyes and focus intensely.")
        print("Try mental math: count backwards from 1000 by 7s")
        print("Or focus on a specific object with full attention.")
        print("Press ENTER when ready...")
        input()
        
        print("Starting focus recording...")
        start_time = time.time()
        
        while time.time() - start_time < 90:
            # Get EEG window
            raw_data = brain_source.get_data(WINDOW_SEC)
            if raw_data is None:
                continue
                
            # Process window
            window = preprocessor.create_window(raw_data, time.time(), CHANNELS)
            if window.is_artifact:
                continue  # Skip artifacted windows
                
            # Extract features
            powers = feature_extractor.extract_features(window, CHANNELS)
            
            # Compute indices
            relax_index = powers.alpha / (powers.theta + powers.beta + 1e-6)
            focus_index = powers.beta / (powers.alpha + 1e-6)
            
            focus_indices.append(focus_index)
            
            # Progress indicator
            elapsed = time.time() - start_time
            print(f"\rFocus: {elapsed:.1f}/90.0s - FocusIndex: {focus_index:.2f}", end="")
        
        print(f"\nFocus phase complete. Collected {len(focus_indices)} samples.")
        
        # Compute statistics and thresholds
        if len(relax_indices) < 10 or len(focus_indices) < 10:
            logging.error("Insufficient data collected for calibration")
            return False
        
        relax_mu = np.mean(relax_indices)
        relax_sigma = np.std(relax_indices)
        relax_threshold = relax_mu - 0.5 * relax_sigma  # Conservative threshold
        
        focus_mu = np.mean(focus_indices)
        focus_sigma = np.std(focus_indices)
        focus_threshold = focus_mu - 0.5 * focus_sigma  # Conservative threshold
        
        # Create profile
        profile = {
            "user_id": user_id,
            "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fs": brain_source.fs,
            "bands": dict(FREQ_BANDS),
            "relax": {
                "mu": float(relax_mu),
                "sigma": float(relax_sigma),
                "threshold": float(relax_threshold),
                "n_samples": len(relax_indices)
            },
            "focus": {
                "mu": float(focus_mu),
                "sigma": float(focus_sigma),
                "threshold": float(focus_threshold),
                "n_samples": len(focus_indices)
            },
            "ema_tau_sec": EMA_TAU_SEC,
            "hold_ms": HOLD_MS,
            "electrodes": ELECTRODE_GROUPS
        }
        
        # Save profile
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print(f"Profile saved: {profile_path}")
        print(f"Relax threshold: {relax_threshold:.3f} (μ={relax_mu:.3f}, σ={relax_sigma:.3f})")
        print(f"Focus threshold: {focus_threshold:.3f} (μ={focus_mu:.3f}, σ={focus_sigma:.3f})")
        print("You can now use --run mode with this profile.")
        
        return True
        
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
        return False
    except Exception as e:
        logging.error(f"Calibration failed: {e}")
        return False

def train_motor_imagery_model(data_source: str, **kwargs) -> bool:
    """
    Train CSP+LDA model for motor imagery classification
    
    This function loads motor imagery data (from CSV or LSL), preprocesses it,
    extracts CSP features, trains an LDA classifier, and saves the model.
    """
    logging.info("Starting motor imagery training...")
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        if data_source == "csv":
            return _train_from_csv(kwargs.get("csv_path"))
        elif data_source == "lsl":
            return _train_from_lsl(kwargs)
        else:
            logging.error(f"Unknown data source: {data_source}")
            return False
            
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return False

def _train_from_csv(csv_path: str) -> bool:
    """Train model from CSV data"""
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return False
    
    # This is a placeholder - in a real implementation, you would:
    # 1. Load CSV data with pandas
    # 2. Extract epochs based on markers (LEFT/RIGHT)
    # 3. Preprocess (band-pass 8-30 Hz)
    # 4. Apply CSP
    # 5. Train LDA
    # 6. Save model with joblib
    
    logging.warning("CSV training not fully implemented - this is a placeholder")
    return False

def _train_from_lsl(kwargs: Dict) -> bool:
    """Train model from LSL streams"""
    # This is a placeholder - in a real implementation, you would:
    # 1. Connect to LSL EEG and marker streams
    # 2. Collect trials with LEFT/RIGHT markers
    # 3. Epoch data around markers
    # 4. Apply CSP and train LDA
    # 5. Save model
    
    logging.warning("LSL training not fully implemented - this is a placeholder")
    return False

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

def run_realtime_processing(user_id: str, brain_source: BrainSource, 
                           udp_host: str, udp_port: int) -> None:
    """
    Main real-time processing loop
    
    This is the core function that continuously processes EEG data,
    extracts features, detects mental states, and sends updates to Unity.
    """
    logging.info("Starting real-time processing...")
    
    # Initialize components
    preprocessor = Preprocessor(brain_source.fs)
    feature_extractor = FeatureExtractor(brain_source.fs)
    
    # Load user profile for relax/focus detection
    profile_path = os.path.join(PROFILE_DIR, f"{user_id}.json")
    relax_focus_detector = RelaxFocusDetector(profile_path)
    
    # Initialize motor imagery detector
    mi_detector = MotorImageryDetector()
    
    # Initialize Unity sender
    unity_sender = UnitySender(udp_host, udp_port)
    
    # Processing state
    last_status_time = 0.0
    status_interval = 2.0  # Print status every 2 seconds
    
    # Graceful shutdown handler
    shutdown_event = Event()
    def signal_handler(signum, frame):
        logging.info("Shutdown signal received")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logging.info("Real-time processing started. Press Ctrl+C to stop.")
        
        while not shutdown_event.is_set():
            current_time = time.time()
            
            # Get EEG data
            raw_data = brain_source.get_data(WINDOW_SEC)
            if raw_data is None:
                time.sleep(0.1)  # Brief pause if no data
                continue
            
            # Create analysis window
            window = preprocessor.create_window(raw_data, current_time, CHANNELS)
            
            # Extract features
            powers = feature_extractor.extract_features(window, CHANNELS)
            
            # Detect relax/focus state
            relax_index, focus_index = relax_focus_detector.compute_indices(powers)
            state = relax_focus_detector.detect_state(powers, current_time)
            
            # Detect motor imagery
            mi_left, mi_right = mi_detector.detect(window, powers, CHANNELS)
            
            # Create mental state object
            mental_state = MentalState(
                timestamp=current_time,
                relax_index=relax_index,
                focus_index=focus_index,
                state=state,
                mi_prob_left=mi_left,
                mi_prob_right=mi_right,
                band_powers=powers,
                artifact=window.is_artifact
            )
            
            # Send to Unity (skip if artifact detected)
            if not window.is_artifact:
                unity_sender.send_state(mental_state)
            
            # Print status periodically
            if current_time - last_status_time > status_interval:
                print(f"State: {state:>7} | Relax: {relax_index:.2f} | Focus: {focus_index:.2f} | "
                      f"MI L/R: {mi_left:.2f}/{mi_right:.2f} | Artifact: {window.is_artifact}")
                last_status_time = current_time
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.05)  # ~20 Hz processing rate
            
    except Exception as e:
        logging.error(f"Processing error: {e}")
    finally:
        # Cleanup
        unity_sender.close()
        logging.info("Real-time processing stopped")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="EEG Bridge - Real-time BCI processing for Unity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate user profile
  python eeg_bridge.py --calibrate --user alice
  
  # Train motor imagery model from CSV
  python eeg_bridge.py --train-mi --csv data/mi_trials.csv
  
  # Run real-time processing
  python eeg_bridge.py --run --user alice
  
  # Test with synthetic data
  python eeg_bridge.py --run --fake --user test
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--calibrate", action="store_true",
                           help="Run calibration procedure")
    mode_group.add_argument("--train-mi", action="store_true", 
                           help="Train motor imagery model")
    mode_group.add_argument("--run", action="store_true",
                           help="Run real-time processing")
    
    # Data source options
    parser.add_argument("--source", choices=["brainflow", "lsl"], default="brainflow",
                       help="EEG data source (default: brainflow)")
    parser.add_argument("--fake", action="store_true",
                       help="Use synthetic EEG data for testing")
    parser.add_argument("--serial-port", default=SERIAL_PORT,
                       help=f"Serial port for BrainFlow (default: {SERIAL_PORT})")
    parser.add_argument("--lsl-stream", default="EEG",
                       help="LSL stream name (default: EEG)")
    
    # User and file options
    parser.add_argument("--user", required=True,
                       help="User ID for profiles and models")
    parser.add_argument("--csv", 
                       help="CSV file path for motor imagery training")
    
    # Processing parameters
    parser.add_argument("--fs", type=int, default=FS_EXPECTED,
                       help=f"Sampling frequency (default: {FS_EXPECTED})")
    parser.add_argument("--notch", type=int, choices=[50, 60], default=NOTCH_HZ,
                       help=f"Notch filter frequency (default: {NOTCH_HZ})")
    parser.add_argument("--bandpass-low", type=float, default=BANDPASS[0],
                       help=f"Bandpass low frequency (default: {BANDPASS[0]})")
    parser.add_argument("--bandpass-high", type=float, default=BANDPASS[1],
                       help=f"Bandpass high frequency (default: {BANDPASS[1]})")
    
    # Communication options  
    parser.add_argument("--udp-host", default=UDP_HOST,
                       help=f"Unity UDP host (default: {UDP_HOST})")
    parser.add_argument("--udp-port", type=int, default=UDP_PORT,
                       help=f"Unity UDP port (default: {UDP_PORT})")
    
    # Logging
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    return parser

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Print startup info
    print("="*60)
    print("EEG Bridge - Real-time BCI Processing")
    print("="*60)
    
    # Update global config from args
    global NOTCH_HZ, BANDPASS, FS_EXPECTED
    NOTCH_HZ = args.notch
    BANDPASS = (args.bandpass_low, args.bandpass_high)
    FS_EXPECTED = args.fs
    
    try:
        # Initialize data source
        if args.fake:
            logging.info("Using synthetic EEG data")
            brain_source = FakeEEGSource(args.fs)
            # Wrap fake source to match BrainSource interface
            class FakeSourceWrapper:
                def __init__(self, fake_source):
                    self.fake_source = fake_source
                    self.fs = fake_source.fs
                    self.is_connected = True
                
                def connect(self):
                    return True
                
                def get_data(self, duration_sec):
                    return self.fake_source.generate_window(duration_sec)
                
                def disconnect(self):
                    pass
            
            brain_source = FakeSourceWrapper(brain_source)
        else:
            brain_source = BrainSource(
                source_type=args.source,
                serial_port=args.serial_port,
                lsl_stream_name=args.lsl_stream
            )
            
            if not brain_source.connect():
                logging.error("Failed to connect to EEG source")
                return 1
        
        # Execute requested mode
        if args.calibrate:
            preprocessor = Preprocessor(brain_source.fs, args.notch, BANDPASS)
            feature_extractor = FeatureExtractor(brain_source.fs)
            success = run_calibration(args.user, brain_source, preprocessor, feature_extractor)
            return 0 if success else 1
            
        elif args.train_mi:
            if args.csv:
                success = train_motor_imagery_model("csv", csv_path=args.csv)
            else:
                success = train_motor_imagery_model("lsl")
            return 0 if success else 1
            
        elif args.run:
            run_realtime_processing(args.user, brain_source, args.udp_host, args.udp_port)
            return 0
            
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        return 0
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return 1
    finally:
        # Cleanup
        if 'brain_source' in locals() and hasattr(brain_source, 'disconnect'):
            brain_source.disconnect()

if __name__ == "__main__":
    sys.exit(main())
