"""
EEG signal preprocessing pipeline

This module handles filtering, artifact detection, and windowing of raw EEG signals.
Preprocessing is crucial for clean feature extraction and reliable mental state detection.
"""

import logging
from typing import Dict, Tuple
import numpy as np
from scipy import signal as sp_signal
from scipy.stats import zscore

from ..core.data_types import EEGWindow
from ..core.config import NOTCH_HZ, BANDPASS, ARTIFACT_ZSCORE_THRESH, ARTIFACT_CHANNELS


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
