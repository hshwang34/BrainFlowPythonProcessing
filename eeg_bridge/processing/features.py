"""
EEG feature extraction

This module extracts frequency domain features from EEG windows using
Welch's method for power spectral density estimation.
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
from scipy import signal as sp_signal

from ..core.data_types import EEGWindow, BandPowers
from ..core.config import FREQ_BANDS, ELECTRODE_GROUPS


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
