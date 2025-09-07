"""
Core data types for EEG Bridge

This module defines the fundamental data structures used throughout the system
for representing EEG data, features, and mental states.
"""

from dataclasses import dataclass
import numpy as np

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
