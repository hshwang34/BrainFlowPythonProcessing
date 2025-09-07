"""
Configuration constants for EEG Bridge

This module contains all configuration parameters that users may need to customize
for their specific hardware setup and processing requirements.
"""

from typing import Dict, Tuple

# ============================================================================
# HARDWARE CONFIGURATION - User should edit these values for their setup
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
# To find correct indices, run: python -m eeg_bridge.utils.channel_finder --board cyton-daisy
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
