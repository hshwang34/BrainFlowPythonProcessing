"""
EEG Bridge - Production-quality BCI processing for Unity integration

A modular Python package for real-time EEG signal processing, mental state detection,
and Unity game integration via UDP communication.

Author: AI Assistant (Claude)
Python: 3.10+
"""

__version__ = "1.0.0"
__author__ = "AI Assistant (Claude)"

# Main package imports for easy access
from .core.data_types import EEGWindow, BandPowers, MentalState
from .acquisition.sources import BrainSource, FakeEEGSource
from .processing.preprocessor import Preprocessor
from .processing.features import FeatureExtractor
from .detection.relax_focus import RelaxFocusDetector
from .detection.motor_imagery import MotorImageryDetector
from .communication.unity_sender import UnitySender

__all__ = [
    'EEGWindow', 'BandPowers', 'MentalState',
    'BrainSource', 'FakeEEGSource',
    'Preprocessor', 'FeatureExtractor',
    'RelaxFocusDetector', 'MotorImageryDetector',
    'UnitySender'
]
