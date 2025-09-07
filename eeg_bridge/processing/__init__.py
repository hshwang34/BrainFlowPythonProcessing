"""
EEG signal processing components

This module contains preprocessing and feature extraction functionality
for real-time EEG analysis.
"""

from .preprocessor import Preprocessor
from .features import FeatureExtractor

__all__ = ['Preprocessor', 'FeatureExtractor']
