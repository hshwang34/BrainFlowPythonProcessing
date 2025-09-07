"""
Core data types and structures for EEG Bridge

This module contains the fundamental data classes used throughout the system.
"""

from .data_types import EEGWindow, BandPowers, MentalState
from .config import *

__all__ = ['EEGWindow', 'BandPowers', 'MentalState']
