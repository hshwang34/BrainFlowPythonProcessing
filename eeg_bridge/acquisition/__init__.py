"""
EEG data acquisition sources

This module handles different types of EEG data sources including
BrainFlow (OpenBCI), LSL streams, and synthetic data generation.
"""

from .sources import BrainSource, FakeEEGSource

__all__ = ['BrainSource', 'FakeEEGSource']
