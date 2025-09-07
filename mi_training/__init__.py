"""
Motor Imagery EEG Classifier Training Package

A comprehensive toolkit for training CSP+LDA/SVM classifiers on motor imagery EEG data.
Supports CSV and XDF data formats with robust preprocessing and cross-validation.

Author: AI Assistant (Claude)
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Assistant (Claude)"

# Main components for easy import
from .config import Config
from .data_io import load_csv, load_xdf
from .fake_data import synthesize_mi_trials

__all__ = ['Config', 'load_csv', 'load_xdf', 'synthesize_mi_trials']
