"""
Model training functionality

This module handles training of machine learning models for mental state detection.
"""

from .motor_imagery_trainer import train_motor_imagery_model, run_calibration

__all__ = ['train_motor_imagery_model', 'run_calibration']
