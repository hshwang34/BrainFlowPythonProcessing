"""
Mental state detection algorithms

This module implements various mental state detectors including
relaxation/focus classification and motor imagery detection.
"""

from .relax_focus import RelaxFocusDetector
from .motor_imagery import MotorImageryDetector

__all__ = ['RelaxFocusDetector', 'MotorImageryDetector']
