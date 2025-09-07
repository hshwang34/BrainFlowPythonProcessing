"""
Relaxation vs Focus state detection

This module implements personalized relaxation and focus detection using
calibrated thresholds and exponential moving average smoothing.
"""

import json
import logging
import os
from typing import Optional, Tuple

from ..core.data_types import BandPowers
from ..core.config import HOLD_MS


class RelaxFocusDetector:
    """
    Detect relaxation vs focus states using calibrated thresholds
    
    This detector uses personalized thresholds learned during calibration
    to classify mental states with hysteresis to prevent rapid switching.
    """
    
    def __init__(self, profile_path: Optional[str] = None):
        self.profile_path = profile_path
        self.profile = None
        self.last_state = "neutral"
        self.last_state_time = 0.0
        self.ema_relax = 0.0
        self.ema_focus = 0.0
        self.ema_alpha = 0.9  # EMA smoothing factor
        
        # Load profile if available
        if profile_path and os.path.exists(profile_path):
            self.load_profile(profile_path)
    
    def load_profile(self, profile_path: str) -> bool:
        """Load user calibration profile"""
        try:
            with open(profile_path, 'r') as f:
                self.profile = json.load(f)
            logging.info(f"Loaded profile: {profile_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load profile: {e}")
            return False
    
    def compute_indices(self, powers: BandPowers) -> Tuple[float, float]:
        """
        Compute RelaxIndex and FocusIndex from band powers
        
        Args:
            powers: Extracted band powers
            
        Returns:
            Tuple[relax_index, focus_index]
        """
        # Prevent division by zero
        epsilon = 1e-6
        
        # RelaxIndex = Alpha / (Theta + Beta)
        relax_index = powers.alpha / (powers.theta + powers.beta + epsilon)
        
        # FocusIndex = Beta / Alpha  
        focus_index = powers.beta / (powers.alpha + epsilon)
        
        return relax_index, focus_index
    
    def update_ema(self, relax_index: float, focus_index: float):
        """Update exponential moving averages"""
        if self.ema_relax == 0.0:  # First update
            self.ema_relax = relax_index
            self.ema_focus = focus_index
        else:
            self.ema_relax = self.ema_alpha * self.ema_relax + (1 - self.ema_alpha) * relax_index
            self.ema_focus = self.ema_alpha * self.ema_focus + (1 - self.ema_alpha) * focus_index
    
    def detect_state(self, powers: BandPowers, timestamp: float) -> str:
        """
        Detect mental state with hysteresis
        
        Args:
            powers: Band powers from current window
            timestamp: Current timestamp
            
        Returns:
            str: Detected state ("relax", "focus", "neutral")
        """
        relax_index, focus_index = self.compute_indices(powers)
        self.update_ema(relax_index, focus_index)
        
        # Use default thresholds if no profile loaded
        if self.profile is None:
            relax_thresh = 1.2  # Default threshold
            focus_thresh = 0.8  # Default threshold
        else:
            relax_thresh = self.profile.get("relax", {}).get("threshold", 1.2)
            focus_thresh = self.profile.get("focus", {}).get("threshold", 0.8)
        
        # State detection with hysteresis
        current_time = timestamp
        hold_time = HOLD_MS / 1000.0  # Convert to seconds
        
        new_state = "neutral"
        if self.ema_relax > relax_thresh:
            new_state = "relax"
        elif self.ema_focus > focus_thresh:
            new_state = "focus"
        
        # Apply minimum hold time to prevent rapid switching
        if new_state != self.last_state:
            if current_time - self.last_state_time < hold_time:
                new_state = self.last_state  # Keep previous state
            else:
                self.last_state = new_state
                self.last_state_time = current_time
        
        return new_state
