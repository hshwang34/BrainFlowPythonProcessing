"""
Motor imagery detection

This module implements motor imagery detection using either trained CSP+LDA models
or a neurophysiologically-based heuristic fallback using mu rhythm ERD.
"""

import logging
import os
from typing import Dict, Tuple
import numpy as np
import joblib

from ..core.data_types import EEGWindow, BandPowers
from ..core.config import MI_MODEL_PATH, ELECTRODE_GROUPS


class MotorImageryDetector:
    """
    Motor imagery detection using trained CSP+LDA model or heuristic fallback
    
    This detector can use a pre-trained machine learning model for accurate
    classification, or fall back to a simple mu rhythm ERD heuristic.
    """
    
    def __init__(self, model_path: str = MI_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.use_model = False
        self.ema_left = 0.5
        self.ema_right = 0.5
        self.ema_alpha = 0.8  # Smoothing factor
        
        # Try to load trained model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load trained CSP+LDA model if available"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.use_model = True
                logging.info(f"Loaded MI model: {self.model_path}")
                return True
            else:
                logging.info("No MI model found - using heuristic fallback")
                return False
        except Exception as e:
            logging.error(f"Failed to load MI model: {e}")
            return False
    
    def _heuristic_detection(self, powers: BandPowers) -> Tuple[float, float]:
        """
        Heuristic motor imagery detection using mu ERD
        
        Right-hand imagery typically causes mu ERD at C3 (contralateral)
        Left-hand imagery typically causes mu ERD at C4 (contralateral)
        
        Args:
            powers: Band powers including mu_c3 and mu_c4
            
        Returns:
            Tuple[prob_left, prob_right]: Pseudo-probabilities
        """
        # Prevent division by zero
        epsilon = 1e-6
        total_mu = powers.mu_c3 + powers.mu_c4 + epsilon
        
        # ERD score: positive means C4 > C3 (right hand imagery)
        erd_score = (powers.mu_c4 - powers.mu_c3) / total_mu
        
        # Convert to pseudo-probabilities with sigmoid and dead zone
        dead_zone = 0.1
        if abs(erd_score) < dead_zone:
            # Neutral zone
            prob_left = 0.5
            prob_right = 0.5
        else:
            # Sigmoid mapping
            sigmoid_input = (erd_score - dead_zone) * 5  # Scale factor
            sigmoid_val = 1 / (1 + np.exp(-sigmoid_input))
            
            prob_right = sigmoid_val
            prob_left = 1 - sigmoid_val
        
        return prob_left, prob_right
    
    def _model_detection(self, window: EEGWindow, channel_map: Dict[str, int]) -> Tuple[float, float]:
        """
        Model-based motor imagery detection
        
        Args:
            window: EEG window data
            channel_map: Channel mapping
            
        Returns:
            Tuple[prob_left, prob_right]: Class probabilities
        """
        try:
            # Extract motor channels (C3, C4, Cz)
            mi_channels = []
            mi_indices = []
            for ch_name in ELECTRODE_GROUPS["mi"]:
                if ch_name in channel_map:
                    ch_idx = channel_map[ch_name]
                    if ch_idx < window.data.shape[0]:
                        mi_channels.append(window.data[ch_idx, :])
                        mi_indices.append(ch_idx)
            
            if len(mi_channels) < 2:
                # Fallback to heuristic if insufficient channels
                return 0.5, 0.5
            
            # Prepare data for model (reshape to match training format)
            X = np.array(mi_channels).reshape(1, -1)  # Flatten for sklearn
            
            # Get predictions
            proba = self.model.predict_proba(X)[0]
            
            # Assume model classes are [LEFT, RIGHT]
            prob_left = proba[0] if len(proba) > 0 else 0.5
            prob_right = proba[1] if len(proba) > 1 else 0.5
            
            return prob_left, prob_right
            
        except Exception as e:
            logging.error(f"Model prediction failed: {e}")
            return 0.5, 0.5
    
    def detect(self, window: EEGWindow, powers: BandPowers, 
              channel_map: Dict[str, int]) -> Tuple[float, float]:
        """
        Detect motor imagery with smoothing
        
        Args:
            window: EEG window
            powers: Band powers
            channel_map: Channel mapping
            
        Returns:
            Tuple[prob_left, prob_right]: Smoothed probabilities
        """
        if self.use_model and self.model is not None:
            prob_left, prob_right = self._model_detection(window, channel_map)
        else:
            prob_left, prob_right = self._heuristic_detection(powers)
        
        # Apply EMA smoothing
        self.ema_left = self.ema_alpha * self.ema_left + (1 - self.ema_alpha) * prob_left
        self.ema_right = self.ema_alpha * self.ema_right + (1 - self.ema_alpha) * prob_right
        
        return self.ema_left, self.ema_right
