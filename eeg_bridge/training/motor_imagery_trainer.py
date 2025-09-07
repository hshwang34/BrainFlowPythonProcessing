"""
Motor imagery model training and calibration

This module handles training CSP+LDA models for motor imagery classification
and running calibration procedures for personalized thresholds.
"""

import json
import logging
import os
import time
from typing import Dict

import numpy as np

from ..core.config import PROFILE_DIR, MODEL_DIR
from ..core.data_types import BandPowers
from ..processing.preprocessor import Preprocessor
from ..processing.features import FeatureExtractor
from ..acquisition.sources import BrainSource


def run_calibration(user_id: str, brain_source: BrainSource, 
                   preprocessor: Preprocessor, feature_extractor: FeatureExtractor) -> bool:
    """
    Run calibration procedure to determine user-specific thresholds
    
    This guides the user through relaxation and focus tasks while collecting
    EEG data to compute personalized detection thresholds.
    """
    logging.info("Starting calibration procedure...")
    
    # Create profile directory
    os.makedirs(PROFILE_DIR, exist_ok=True)
    profile_path = os.path.join(PROFILE_DIR, f"{user_id}.json")
    
    # Import here to avoid circular imports
    from ..core.config import CHANNELS, FREQ_BANDS, EMA_TAU_SEC, HOLD_MS, ELECTRODE_GROUPS
    
    # Data collection lists
    relax_indices = []
    focus_indices = []
    
    try:
        # Phase 1: Relaxation (90 seconds)
        print("\n" + "="*60)
        print("CALIBRATION PHASE 1: RELAXATION (90 seconds)")
        print("Please sit comfortably, close your eyes, and relax.")
        print("Try to clear your mind and focus on your breathing.")
        print("Press ENTER when ready...")
        input()
        
        print("Starting relaxation recording...")
        start_time = time.time()
        
        while time.time() - start_time < 90:
            # Get EEG window
            raw_data = brain_source.get_data(1.0)  # 1 second window
            if raw_data is None:
                continue
                
            # Process window
            window = preprocessor.create_window(raw_data, time.time(), CHANNELS)
            if window.is_artifact:
                continue  # Skip artifacted windows
                
            # Extract features
            powers = feature_extractor.extract_features(window, CHANNELS)
            
            # Compute indices
            relax_index = powers.alpha / (powers.theta + powers.beta + 1e-6)
            focus_index = powers.beta / (powers.alpha + 1e-6)
            
            relax_indices.append(relax_index)
            
            # Progress indicator
            elapsed = time.time() - start_time
            print(f"\rRelaxation: {elapsed:.1f}/90.0s - RelaxIndex: {relax_index:.2f}", end="")
        
        print(f"\nRelaxation phase complete. Collected {len(relax_indices)} samples.")
        
        # Phase 2: Focus (90 seconds)
        print("\n" + "="*60)
        print("CALIBRATION PHASE 2: FOCUS (90 seconds)")
        print("Please open your eyes and focus intensely.")
        print("Try mental math: count backwards from 1000 by 7s")
        print("Or focus on a specific object with full attention.")
        print("Press ENTER when ready...")
        input()
        
        print("Starting focus recording...")
        start_time = time.time()
        
        while time.time() - start_time < 90:
            # Get EEG window
            raw_data = brain_source.get_data(1.0)
            if raw_data is None:
                continue
                
            # Process window
            window = preprocessor.create_window(raw_data, time.time(), CHANNELS)
            if window.is_artifact:
                continue  # Skip artifacted windows
                
            # Extract features
            powers = feature_extractor.extract_features(window, CHANNELS)
            
            # Compute indices
            relax_index = powers.alpha / (powers.theta + powers.beta + 1e-6)
            focus_index = powers.beta / (powers.alpha + 1e-6)
            
            focus_indices.append(focus_index)
            
            # Progress indicator
            elapsed = time.time() - start_time
            print(f"\rFocus: {elapsed:.1f}/90.0s - FocusIndex: {focus_index:.2f}", end="")
        
        print(f"\nFocus phase complete. Collected {len(focus_indices)} samples.")
        
        # Compute statistics and thresholds
        if len(relax_indices) < 10 or len(focus_indices) < 10:
            logging.error("Insufficient data collected for calibration")
            return False
        
        relax_mu = np.mean(relax_indices)
        relax_sigma = np.std(relax_indices)
        relax_threshold = relax_mu - 0.5 * relax_sigma  # Conservative threshold
        
        focus_mu = np.mean(focus_indices)
        focus_sigma = np.std(focus_indices)
        focus_threshold = focus_mu - 0.5 * focus_sigma  # Conservative threshold
        
        # Create profile
        profile = {
            "user_id": user_id,
            "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fs": brain_source.fs,
            "bands": dict(FREQ_BANDS),
            "relax": {
                "mu": float(relax_mu),
                "sigma": float(relax_sigma),
                "threshold": float(relax_threshold),
                "n_samples": len(relax_indices)
            },
            "focus": {
                "mu": float(focus_mu),
                "sigma": float(focus_sigma),
                "threshold": float(focus_threshold),
                "n_samples": len(focus_indices)
            },
            "ema_tau_sec": EMA_TAU_SEC,
            "hold_ms": HOLD_MS,
            "electrodes": ELECTRODE_GROUPS
        }
        
        # Save profile
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print(f"Profile saved: {profile_path}")
        print(f"Relax threshold: {relax_threshold:.3f} (μ={relax_mu:.3f}, σ={relax_sigma:.3f})")
        print(f"Focus threshold: {focus_threshold:.3f} (μ={focus_mu:.3f}, σ={focus_sigma:.3f})")
        print("You can now use --run mode with this profile.")
        
        return True
        
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
        return False
    except Exception as e:
        logging.error(f"Calibration failed: {e}")
        return False


def train_motor_imagery_model(data_source: str, **kwargs) -> bool:
    """
    Train CSP+LDA model for motor imagery classification
    
    This function loads motor imagery data (from CSV or LSL), preprocesses it,
    extracts CSP features, trains an LDA classifier, and saves the model.
    """
    logging.info("Starting motor imagery training...")
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        if data_source == "csv":
            return _train_from_csv(kwargs.get("csv_path"))
        elif data_source == "lsl":
            return _train_from_lsl(kwargs)
        else:
            logging.error(f"Unknown data source: {data_source}")
            return False
            
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return False


def _train_from_csv(csv_path: str) -> bool:
    """Train model from CSV data"""
    if not csv_path or not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return False
    
    # This is a placeholder - in a real implementation, you would:
    # 1. Load CSV data with pandas
    # 2. Extract epochs based on markers (LEFT/RIGHT)
    # 3. Preprocess (band-pass 8-30 Hz)
    # 4. Apply CSP
    # 5. Train LDA
    # 6. Save model with joblib
    
    logging.warning("CSV training not fully implemented - this is a placeholder")
    return False


def _train_from_lsl(kwargs: Dict) -> bool:
    """Train model from LSL streams"""
    # This is a placeholder - in a real implementation, you would:
    # 1. Connect to LSL EEG and marker streams
    # 2. Collect trials with LEFT/RIGHT markers
    # 3. Epoch data around markers
    # 4. Apply CSP and train LDA
    # 5. Save model
    
    logging.warning("LSL training not fully implemented - this is a placeholder")
    return False
