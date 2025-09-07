"""
Main CLI entry point for EEG Bridge

This module provides the command-line interface and main processing loops
for the EEG Bridge system.
"""

import argparse
import logging
import os
import signal
import sys
import time
from threading import Event
from typing import Any

from ..core.config import *
from ..core.data_types import MentalState
from ..acquisition.sources import BrainSource, FakeEEGSource
from ..processing.preprocessor import Preprocessor
from ..processing.features import FeatureExtractor
from ..detection.relax_focus import RelaxFocusDetector
from ..detection.motor_imagery import MotorImageryDetector
from ..communication.unity_sender import UnitySender
from ..training.motor_imagery_trainer import run_calibration, train_motor_imagery_model


def run_realtime_processing(user_id: str, brain_source: BrainSource, 
                           udp_host: str, udp_port: int) -> None:
    """
    Main real-time processing loop
    
    This is the core function that continuously processes EEG data,
    extracts features, detects mental states, and sends updates to Unity.
    """
    logging.info("Starting real-time processing...")
    
    # Initialize components
    preprocessor = Preprocessor(brain_source.fs)
    feature_extractor = FeatureExtractor(brain_source.fs)
    
    # Load user profile for relax/focus detection
    profile_path = os.path.join(PROFILE_DIR, f"{user_id}.json")
    relax_focus_detector = RelaxFocusDetector(profile_path)
    
    # Initialize motor imagery detector
    mi_detector = MotorImageryDetector()
    
    # Initialize Unity sender
    unity_sender = UnitySender(udp_host, udp_port)
    
    # Processing state
    last_status_time = 0.0
    status_interval = 2.0  # Print status every 2 seconds
    
    # Graceful shutdown handler
    shutdown_event = Event()
    def signal_handler(signum, frame):
        logging.info("Shutdown signal received")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logging.info("Real-time processing started. Press Ctrl+C to stop.")
        
        while not shutdown_event.is_set():
            current_time = time.time()
            
            # Get EEG data
            raw_data = brain_source.get_data(WINDOW_SEC)
            if raw_data is None:
                time.sleep(0.1)  # Brief pause if no data
                continue
            
            # Create analysis window
            window = preprocessor.create_window(raw_data, current_time, CHANNELS)
            
            # Extract features
            powers = feature_extractor.extract_features(window, CHANNELS)
            
            # Detect relax/focus state
            relax_index, focus_index = relax_focus_detector.compute_indices(powers)
            state = relax_focus_detector.detect_state(powers, current_time)
            
            # Detect motor imagery
            mi_left, mi_right = mi_detector.detect(window, powers, CHANNELS)
            
            # Create mental state object
            mental_state = MentalState(
                timestamp=current_time,
                relax_index=relax_index,
                focus_index=focus_index,
                state=state,
                mi_prob_left=mi_left,
                mi_prob_right=mi_right,
                band_powers=powers,
                artifact=window.is_artifact
            )
            
            # Send to Unity (skip if artifact detected)
            if not window.is_artifact:
                unity_sender.send_state(mental_state)
            
            # Print status periodically
            if current_time - last_status_time > status_interval:
                print(f"State: {state:>7} | Relax: {relax_index:.2f} | Focus: {focus_index:.2f} | "
                      f"MI L/R: {mi_left:.2f}/{mi_right:.2f} | Artifact: {window.is_artifact}")
                last_status_time = current_time
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.05)  # ~20 Hz processing rate
            
    except Exception as e:
        logging.error(f"Processing error: {e}")
    finally:
        # Cleanup
        unity_sender.close()
        logging.info("Real-time processing stopped")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="EEG Bridge - Real-time BCI processing for Unity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate user profile
  python -m eeg_bridge --calibrate --user alice
  
  # Train motor imagery model from CSV
  python -m eeg_bridge --train-mi --csv data/mi_trials.csv
  
  # Run real-time processing
  python -m eeg_bridge --run --user alice
  
  # Test with synthetic data
  python -m eeg_bridge --run --fake --user test
        """
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--calibrate", action="store_true",
                           help="Run calibration procedure")
    mode_group.add_argument("--train-mi", action="store_true", 
                           help="Train motor imagery model")
    mode_group.add_argument("--run", action="store_true",
                           help="Run real-time processing")
    
    # Data source options
    parser.add_argument("--source", choices=["brainflow", "lsl"], default="brainflow",
                       help="EEG data source (default: brainflow)")
    parser.add_argument("--fake", action="store_true",
                       help="Use synthetic EEG data for testing")
    parser.add_argument("--serial-port", default=SERIAL_PORT,
                       help=f"Serial port for BrainFlow (default: {SERIAL_PORT})")
    parser.add_argument("--lsl-stream", default="EEG",
                       help="LSL stream name (default: EEG)")
    
    # User and file options
    parser.add_argument("--user", required=True,
                       help="User ID for profiles and models")
    parser.add_argument("--csv", 
                       help="CSV file path for motor imagery training")
    
    # Processing parameters
    parser.add_argument("--fs", type=int, default=FS_EXPECTED,
                       help=f"Sampling frequency (default: {FS_EXPECTED})")
    parser.add_argument("--notch", type=int, choices=[50, 60], default=NOTCH_HZ,
                       help=f"Notch filter frequency (default: {NOTCH_HZ})")
    parser.add_argument("--bandpass-low", type=float, default=BANDPASS[0],
                       help=f"Bandpass low frequency (default: {BANDPASS[0]})")
    parser.add_argument("--bandpass-high", type=float, default=BANDPASS[1],
                       help=f"Bandpass high frequency (default: {BANDPASS[1]})")
    
    # Communication options  
    parser.add_argument("--udp-host", default=UDP_HOST,
                       help=f"Unity UDP host (default: {UDP_HOST})")
    parser.add_argument("--udp-port", type=int, default=UDP_PORT,
                       help=f"Unity UDP port (default: {UDP_PORT})")
    
    # Logging
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Print startup info
    print("="*60)
    print("EEG Bridge - Real-time BCI Processing")
    print("="*60)
    
    # Update global config from args (modify the imported module)
    import eeg_bridge.core.config as config
    config.NOTCH_HZ = args.notch
    config.BANDPASS = (args.bandpass_low, args.bandpass_high)
    config.FS_EXPECTED = args.fs
    
    try:
        # Initialize data source
        if args.fake:
            logging.info("Using synthetic EEG data")
            fake_source = FakeEEGSource(args.fs)
            
            # Wrap fake source to match BrainSource interface
            class FakeSourceWrapper:
                def __init__(self, fake_source):
                    self.fake_source = fake_source
                    self.fs = fake_source.fs
                    self.is_connected = True
                
                def connect(self):
                    return True
                
                def get_data(self, duration_sec):
                    return self.fake_source.generate_window(duration_sec)
                
                def disconnect(self):
                    pass
            
            brain_source = FakeSourceWrapper(fake_source)
        else:
            brain_source = BrainSource(
                source_type=args.source,
                serial_port=args.serial_port,
                lsl_stream_name=args.lsl_stream
            )
            
            if not brain_source.connect():
                logging.error("Failed to connect to EEG source")
                return 1
        
        # Execute requested mode
        if args.calibrate:
            preprocessor = Preprocessor(brain_source.fs, args.notch, config.BANDPASS)
            feature_extractor = FeatureExtractor(brain_source.fs)
            success = run_calibration(args.user, brain_source, preprocessor, feature_extractor)
            return 0 if success else 1
            
        elif args.train_mi:
            if args.csv:
                success = train_motor_imagery_model("csv", csv_path=args.csv)
            else:
                success = train_motor_imagery_model("lsl")
            return 0 if success else 1
            
        elif args.run:
            run_realtime_processing(args.user, brain_source, args.udp_host, args.udp_port)
            return 0
            
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        return 0
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return 1
    finally:
        # Cleanup
        if 'brain_source' in locals() and hasattr(brain_source, 'disconnect'):
            brain_source.disconnect()


if __name__ == "__main__":
    sys.exit(main())
