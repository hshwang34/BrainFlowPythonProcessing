"""
EEG data acquisition sources

This module provides unified interfaces for different EEG data sources including
BrainFlow (OpenBCI), LSL streams, and synthetic data generation for testing.
"""

import logging
import time
from typing import Optional
import numpy as np

from ..core.config import SERIAL_PORT, FS_EXPECTED

# Optional imports with fallbacks
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    logging.warning("BrainFlow not available - use LSL or fake mode instead")

try:
    import pylsl
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    logging.warning("pylsl not available - BrainFlow and fake modes only")


class BrainSource:
    """
    Unified interface for EEG data acquisition from BrainFlow or LSL
    
    This class handles the complexity of different data sources and provides
    a consistent interface for the main processing loop.
    """
    
    def __init__(self, source_type: str = "brainflow", serial_port: str = SERIAL_PORT, 
                 lsl_stream_name: str = "EEG"):
        self.source_type = source_type
        self.serial_port = serial_port
        self.lsl_stream_name = lsl_stream_name
        self.board = None
        self.lsl_inlet = None
        self.fs = FS_EXPECTED
        self.eeg_channels = []
        self.is_connected = False
        
    def connect(self) -> bool:
        """
        Establish connection to the EEG data source
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.source_type == "brainflow":
                return self._connect_brainflow()
            elif self.source_type == "lsl":
                return self._connect_lsl()
            else:
                logging.error(f"Unknown source type: {self.source_type}")
                return False
        except Exception as e:
            logging.error(f"Failed to connect to {self.source_type}: {e}")
            return False
    
    def _connect_brainflow(self) -> bool:
        """Connect to OpenBCI via BrainFlow"""
        if not BRAINFLOW_AVAILABLE:
            logging.error("BrainFlow not available. Install with: pip install brainflow")
            return False
            
        try:
            # Configure BrainFlow parameters
            params = BrainFlowInputParams()
            params.serial_port = self.serial_port
            
            # Use Cyton+Daisy board (16 channels)
            board_id = BoardIds.CYTON_DAISY_BOARD
            self.board = BoardShim(board_id, params)
            
            # Get board info
            self.eeg_channels = BoardShim.get_eeg_channels(board_id)
            self.fs = BoardShim.get_sampling_rate(board_id)
            
            logging.info(f"BrainFlow EEG channels: {self.eeg_channels}")
            logging.info(f"Sampling rate: {self.fs} Hz")
            
            # Prepare and start session
            self.board.prepare_session()
            self.board.start_stream()
            
            self.is_connected = True
            logging.info(f"Connected to OpenBCI on {self.serial_port}")
            return True
            
        except Exception as e:
            logging.error(f"BrainFlow connection failed: {e}")
            logging.error("Hint: Check COM port, ensure board is on, and no other software is using it")
            return False
    
    def _connect_lsl(self) -> bool:
        """Connect to LSL EEG stream"""
        if not LSL_AVAILABLE:
            logging.error("pylsl not available. Install with: pip install pylsl")
            return False
            
        try:
            # Look for EEG streams
            logging.info(f"Looking for LSL stream: {self.lsl_stream_name}")
            streams = pylsl.resolve_stream('name', self.lsl_stream_name)
            
            if not streams:
                # Try generic EEG type
                streams = pylsl.resolve_stream('type', 'EEG')
                
            if not streams:
                logging.error("No LSL EEG streams found")
                return False
                
            # Connect to first available stream
            stream_info = streams[0]
            self.lsl_inlet = pylsl.StreamInlet(stream_info)
            
            # Get stream info
            self.fs = stream_info.nominal_srate()
            n_channels = stream_info.channel_count()
            self.eeg_channels = list(range(n_channels))
            
            logging.info(f"Connected to LSL stream: {stream_info.name()}")
            logging.info(f"Channels: {n_channels}, Sample rate: {self.fs} Hz")
            
            self.is_connected = True
            return True
            
        except Exception as e:
            logging.error(f"LSL connection failed: {e}")
            return False
    
    def get_data(self, duration_sec: float) -> Optional[np.ndarray]:
        """
        Get EEG data for specified duration
        
        Args:
            duration_sec: Duration of data to retrieve (seconds)
            
        Returns:
            np.ndarray: EEG data (channels x samples) or None if failed
        """
        if not self.is_connected:
            return None
            
        try:
            if self.source_type == "brainflow":
                return self._get_brainflow_data(duration_sec)
            elif self.source_type == "lsl":
                return self._get_lsl_data(duration_sec)
        except Exception as e:
            logging.error(f"Failed to get data: {e}")
            return None
    
    def _get_brainflow_data(self, duration_sec: float) -> Optional[np.ndarray]:
        """Get data from BrainFlow"""
        n_samples = int(duration_sec * self.fs)
        
        # Wait for enough data to accumulate
        time.sleep(duration_sec)
        
        # Get all available data
        data = self.board.get_board_data()
        
        if data.shape[1] < n_samples:
            logging.warning(f"Insufficient data: got {data.shape[1]}, needed {n_samples}")
            return None
            
        # Extract EEG channels and most recent samples
        eeg_data = data[self.eeg_channels, -n_samples:]
        return eeg_data
    
    def _get_lsl_data(self, duration_sec: float) -> Optional[np.ndarray]:
        """Get data from LSL"""
        n_samples = int(duration_sec * self.fs)
        
        # Collect samples
        samples = []
        for _ in range(n_samples):
            sample, _ = self.lsl_inlet.pull_sample(timeout=1.0)
            if sample is None:
                logging.warning("LSL timeout - no data received")
                return None
            samples.append(sample)
        
        # Convert to numpy array and transpose to (channels x samples)
        eeg_data = np.array(samples).T
        return eeg_data
    
    def disconnect(self):
        """Clean disconnect from data source"""
        try:
            if self.board is not None:
                self.board.stop_stream()
                self.board.release_session()
                logging.info("BrainFlow disconnected")
                
            if self.lsl_inlet is not None:
                self.lsl_inlet.close_stream()
                logging.info("LSL disconnected")
                
        except Exception as e:
            logging.error(f"Disconnect error: {e}")
        finally:
            self.is_connected = False


class FakeEEGSource:
    """
    Generate synthetic EEG data for testing Unity integration
    
    This creates realistic-looking EEG with controllable alpha/beta patterns
    and motor imagery signatures for development and testing.
    """
    
    def __init__(self, fs: float = FS_EXPECTED, n_channels: int = 16):
        self.fs = fs
        self.n_channels = n_channels
        self.time = 0.0
        self.state_cycle_time = 10.0  # Switch states every 10 seconds
        self.mi_cycle_time = 8.0      # Switch MI every 8 seconds
        
    def generate_window(self, duration_sec: float) -> np.ndarray:
        """
        Generate synthetic EEG window
        
        Args:
            duration_sec: Duration of data to generate
            
        Returns:
            np.ndarray: Synthetic EEG data (channels x samples)
        """
        n_samples = int(duration_sec * self.fs)
        t = np.linspace(self.time, self.time + duration_sec, n_samples)
        
        # Base EEG with 1/f noise
        data = np.random.randn(self.n_channels, n_samples) * 10
        
        # Add frequency components
        for ch in range(self.n_channels):
            # Add alpha rhythm (8-12 Hz) - stronger in occipital
            if ch in [0, 1, 2]:  # O1, O2, Pz
                alpha_amp = 15 + 10 * np.sin(2 * np.pi * self.time / self.state_cycle_time)
                data[ch, :] += alpha_amp * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
            
            # Add beta rhythm (13-30 Hz) - stronger in frontal
            if ch in [3, 4, 5]:  # Fz, F3, F4
                beta_amp = 8 + 6 * np.cos(2 * np.pi * self.time / self.state_cycle_time)
                data[ch, :] += beta_amp * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
            
            # Add mu rhythm modulation for motor imagery (C3, C4)
            if ch in [6, 7]:  # C3, C4
                mu_phase = 2 * np.pi * self.time / self.mi_cycle_time
                if ch == 6:  # C3 - left motor cortex
                    mu_amp = 12 - 4 * np.sin(mu_phase)  # ERD during right imagery
                else:  # C4 - right motor cortex  
                    mu_amp = 12 - 4 * np.cos(mu_phase)  # ERD during left imagery
                data[ch, :] += mu_amp * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
        
        self.time += duration_sec
        return data
