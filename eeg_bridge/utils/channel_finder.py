"""
BrainFlow channel finder utility

This module provides functions to discover and map BrainFlow channel indices
for different OpenBCI board configurations.
"""

import logging
from typing import Dict, List, Optional

# Optional import with fallback
try:
    from brainflow.board_shim import BoardShim, BoardIds
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    logging.warning("BrainFlow not available")


def list_available_boards() -> Dict[str, int]:
    """
    List all available BrainFlow boards
    
    Returns:
        Dict[str, int]: Mapping of board names to board IDs
    """
    if not BRAINFLOW_AVAILABLE:
        logging.error("BrainFlow not installed. Install with: pip install brainflow")
        return {}
    
    # Common boards
    boards = {
        "cyton": BoardIds.CYTON_BOARD,
        "cyton-daisy": BoardIds.CYTON_DAISY_BOARD,
        "ganglion": BoardIds.GANGLION_BOARD,
        "synthetic": BoardIds.SYNTHETIC_BOARD,
        "streaming": BoardIds.STREAMING_BOARD,
    }
    
    print("Available BrainFlow boards:")
    print("-" * 40)
    
    available_boards = {}
    for name, board_id in boards.items():
        try:
            eeg_channels = BoardShim.get_eeg_channels(board_id)
            sampling_rate = BoardShim.get_sampling_rate(board_id)
            print(f"{name:15} (ID: {board_id:2d}) - {len(eeg_channels):2d} EEG channels @ {sampling_rate:3.0f} Hz")
            available_boards[name] = board_id
        except Exception as e:
            print(f"{name:15} (ID: {board_id:2d}) - Error: {e}")
    
    return available_boards


def find_board_channels(board_name: str) -> Optional[Dict[str, int]]:
    """
    Find channel mapping for a specific board
    
    Args:
        board_name: Name of the board ("cyton", "cyton-daisy", etc.)
        
    Returns:
        Optional[Dict[str, int]]: Channel name to index mapping, or None if failed
    """
    if not BRAINFLOW_AVAILABLE:
        logging.error("BrainFlow not installed. Install with: pip install brainflow")
        return None
    
    # Map board names to IDs
    board_map = {
        "cyton": BoardIds.CYTON_BOARD,
        "cyton-daisy": BoardIds.CYTON_DAISY_BOARD,
        "ganglion": BoardIds.GANGLION_BOARD,
        "synthetic": BoardIds.SYNTHETIC_BOARD,
    }
    
    if board_name not in board_map:
        logging.error(f"Unknown board '{board_name}'")
        logging.info(f"Available boards: {list(board_map.keys())}")
        return None
    
    board_id = board_map[board_name]
    
    try:
        print(f"Board Information: {board_name.upper()}")
        print("=" * 50)
        
        # Get channel information
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        
        print(f"Board ID: {board_id}")
        print(f"Sampling Rate: {sampling_rate} Hz")
        print(f"Number of EEG channels: {len(eeg_channels)}")
        print(f"EEG channel indices: {eeg_channels}")
        
        # Standard electrode positions for common boards
        if board_name == "cyton":
            # 8-channel Cyton board
            electrode_names = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]
        elif board_name == "cyton-daisy":
            # 16-channel Cyton + Daisy
            electrode_names = [
                "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",  # Cyton
                "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"     # Daisy
            ]
        else:
            electrode_names = [f"Ch{i+1}" for i in range(len(eeg_channels))]
        
        # Create channel mapping
        channel_mapping = {}
        for i, ch_idx in enumerate(eeg_channels):
            if i < len(electrode_names):
                electrode = electrode_names[i]
                channel_mapping[electrode] = ch_idx
        
        print("\nSuggested Channel Mapping:")
        print("-" * 30)
        print("CHANNELS = {")
        
        for electrode, ch_idx in channel_mapping.items():
            print(f'    "{electrode}": {ch_idx},')
        
        print("}")
        
        # Show specific channels needed for EEG Bridge
        print("\nChannels needed for EEG Bridge:")
        print("-" * 35)
        
        needed_channels = {
            "Occipital (for alpha/relaxation)": ["O1", "O2", "Pz"],
            "Frontal (for beta/focus)": ["Fz", "F3", "F4", "Fp1", "Fp2"],
            "Central (for motor imagery)": ["C3", "C4", "Cz"],
        }
        
        for category, channels in needed_channels.items():
            print(f"\n{category}:")
            for ch in channels:
                if ch in electrode_names:
                    idx = electrode_names.index(ch)
                    if idx < len(eeg_channels):
                        print(f"  {ch}: index {eeg_channels[idx]} ✓")
                    else:
                        print(f"  {ch}: not available ✗")
                else:
                    print(f"  {ch}: not available ✗")
        
        # Additional board-specific information
        try:
            other_channels = {
                "Timestamp": BoardShim.get_timestamp_channel(board_id),
                "Marker": BoardShim.get_marker_channel(board_id),
            }
            
            print(f"\nOther channel indices:")
            print("-" * 25)
            for name, idx in other_channels.items():
                if idx is not None:
                    print(f"{name}: {idx}")
                    
        except Exception as e:
            logging.debug(f"Could not get additional channel info: {e}")
        
        print(f"\nTo use these channels in eeg_bridge:")
        print("1. Copy the CHANNELS dictionary above")
        print("2. Replace the placeholder CHANNELS in core/config.py")
        print("3. Make sure your electrode placement matches the names")
        
        return channel_mapping
        
    except Exception as e:
        logging.error(f"Could not get board information: {e}")
        return None
