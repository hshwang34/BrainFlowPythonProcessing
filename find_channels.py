#!/usr/bin/env python3
"""
Helper script to find BrainFlow channel indices for your OpenBCI setup

Run this script to see the actual EEG channel indices that BrainFlow
uses for your board. Use these indices to update the CHANNELS dictionary
in eeg_bridge.py.

Usage:
    python find_channels.py --board cyton-daisy
    python find_channels.py --board cyton
    python find_channels.py --list-boards
"""

import argparse
import sys

try:
    from brainflow.board_shim import BoardShim, BoardIds
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    print("ERROR: BrainFlow not installed. Install with: pip install brainflow")

def list_available_boards():
    """List all available BrainFlow boards"""
    print("Available BrainFlow boards:")
    print("-" * 40)
    
    # Common boards
    boards = {
        "cyton": BoardIds.CYTON_BOARD,
        "cyton-daisy": BoardIds.CYTON_DAISY_BOARD,
        "ganglion": BoardIds.GANGLION_BOARD,
        "synthetic": BoardIds.SYNTHETIC_BOARD,
        "streaming": BoardIds.STREAMING_BOARD,
    }
    
    for name, board_id in boards.items():
        try:
            eeg_channels = BoardShim.get_eeg_channels(board_id)
            sampling_rate = BoardShim.get_sampling_rate(board_id)
            print(f"{name:15} (ID: {board_id:2d}) - {len(eeg_channels):2d} EEG channels @ {sampling_rate:3.0f} Hz")
        except Exception as e:
            print(f"{name:15} (ID: {board_id:2d}) - Error: {e}")

def show_board_info(board_name):
    """Show detailed information for a specific board"""
    
    # Map board names to IDs
    board_map = {
        "cyton": BoardIds.CYTON_BOARD,
        "cyton-daisy": BoardIds.CYTON_DAISY_BOARD,
        "ganglion": BoardIds.GANGLION_BOARD,
        "synthetic": BoardIds.SYNTHETIC_BOARD,
    }
    
    if board_name not in board_map:
        print(f"ERROR: Unknown board '{board_name}'")
        print("Available boards:", list(board_map.keys()))
        return False
    
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
        
        print("\nSuggested Channel Mapping:")
        print("-" * 30)
        print("CHANNELS = {")
        
        # Create mapping based on standard 10-20 positions
        for i, ch_idx in enumerate(eeg_channels):
            if i < len(electrode_names):
                electrode = electrode_names[i]
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
                "Battery": BoardShim.get_battery_channel(board_id) if hasattr(BoardShim, 'get_battery_channel') else None,
            }
            
            print(f"\nOther channel indices:")
            print("-" * 25)
            for name, idx in other_channels.items():
                if idx is not None:
                    print(f"{name}: {idx}")
                    
        except Exception as e:
            print(f"Could not get additional channel info: {e}")
        
        print(f"\nTo use these channels in eeg_bridge.py:")
        print("1. Copy the CHANNELS dictionary above")
        print("2. Replace the placeholder CHANNELS in eeg_bridge.py")
        print("3. Make sure your electrode placement matches the names")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Could not get board information: {e}")
        return False

def main():
    if not BRAINFLOW_AVAILABLE:
        return 1
    
    parser = argparse.ArgumentParser(
        description="Find BrainFlow channel indices for OpenBCI boards"
    )
    
    parser.add_argument("--board", 
                       choices=["cyton", "cyton-daisy", "ganglion", "synthetic"],
                       help="Show channel info for specific board")
    parser.add_argument("--list-boards", action="store_true",
                       help="List all available boards")
    
    args = parser.parse_args()
    
    if args.list_boards:
        list_available_boards()
    elif args.board:
        show_board_info(args.board)
    else:
        print("Please specify --board <name> or --list-boards")
        print("Example: python find_channels.py --board cyton-daisy")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
