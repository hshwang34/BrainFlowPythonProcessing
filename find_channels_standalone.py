#!/usr/bin/env python3
"""
Standalone channel finder utility for EEG Bridge

This script helps find BrainFlow channel indices for your OpenBCI setup.
Can be run independently of the main EEG Bridge package.

Usage:
    python find_channels_standalone.py --board cyton-daisy
    python find_channels_standalone.py --board cyton
    python find_channels_standalone.py --list-boards
"""

import argparse
import sys

try:
    from eeg_bridge.utils.channel_finder import find_board_channels, list_available_boards
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False
    
    # Fallback implementation if package not installed
    try:
        from brainflow.board_shim import BoardShim, BoardIds
        BRAINFLOW_AVAILABLE = True
    except ImportError:
        BRAINFLOW_AVAILABLE = False
        print("ERROR: Neither EEG Bridge package nor BrainFlow is available.")
        print("Install with: pip install brainflow")
        sys.exit(1)


def fallback_list_boards():
    """Fallback board listing if package not available"""
    if not BRAINFLOW_AVAILABLE:
        print("BrainFlow not available")
        return
    
    boards = {
        "cyton": BoardIds.CYTON_BOARD,
        "cyton-daisy": BoardIds.CYTON_DAISY_BOARD,
        "ganglion": BoardIds.GANGLION_BOARD,
        "synthetic": BoardIds.SYNTHETIC_BOARD,
    }
    
    print("Available BrainFlow boards:")
    print("-" * 40)
    
    for name, board_id in boards.items():
        try:
            eeg_channels = BoardShim.get_eeg_channels(board_id)
            sampling_rate = BoardShim.get_sampling_rate(board_id)
            print(f"{name:15} (ID: {board_id:2d}) - {len(eeg_channels):2d} EEG channels @ {sampling_rate:3.0f} Hz")
        except Exception as e:
            print(f"{name:15} (ID: {board_id:2d}) - Error: {e}")


def fallback_show_board(board_name):
    """Fallback board info if package not available"""
    if not BRAINFLOW_AVAILABLE:
        print("BrainFlow not available")
        return
    
    board_map = {
        "cyton": BoardIds.CYTON_BOARD,
        "cyton-daisy": BoardIds.CYTON_DAISY_BOARD,
        "ganglion": BoardIds.GANGLION_BOARD,
        "synthetic": BoardIds.SYNTHETIC_BOARD,
    }
    
    if board_name not in board_map:
        print(f"ERROR: Unknown board '{board_name}'")
        print("Available boards:", list(board_map.keys()))
        return
    
    board_id = board_map[board_name]
    
    try:
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        
        print(f"Board Information: {board_name.upper()}")
        print("=" * 50)
        print(f"Board ID: {board_id}")
        print(f"Sampling Rate: {sampling_rate} Hz")
        print(f"EEG channel indices: {eeg_channels}")
        
        print("\nTo use in EEG Bridge, update CHANNELS in core/config.py")
        
    except Exception as e:
        print(f"Error getting board info: {e}")


def main():
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
        if PACKAGE_AVAILABLE:
            list_available_boards()
        else:
            fallback_list_boards()
    elif args.board:
        if PACKAGE_AVAILABLE:
            find_board_channels(args.board)
        else:
            fallback_show_board(args.board)
    else:
        print("Please specify --board <name> or --list-boards")
        print("Example: python find_channels_standalone.py --board cyton-daisy")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
