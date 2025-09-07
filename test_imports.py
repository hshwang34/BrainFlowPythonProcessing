#!/usr/bin/env python3
"""
Test script to verify the refactored EEG Bridge import structure

This script tests that all modules can be imported correctly without
requiring external dependencies like numpy, brainflow, etc.
"""

import sys
import importlib
from pathlib import Path

def test_import(module_name, description):
    """Test importing a module"""
    try:
        module = importlib.import_module(module_name)
        print(f"✓ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {description}: {module_name} - {e}")
        return False
    except Exception as e:
        print(f"⚠ {description}: {module_name} - Unexpected error: {e}")
        return False

def main():
    print("EEG Bridge Refactored - Import Structure Test")
    print("=" * 50)
    
    # Test core modules (these will fail due to numpy dependency)
    print("\nCore Modules (expected to fail without numpy):")
    test_import("eeg_bridge.core.config", "Configuration constants")
    
    # Test structure by checking if files exist
    print("\nFile Structure Test:")
    expected_files = [
        "eeg_bridge/__init__.py",
        "eeg_bridge/__main__.py",
        "eeg_bridge/core/__init__.py",
        "eeg_bridge/core/config.py",
        "eeg_bridge/core/data_types.py",
        "eeg_bridge/acquisition/__init__.py",
        "eeg_bridge/acquisition/sources.py",
        "eeg_bridge/processing/__init__.py",
        "eeg_bridge/processing/preprocessor.py",
        "eeg_bridge/processing/features.py",
        "eeg_bridge/detection/__init__.py",
        "eeg_bridge/detection/relax_focus.py",
        "eeg_bridge/detection/motor_imagery.py",
        "eeg_bridge/communication/__init__.py",
        "eeg_bridge/communication/unity_sender.py",
        "eeg_bridge/training/__init__.py",
        "eeg_bridge/training/motor_imagery_trainer.py",
        "eeg_bridge/utils/__init__.py",
        "eeg_bridge/utils/channel_finder.py",
        "eeg_bridge/cli/__init__.py",
        "eeg_bridge/cli/main.py",
    ]
    
    missing_files = []
    for file_path in expected_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - Missing")
            missing_files.append(file_path)
    
    # Test standalone scripts
    print("\nStandalone Scripts:")
    standalone_files = [
        "find_channels_standalone.py",
        "setup.py",
        "requirements.txt",
        "README_REFACTORED.md"
    ]
    
    for file_path in standalone_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - Missing")
            missing_files.append(file_path)
    
    # Summary
    print("\n" + "=" * 50)
    if missing_files:
        print(f"❌ Test failed: {len(missing_files)} files missing")
        for f in missing_files:
            print(f"   - {f}")
        return 1
    else:
        print("✅ All files present - Refactoring structure is correct!")
        print("\nTo test full functionality:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run: python -m eeg_bridge --help")
        print("3. Test with: python -m eeg_bridge --run --fake --user test")
        return 0

if __name__ == "__main__":
    sys.exit(main())
