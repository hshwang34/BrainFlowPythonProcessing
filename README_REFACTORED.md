# EEG Bridge - Modular BCI Processing Package

A professionally structured Python package for real-time EEG signal processing, mental state detection, and Unity integration. This is the refactored version of the original single-file implementation, organized into logical modules for better maintainability and extensibility.

## 📁 Project Structure

```
eeg_bridge/
├── __init__.py                    # Main package imports
├── __main__.py                    # Package entry point (python -m eeg_bridge)
├── core/                          # Core data types and configuration
│   ├── __init__.py
│   ├── data_types.py             # EEGWindow, BandPowers, MentalState
│   └── config.py                 # All configuration constants
├── acquisition/                   # EEG data sources
│   ├── __init__.py
│   └── sources.py                # BrainSource, FakeEEGSource
├── processing/                    # Signal processing
│   ├── __init__.py
│   ├── preprocessor.py           # Filtering and artifact detection
│   └── features.py               # Frequency domain feature extraction
├── detection/                     # Mental state detectors
│   ├── __init__.py
│   ├── relax_focus.py            # Relaxation vs focus classification
│   └── motor_imagery.py          # Motor imagery detection (CSP+LDA + heuristic)
├── communication/                 # External interfaces
│   ├── __init__.py
│   └── unity_sender.py           # UDP JSON messaging to Unity
├── training/                      # Model training and calibration
│   ├── __init__.py
│   └── motor_imagery_trainer.py  # Calibration and ML model training
├── utils/                         # Utility functions
│   ├── __init__.py
│   └── channel_finder.py         # BrainFlow channel discovery
└── cli/                          # Command line interface
    ├── __init__.py
    └── main.py                   # CLI parser and main processing loops

# Additional files
find_channels_standalone.py       # Standalone channel finder utility
setup.py                          # Package installation script
requirements.txt                  # Dependencies
README_REFACTORED.md              # This documentation
```

## 🚀 Installation & Setup

### Option 1: Development Installation
```bash
# Clone and install in development mode
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

### Option 2: Direct Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python -m eeg_bridge --help
```

## 📋 Usage Examples

### Find Your Channel Mapping
```bash
# Using the standalone utility
python find_channels_standalone.py --board cyton-daisy

# Or using the package
python -m eeg_bridge.utils.channel_finder --board cyton-daisy
```

### Calibration
```bash
python -m eeg_bridge --calibrate --user alice --source brainflow --serial-port COM3
```

### Real-time Processing
```bash
# With real hardware
python -m eeg_bridge --run --user alice --source brainflow

# With synthetic data for testing
python -m eeg_bridge --run --fake --user test --udp-host 127.0.0.1 --udp-port 5005
```

### Training Motor Imagery
```bash
python -m eeg_bridge --train-mi --csv data/motor_imagery.csv --user alice
```

## 🔧 Configuration

Edit `eeg_bridge/core/config.py` to customize:

```python
# Hardware settings
SERIAL_PORT = "COM3"              # Your OpenBCI port
FS_EXPECTED = 250                 # Sampling rate
NOTCH_HZ = 60                     # Power line frequency

# Channel mapping (update with your actual indices)
CHANNELS = {
    "O1": 0, "O2": 1, "Pz": 2,    # Occipital for alpha
    "Fz": 3, "F3": 4, "F4": 5,    # Frontal for beta  
    "C3": 6, "C4": 7, "Cz": 8,    # Central for motor imagery
}

# Processing parameters
WINDOW_SEC = 1.0                  # Analysis window size
EMA_TAU_SEC = 4.0                 # Smoothing time constant
HOLD_MS = 800                     # State change hold time

# Unity communication
UDP_HOST = "127.0.0.1"
UDP_PORT = 5005
```

## 🧩 Module Overview

### Core Modules (`core/`)
- **`data_types.py`**: Fundamental data structures (EEGWindow, BandPowers, MentalState)
- **`config.py`**: All configuration constants and user-customizable parameters

### Data Acquisition (`acquisition/`)
- **`sources.py`**: 
  - `BrainSource`: Unified interface for BrainFlow and LSL
  - `FakeEEGSource`: Synthetic EEG generation for testing

### Signal Processing (`processing/`)
- **`preprocessor.py`**: 
  - Digital filtering (notch, bandpass)
  - Artifact detection using z-score thresholds
  - EEG window creation
- **`features.py`**:
  - Welch PSD computation
  - Frequency band power extraction (theta, alpha, beta, mu)

### Mental State Detection (`detection/`)
- **`relax_focus.py`**:
  - Personalized calibration-based thresholds
  - RelaxIndex = Alpha/(Theta+Beta)
  - FocusIndex = Beta/Alpha
  - EMA smoothing and hysteresis
- **`motor_imagery.py`**:
  - CSP+LDA model loading and prediction
  - Mu ERD heuristic fallback
  - Left/right hand imagery probabilities

### Communication (`communication/`)
- **`unity_sender.py`**: UDP JSON messaging to Unity applications

### Training (`training/`)
- **`motor_imagery_trainer.py`**:
  - User calibration procedures (90s relax + 90s focus)
  - ML model training frameworks (placeholders for CSV/LSL data)

### Utilities (`utils/`)
- **`channel_finder.py`**: BrainFlow board discovery and channel mapping

### CLI (`cli/`)
- **`main.py`**: Command-line interface, argument parsing, main processing loops

## 🎮 Unity Integration

The package maintains the same UDP JSON protocol:

```json
{
  "t": 1693948123.42,
  "alpha": 12.4, "beta": 7.8, "theta": 4.1,
  "relax_index": 1.35, "focus_index": 0.63,
  "state": "relax",
  "mi_prob_left": 0.18, "mi_prob_right": 0.72,
  "artifact": false
}
```

Use the same Unity C# scripts from the original implementation.

## 🔍 Key Improvements

### Maintainability
- **Separation of Concerns**: Each module has a single responsibility
- **Clear Dependencies**: Import structure shows relationships
- **Modular Testing**: Each component can be tested independently

### Extensibility  
- **Plugin Architecture**: Easy to add new detectors or data sources
- **Configuration Management**: Centralized settings in `config.py`
- **Interface Consistency**: All components follow similar patterns

### Development Experience
- **Import Clarity**: `from eeg_bridge.detection import RelaxFocusDetector`
- **IDE Support**: Better autocomplete and navigation
- **Documentation**: Each module has focused docstrings

### Professional Structure
- **Package Installation**: Proper `setup.py` for pip installation
- **Entry Points**: Console scripts for easy CLI access
- **Namespace Management**: Clean package-level imports

## 🧪 Testing the Refactored Code

<function_calls>
<invoke name="todo_write">
<parameter name="merge">true
