"""
Data input/output functions for Motor Imagery training

This module handles loading EEG data and event markers from various file formats.
It supports CSV files (common for exported data) and XDF files (Lab Streaming Layer format).
"""

import logging
import os
from typing import List, Tuple, Optional, Union
import numpy as np
import pandas as pd

# Optional dependency - only imported if needed
try:
    import pyxdf
    PYXDF_AVAILABLE = True
except ImportError:
    PYXDF_AVAILABLE = False


def load_csv(
    path: str, 
    fs: int, 
    ch_names: Optional[List[str]] = None,
    marker_col: str = "marker",
    timestamp_col: Optional[str] = None
) -> Tuple[np.ndarray, List[Tuple[int, str]], int, List[str]]:
    """
    Load EEG data and markers from a CSV file
    
    This function expects a CSV with EEG channels as columns and samples as rows.
    The marker column should contain "LEFT", "RIGHT", or empty/NaN for rest periods.
    
    Why CSV format? It's the most common export format from EEG analysis software
    and is human-readable for debugging. The downside is larger file sizes compared
    to binary formats.
    
    Args:
        path: Path to CSV file
        fs: Sampling frequency in Hz
        ch_names: List of channel names to load (if None, load all numeric columns)
        marker_col: Name of the marker/label column
        timestamp_col: Name of timestamp column (optional)
        
    Returns:
        Tuple of:
        - data: EEG data array [channels x samples]
        - markers: List of (sample_index, label) tuples for LEFT/RIGHT events
        - fs: Sampling frequency (same as input)
        - ch_names: List of channel names in the data array
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing or data format is invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    logging.info(f"Loading CSV data from: {path}")
    
    # Load the entire CSV file
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
    
    logging.info(f"CSV shape: {df.shape}")
    logging.info(f"CSV columns: {list(df.columns)}")
    
    # Validate marker column exists
    if marker_col not in df.columns:
        raise ValueError(f"Marker column '{marker_col}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Determine which columns are EEG channels
    if ch_names is None:
        # Auto-detect: use all numeric columns except marker and timestamp
        exclude_cols = [marker_col]
        if timestamp_col and timestamp_col in df.columns:
            exclude_cols.append(timestamp_col)
        
        # Find numeric columns that aren't excluded
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ch_names = [col for col in numeric_cols if col not in exclude_cols]
        
        if not ch_names:
            raise ValueError("No numeric EEG channels found in CSV")
        
        logging.info(f"Auto-detected {len(ch_names)} EEG channels: {ch_names}")
    else:
        # Validate that specified channels exist
        missing_channels = [ch for ch in ch_names if ch not in df.columns]
        if missing_channels:
            raise ValueError(f"Missing channels in CSV: {missing_channels}")
    
    # Extract EEG data - transpose to [channels x samples]
    eeg_data = df[ch_names].values.T
    logging.info(f"EEG data shape: {eeg_data.shape}")
    
    # Extract markers - find non-empty, non-NaN entries
    markers = []
    marker_series = df[marker_col]
    
    for idx, marker_value in enumerate(marker_series):
        # Skip NaN, None, or empty string markers
        if pd.isna(marker_value) or marker_value == "" or marker_value is None:
            continue
        
        # Convert to string and normalize case
        marker_str = str(marker_value).strip().upper()
        
        # Only keep LEFT and RIGHT markers
        if marker_str in ["LEFT", "RIGHT"]:
            markers.append((idx, marker_str))
    
    logging.info(f"Found {len(markers)} motor imagery markers")
    
    # Count markers by class
    left_count = sum(1 for _, label in markers if label == "LEFT")
    right_count = sum(1 for _, label in markers if label == "RIGHT")
    logging.info(f"Marker distribution: LEFT={left_count}, RIGHT={right_count}")
    
    if len(markers) == 0:
        raise ValueError("No LEFT or RIGHT markers found in data")
    
    if left_count == 0 or right_count == 0:
        logging.warning("Unbalanced classes detected - this may affect training performance")
    
    return eeg_data, markers, fs, ch_names


def load_xdf(
    path: str,
    stream_name: str = "EEG",
    marker_stream: str = "Markers"
) -> Tuple[np.ndarray, List[Tuple[int, str]], int, List[str]]:
    """
    Load EEG data and markers from an XDF file
    
    XDF (Extensible Data Format) is the native format for Lab Streaming Layer (LSL).
    It can contain multiple synchronized streams (EEG, markers, etc.) with precise timing.
    
    Why XDF? It's designed for real-time experiments and maintains precise timing
    relationships between different data streams. However, it requires the pyxdf library.
    
    Args:
        path: Path to XDF file
        stream_name: Name of the EEG stream to load
        marker_stream: Name of the marker stream to load
        
    Returns:
        Same format as load_csv: (data, markers, fs, ch_names)
        
    Raises:
        ImportError: If pyxdf is not installed
        FileNotFoundError: If XDF file doesn't exist
        ValueError: If required streams are not found
    """
    if not PYXDF_AVAILABLE:
        raise ImportError(
            "pyxdf is required for XDF file loading. Install with: pip install pyxdf"
        )
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"XDF file not found: {path}")
    
    logging.info(f"Loading XDF data from: {path}")
    
    # Load XDF file - this returns streams and header info
    try:
        streams, header = pyxdf.load_xdf(path)
    except Exception as e:
        raise ValueError(f"Failed to read XDF file: {e}")
    
    logging.info(f"Found {len(streams)} streams in XDF file")
    
    # Find EEG stream
    eeg_stream = None
    for stream in streams:
        if stream['info']['name'][0] == stream_name:
            eeg_stream = stream
            break
    
    if eeg_stream is None:
        available_streams = [s['info']['name'][0] for s in streams]
        raise ValueError(f"EEG stream '{stream_name}' not found. Available streams: {available_streams}")
    
    # Find marker stream
    marker_stream_data = None
    for stream in streams:
        if stream['info']['name'][0] == marker_stream:
            marker_stream_data = stream
            break
    
    if marker_stream_data is None:
        available_streams = [s['info']['name'][0] for s in streams]
        raise ValueError(f"Marker stream '{marker_stream}' not found. Available streams: {available_streams}")
    
    # Extract EEG data and metadata
    eeg_data = eeg_stream['time_series'].T  # Transpose to [channels x samples]
    eeg_timestamps = eeg_stream['time_stamps']
    fs = float(eeg_stream['info']['nominal_srate'][0])
    
    # Extract channel names
    try:
        ch_info = eeg_stream['info']['desc'][0]['channels'][0]['channel']
        if isinstance(ch_info, list):
            ch_names = [ch['label'][0] for ch in ch_info]
        else:
            # Single channel case
            ch_names = [ch_info['label'][0]]
    except (KeyError, IndexError):
        # Fallback to generic channel names
        ch_names = [f"Ch{i+1}" for i in range(eeg_data.shape[0])]
        logging.warning("Could not extract channel names from XDF, using generic names")
    
    logging.info(f"EEG data shape: {eeg_data.shape}, fs: {fs} Hz")
    logging.info(f"Channel names: {ch_names}")
    
    # Extract markers and convert timestamps to sample indices
    marker_timestamps = marker_stream_data['time_stamps']
    marker_labels = marker_stream_data['time_series']
    
    markers = []
    for timestamp, label_array in zip(marker_timestamps, marker_labels):
        # Handle different marker formats
        if isinstance(label_array, (list, np.ndarray)) and len(label_array) > 0:
            label = str(label_array[0]).strip().upper()
        else:
            label = str(label_array).strip().upper()
        
        # Only keep LEFT and RIGHT markers
        if label in ["LEFT", "RIGHT"]:
            # Convert timestamp to sample index
            # Find the closest EEG sample to this marker timestamp
            sample_idx = np.argmin(np.abs(eeg_timestamps - timestamp))
            markers.append((int(sample_idx), label))
    
    logging.info(f"Found {len(markers)} motor imagery markers")
    
    # Count markers by class
    left_count = sum(1 for _, label in markers if label == "LEFT")
    right_count = sum(1 for _, label in markers if label == "RIGHT")
    logging.info(f"Marker distribution: LEFT={left_count}, RIGHT={right_count}")
    
    if len(markers) == 0:
        raise ValueError("No LEFT or RIGHT markers found in XDF data")
    
    return eeg_data, markers, int(fs), ch_names


def resolve_motor_channels(
    ch_names: List[str],
    motor_ch_names: Optional[List[str]] = None,
    motor_indices: Optional[List[int]] = None
) -> Tuple[List[int], List[str]]:
    """
    Resolve motor cortex channel indices from names or explicit indices
    
    This function handles the common problem of channel selection: sometimes you know
    the channel names (e.g., "C3", "C4"), sometimes you only know the indices.
    It provides a flexible interface for both cases.
    
    Why motor channels? Motor imagery primarily affects the sensorimotor cortex,
    especially channels C3 (left motor cortex) and C4 (right motor cortex).
    Including Cz (central) can help capture shared motor activity.
    
    Args:
        ch_names: List of all available channel names
        motor_ch_names: Desired motor channel names (e.g., ["C3", "Cz", "C4"])
        motor_indices: Explicit channel indices to use instead
        
    Returns:
        Tuple of (indices, names) for the selected motor channels
        
    Raises:
        ValueError: If specified channels are not found or indices are invalid
    """
    if motor_indices is not None:
        # Use explicit indices
        invalid_indices = [i for i in motor_indices if i < 0 or i >= len(ch_names)]
        if invalid_indices:
            raise ValueError(f"Invalid channel indices: {invalid_indices}. Valid range: 0-{len(ch_names)-1}")
        
        selected_names = [ch_names[i] for i in motor_indices]
        logging.info(f"Using explicit motor channel indices: {motor_indices} -> {selected_names}")
        return motor_indices, selected_names
    
    elif motor_ch_names is not None:
        # Find indices by name
        indices = []
        missing_channels = []
        
        for name in motor_ch_names:
            try:
                idx = ch_names.index(name)
                indices.append(idx)
            except ValueError:
                missing_channels.append(name)
        
        if missing_channels:
            raise ValueError(f"Motor channels not found: {missing_channels}. Available channels: {ch_names}")
        
        logging.info(f"Resolved motor channels: {motor_ch_names} -> indices {indices}")
        return indices, motor_ch_names
    
    else:
        # Default: try to find C3, Cz, C4
        default_names = ["C3", "Cz", "C4"]
        found_indices = []
        found_names = []
        
        for name in default_names:
            try:
                idx = ch_names.index(name)
                found_indices.append(idx)
                found_names.append(name)
            except ValueError:
                logging.warning(f"Default motor channel '{name}' not found in data")
        
        if not found_indices:
            raise ValueError(f"No default motor channels found. Available channels: {ch_names}")
        
        logging.info(f"Using default motor channels: {found_names} -> indices {found_indices}")
        return found_indices, found_names
