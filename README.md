# EEG Bridge - Real-time BCI Processing for Unity

A production-quality Python script that connects to OpenBCI Cyton+Daisy or LSL EEG streams, processes brain signals in real-time, detects mental states (relaxation/focus/motor imagery), and streams results to Unity applications via UDP.

## Features

- **Multi-source EEG acquisition**: OpenBCI via BrainFlow, LSL streams, or synthetic data
- **Real-time preprocessing**: Band-pass filtering, notch filtering, artifact detection
- **Mental state detection**: 
  - Relaxation vs Focus classification with user calibration
  - Motor imagery detection (left/right hand) using CSP+LDA or heuristic fallback
- **Unity integration**: JSON over UDP for seamless game/app integration
- **Comprehensive CLI**: Calibration, training, and real-time modes

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Hardware setup (for OpenBCI):**
   - Connect your OpenBCI Cyton+Daisy board
   - Note the COM port (Windows) or device path (Linux/Mac)
   - Update `SERIAL_PORT` in the script configuration section

3. **Configure channels:**
   - Update the `CHANNELS` dictionary with your actual electrode positions
   - Use this helper to find BrainFlow channel indices:
     ```python
     from brainflow import BoardShim, BoardIds
     print(BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD))
     ```

## Quick Start

### 1. Test with synthetic data
```bash
# Test Unity communication with fake EEG data
python eeg_bridge.py --run --fake --user test
```

### 2. Calibrate your profile
```bash
# Create personalized thresholds (requires real EEG)
python eeg_bridge.py --calibrate --user yourname --source brainflow
```

### 3. Real-time processing
```bash
# Run with your calibrated profile
python eeg_bridge.py --run --user yourname --source brainflow
```

## Configuration

Edit these constants in `eeg_bridge.py` for your setup:

```python
# Hardware
SERIAL_PORT = "COM3"              # Your OpenBCI port
FS_EXPECTED = 250                 # Sampling rate
NOTCH_HZ = 60                     # 60Hz (US) or 50Hz (EU)

# Channels - UPDATE THESE INDICES
CHANNELS = {
    "O1": 0,  # TODO: Replace with actual BrainFlow indices
    "O2": 1,  # TODO: Replace with actual BrainFlow indices  
    "Pz": 2,  # TODO: Replace with actual BrainFlow indices
    # ... etc
}

# Unity communication
UDP_HOST = "127.0.0.1"
UDP_PORT = 5005
```

## Usage Examples

### Calibration Mode
Guides you through 90s relaxation + 90s focus sessions to learn your personal brain patterns:

```bash
python eeg_bridge.py --calibrate --user alice --source brainflow --serial-port COM3
```

### Training Motor Imagery
Train a machine learning model for left/right hand imagery:

```bash
# From CSV data
python eeg_bridge.py --train-mi --user alice --csv data/motor_imagery.csv

# From LSL streams (live collection)
python eeg_bridge.py --train-mi --user alice --source lsl
```

### Real-time Processing
Stream brain states to Unity:

```bash
# With OpenBCI
python eeg_bridge.py --run --user alice --source brainflow

# With LSL
python eeg_bridge.py --run --user alice --source lsl

# With custom Unity address
python eeg_bridge.py --run --user alice --udp-host 192.168.1.100 --udp-port 8080
```

## Unity Integration

The script sends JSON messages via UDP with this format:

```json
{
  "t": 1693948123.42,
  "alpha": 12.4,
  "beta": 7.8, 
  "theta": 4.1,
  "relax_index": 1.35,
  "focus_index": 0.63,
  "state": "relax",
  "mi_prob_left": 0.18,
  "mi_prob_right": 0.72,
  "artifact": false
}
```

### Unity C# Example

```csharp
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Newtonsoft.Json;

public class EEGReceiver : MonoBehaviour 
{
    private UdpClient udpClient;
    
    void Start() 
    {
        udpClient = new UdpClient(5005);
        udpClient.BeginReceive(ReceiveCallback, null);
    }
    
    void ReceiveCallback(IAsyncResult ar) 
    {
        IPEndPoint ip = new IPEndPoint(IPAddress.Any, 5005);
        byte[] bytes = udpClient.EndReceive(ar, ref ip);
        string message = Encoding.UTF8.GetString(bytes);
        
        var eegData = JsonConvert.DeserializeObject<EEGData>(message);
        
        // Use the data in your game
        Debug.Log($"State: {eegData.state}, Focus: {eegData.focus_index}");
        
        udpClient.BeginReceive(ReceiveCallback, null);
    }
}

[Serializable]
public class EEGData 
{
    public float t;
    public float alpha, beta, theta;
    public float relax_index, focus_index;
    public string state;
    public float mi_prob_left, mi_prob_right;
    public bool artifact;
}
```

## File Structure

After running, the script creates:

```
profiles/
├── alice.json          # User calibration profile
└── bob.json           # Another user's profile

models/
└── mi_csp_lda.joblib  # Trained motor imagery model
```

## Troubleshooting

### Connection Issues
- **BrainFlow**: Check COM port, ensure board is powered, close other software
- **LSL**: Verify stream name, check network connectivity

### Poor Detection
- **Calibration**: Ensure good electrode contact, follow instructions carefully
- **Artifacts**: Check for loose electrodes, muscle tension, eye blinks
- **Thresholds**: Re-run calibration in your typical usage environment

### Unity Communication
- **Firewall**: Ensure UDP port is open
- **Network**: Check IP address if Unity is on different machine
- **JSON**: Verify Unity JSON parsing matches the message format

## Advanced Features

### Custom Frequency Bands
Modify `FREQ_BANDS` dictionary to use different frequency ranges:

```python
FREQ_BANDS = {
    "theta": (4, 7),
    "alpha": (8, 12), 
    "beta": (13, 30),
    "gamma": (30, 45)  # Add gamma band
}
```

### Electrode Configurations
Update `ELECTRODE_GROUPS` for different montages:

```python
ELECTRODE_GROUPS = {
    "alpha": ["O1", "O2", "Pz"],      # Occipital
    "beta": ["F3", "F4", "Fz"],       # Frontal  
    "mi": ["C3", "C4", "Cz"],         # Central
}
```

## License

This code is provided as-is for educational and research purposes. See LICENSE file for details.

## Contributing

This is a comprehensive single-file implementation. For modifications:

1. Test changes with `--fake` mode first
2. Validate with real EEG hardware
3. Ensure Unity integration still works
4. Update documentation as needed

## Support

For issues:
1. Check hardware connections and driver installation
2. Verify Python dependencies are installed correctly
3. Test with synthetic data (`--fake`) to isolate hardware issues
4. Review logs with `--verbose` flag for detailed debugging
