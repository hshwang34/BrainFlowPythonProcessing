# Motor Imagery EEG Classifier Training

A comprehensive, production-quality Python toolkit for training CSP+LDA/SVM classifiers on motor imagery EEG data. Designed for researchers and developers working with brain-computer interfaces.

## 🎯 **Features**

- **Multiple Data Sources**: CSV, XDF, and synthetic data generation
- **Robust Preprocessing**: Notch filtering, band-pass filtering, artifact rejection
- **Advanced Feature Extraction**: CSP and Filter Bank CSP (FBCSP)
- **Multiple Classifiers**: LDA and SVM with probability estimates
- **Cross-Validation**: Stratified k-fold with comprehensive evaluation
- **Visualization**: Confusion matrix plots and performance metrics
- **Real-time Preview**: Test trained models on new data
- **Comprehensive Logging**: Detailed progress and debugging information

## 📁 **Project Structure**

```
mi_training/
├── __init__.py              # Package initialization
├── config.py                # Configuration dataclass with defaults
├── data_io.py               # Data loading (CSV, XDF, channel resolution)
├── preprocessing.py         # Filtering, artifact rejection
├── epoching.py              # Epoch extraction, baseline correction
├── features.py              # CSP, FBCSP feature extraction
├── models.py                # Pipeline building, cross-validation
├── utils.py                 # Logging, plotting, metadata handling
└── fake_data.py             # Synthetic EEG data generation

train.py                     # Main training script
runtime_preview.py           # Model testing and real-time preview
requirements_mi.txt          # Python dependencies
README_MI_Training.md        # This documentation
```

## 🚀 **Quick Start**

### 1. Installation

```bash
# Install dependencies
pip install -r requirements_mi.txt

# Verify installation
python -c "import mne, sklearn, numpy; print('✓ All dependencies installed')"
```

### 2. Test with Synthetic Data

```bash
# Train a classifier on synthetic motor imagery data
python train.py --fake --user demo

# Preview the trained model
python runtime_preview.py --model models/mi_lda_demo.joblib --fake
```

### 3. Train on Real Data

```bash
# Train on CSV data
python train.py --csv data/motor_imagery.csv --user alice

# Train on XDF data  
python train.py --xdf recording.xdf --user bob
```

## 📊 **Data Format Requirements**

### CSV Format
```csv
timestamp,C3,Cz,C4,F3,F4,marker
0.000,12.3,-5.1,8.7,15.2,-3.4,
0.004,11.8,-4.9,9.1,14.7,-3.8,
1.000,10.2,-6.3,7.9,13.1,-4.2,LEFT
1.004,9.8,-6.1,8.3,12.9,-4.0,
...
5.000,13.1,-5.7,8.8,15.8,-3.6,RIGHT
```

**Requirements:**
- EEG channels as numeric columns (microvolts)
- `marker` column with "LEFT", "RIGHT", or empty for rest
- Optional `timestamp` column
- Sampling rate must be specified via `--fs` parameter

### XDF Format
- EEG stream with channel names and sampling rate
- Marker stream with "LEFT"/"RIGHT" event labels
- Automatic timestamp synchronization

## ⚙️ **Configuration**

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fs` | 250 | Sampling frequency (Hz) |
| `--notch` | 60 | Notch filter frequency (50/60 Hz) |
| `--bp-low` | 8.0 | Band-pass low cutoff (Hz) |
| `--bp-high` | 30.0 | Band-pass high cutoff (Hz) |
| `--motor-channels` | C3 Cz C4 | Motor cortex channels |
| `--csp-components` | 6 | Number of CSP components |
| `--classifier` | lda | Classifier type (lda/svm) |
| `--cv-folds` | 5 | Cross-validation folds |

### Advanced Configuration

```bash
# Filter Bank CSP with custom frequency bands
python train.py --csv data.csv --user test \
  --fbcsp-bands "8,12" "12,16" "16,20" "20,26" "26,30"

# SVM classifier with custom parameters
python train.py --csv data.csv --user test \
  --classifier svm --svm-c 10.0 --csp-components 8

# Strict artifact rejection
python train.py --csv data.csv --user test \
  --ptp-threshold 100.0 --z-threshold 3.0
```

## 🧠 **Algorithm Details**

### Common Spatial Patterns (CSP)
CSP finds spatial filters that maximize the variance ratio between classes:
```
W = argmax(W^T * C1 * W) / (W^T * C2 * W)
```
Where C1, C2 are class covariance matrices.

### Filter Bank CSP (FBCSP)
1. Apply band-pass filters to multiple frequency bands
2. Compute CSP for each band independently  
3. Extract log-variance features from each band
4. Concatenate features for classification

### Default Frequency Bands
- **8-12 Hz**: Classical mu rhythm
- **12-16 Hz**: Low beta rhythm
- **16-20 Hz**: Mid beta rhythm  
- **20-26 Hz**: High beta rhythm
- **26-30 Hz**: Very high beta rhythm

## 📈 **Performance Evaluation**

The training pipeline provides comprehensive evaluation:

- **Stratified Cross-Validation**: Balanced folds preserve class ratios
- **Per-Class Accuracy**: Individual LEFT/RIGHT performance
- **Confusion Matrix**: Visual representation of classification errors
- **Confidence Analysis**: Prediction certainty statistics

### Interpreting Results

| Accuracy | Interpretation |
|----------|----------------|
| > 80% | Excellent - ready for real-time use |
| 70-80% | Good - suitable for most applications |
| 60-70% | Fair - may need more data or tuning |
| < 60% | Poor - check data quality and parameters |

## 🔬 **Real-time Usage**

### Model Testing
```bash
# Test trained model on new data
python runtime_preview.py --model models/mi_lda_alice.joblib --csv test_data.csv

# Real-time simulation
python runtime_preview.py --model models/mi_svm_bob.joblib --fake --realtime --delay 2.0
```

### Integration Example
```python
from runtime_preview import MotorImageryPredictor

# Load trained model
predictor = MotorImageryPredictor('models/mi_lda_alice.joblib')

# Predict on single epoch [channels x samples]
prediction = predictor.predict_epoch(epoch_data)
print(f"Predicted: {prediction['predicted_label']}")
print(f"Confidence: {prediction['confidence']:.3f}")
print(f"Probabilities: L={prediction['prob_left']:.3f}, R={prediction['prob_right']:.3f}")
```

## 📋 **Output Files**

Training produces three files in the `models/` directory:

1. **`mi_lda_user.joblib`**: Trained sklearn pipeline
2. **`mi_meta_user.json`**: Training metadata and parameters
3. **`mi_confusion_user.png`**: Confusion matrix visualization

### Metadata Contents
```json
{
  "config": {
    "fs": 250,
    "classifier": "lda",
    "csp_components": 6,
    "fbcsp_bands": [[8,12], [12,16], [16,20], [20,26], [26,30]]
  },
  "results": {
    "mean_accuracy": 0.847,
    "cv_scores": [0.83, 0.85, 0.82, 0.89, 0.84]
  },
  "motor_channels_used": ["C3", "Cz", "C4"],
  "training_date": "2024-01-15T14:30:00"
}
```

## 🛠️ **Troubleshooting**

### Common Issues

**Low Accuracy (< 60%)**
- Check electrode placement and impedances
- Verify motor imagery task instructions
- Increase training data amount
- Try different frequency bands or CSP components

**High Artifact Rejection**
- Check electrode connections
- Reduce PTP threshold: `--ptp-threshold 300`
- Relax z-score threshold: `--z-threshold 6`

**Memory Issues**
- Reduce epoch length or resample: `--resample 125`
- Use fewer CSP components: `--csp-components 4`
- Process data in smaller batches

**Channel Not Found Errors**
- Check channel names match your data
- Use `--motor-channels` to specify available channels
- Verify CSV column names or XDF channel labels

### Debug Mode
```bash
# Enable detailed logging
python train.py --csv data.csv --user test --debug
```

## 🧪 **Advanced Usage**

### Custom Preprocessing
```python
from mi_training.preprocessing import apply_filters
from mi_training.epoching import build_epochs

# Apply custom preprocessing
filtered_data = apply_filters(
    data=raw_data,
    fs=250,
    notch=50,  # European power line
    bp_low=7,   # Wider band
    bp_high=35
)
```

### Custom Feature Extraction
```python
from mi_training.features import FilterBankCSP

# Create custom FBCSP
fbcsp = FilterBankCSP(
    bands=[(6, 10), (10, 14), (14, 18), (18, 22), (22, 30)],
    n_components=8,
    fs=250
)
```

## 📚 **References**

1. Blankertz, B., et al. (2008). The BCI competition 2003: progress and perspectives in detection and discrimination of EEG single trials. IEEE Transactions on Biomedical Engineering, 51(6), 1044-1051.

2. Ang, K. K., et al. (2008). Filter bank common spatial pattern (FBCSP) in brain-computer interface. IEEE International Joint Conference on Neural Networks.

3. Ramoser, H., Muller-Gerking, J., & Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. IEEE Transactions on Rehabilitation Engineering, 8(4), 441-446.

## 🤝 **Contributing**

This is a comprehensive implementation designed for production use. For modifications:

1. Test changes with synthetic data first
2. Validate with real EEG data
3. Ensure backward compatibility
4. Update documentation

## 📄 **License**

This code is provided for educational and research purposes. Please cite appropriately if used in publications.

---

**Happy BCI Development! 🧠⚡**
