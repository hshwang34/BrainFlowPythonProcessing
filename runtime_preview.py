#!/usr/bin/env python3
"""
Motor Imagery Model Runtime Preview

This script demonstrates how to use a trained motor imagery model for
real-time classification. It loads a trained model and applies it to
test data, showing the prediction probabilities and confidence measures.

Usage Examples:
    # Test model on synthetic data
    python runtime_preview.py --model models/mi_lda_demo.joblib --fake

    # Test model on CSV file
    python runtime_preview.py --model models/mi_lda_alice.joblib --csv test_data.csv

    # Test with real-time simulation
    python runtime_preview.py --model models/mi_svm_bob.joblib --fake --realtime

Author: AI Assistant (Claude)
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Any, Optional

import numpy as np
import joblib

# Import our modules
from mi_training.config import Config, setup_logging
from mi_training.data_io import load_csv, resolve_motor_channels
from mi_training.preprocessing import apply_filters
from mi_training.epoching import build_epochs, baseline_correct
from mi_training.fake_data import synthesize_mi_trials
from mi_training.utils import setup_logging, class_distribution


class MotorImageryPredictor:
    """
    Real-time motor imagery classifier
    
    This class wraps a trained sklearn pipeline and provides methods
    for real-time prediction on new EEG data. It handles the preprocessing
    and epoching required to match the training pipeline.
    """
    
    def __init__(self, model_path: str, metadata_path: Optional[str] = None):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained sklearn pipeline (.joblib file)
            metadata_path: Path to training metadata (.json file)
        """
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.class_names = ['LEFT', 'RIGHT']
        
        # Load model and metadata
        self._load_model()
        self._load_metadata()
        
    def _load_model(self):
        """Load the trained sklearn pipeline"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            logging.info(f"Model loaded successfully: {self.model_path}")
            
            # Verify model has required methods
            if not hasattr(self.model, 'predict_proba'):
                raise ValueError("Model does not support probability prediction")
                
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
    
    def _load_metadata(self):
        """Load training metadata if available"""
        if self.metadata_path and os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logging.info(f"Metadata loaded: {self.metadata_path}")
            except Exception as e:
                logging.warning(f"Failed to load metadata: {e}")
        else:
            # Try to find metadata file based on model path
            base_name = os.path.splitext(self.model_path)[0]
            possible_meta_paths = [
                base_name + "_meta.json",
                os.path.join(os.path.dirname(self.model_path), "mi_meta.json")
            ]
            
            for meta_path in possible_meta_paths:
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            self.metadata = json.load(f)
                        self.metadata_path = meta_path
                        logging.info(f"Found and loaded metadata: {meta_path}")
                        break
                    except Exception:
                        continue
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            'model_path': self.model_path,
            'model_type': str(type(self.model)),
            'has_metadata': self.metadata is not None
        }
        
        if self.metadata:
            config = self.metadata.get('config', {})
            info.update({
                'training_accuracy': self.metadata.get('results', {}).get('mean_accuracy'),
                'classifier': config.get('classifier'),
                'csp_components': config.get('csp_components'),
                'frequency_bands': config.get('fbcsp_bands'),
                'motor_channels': self.metadata.get('motor_channels_used'),
                'training_date': self.metadata.get('training_date')
            })
        
        return info
    
    def predict_epoch(self, epoch: np.ndarray) -> Dict[str, float]:
        """
        Predict motor imagery class for a single epoch
        
        Args:
            epoch: Single epoch data [n_channels x n_samples]
            
        Returns:
            Dictionary with prediction results
        """
        # Reshape for sklearn (expects 3D input for CSP)
        epoch_3d = epoch.reshape(1, epoch.shape[0], epoch.shape[1])
        
        # Get predictions
        probabilities = self.model.predict_proba(epoch_3d)[0]
        predicted_class = self.model.predict(epoch_3d)[0]
        
        # Calculate confidence metrics
        max_prob = np.max(probabilities)
        confidence = max_prob  # Simple confidence measure
        certainty = max_prob - np.min(probabilities)  # Separation between classes
        
        return {
            'predicted_class': int(predicted_class),
            'predicted_label': self.class_names[predicted_class],
            'prob_left': float(probabilities[0]),
            'prob_right': float(probabilities[1]),
            'confidence': float(confidence),
            'certainty': float(certainty),
            'is_confident': confidence > 0.7  # Threshold for "confident" prediction
        }
    
    def predict_epochs(self, epochs: np.ndarray) -> list:
        """
        Predict motor imagery classes for multiple epochs
        
        Args:
            epochs: Epoch data [n_epochs x n_channels x n_samples]
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for i in range(epochs.shape[0]):
            epoch = epochs[i, :, :]
            pred = self.predict_epoch(epoch)
            pred['epoch_index'] = i
            predictions.append(pred)
        
        return predictions


def load_test_data(data_source: str, **kwargs) -> tuple:
    """Load test data from various sources"""
    if data_source == 'fake':
        logging.info("Generating synthetic test data")
        return synthesize_mi_trials(
            n_trials_per_class=kwargs.get('n_trials', 20),
            fs=kwargs.get('fs', 250),
            trial_dur=4.0,
            seed=kwargs.get('seed', 123)  # Different seed from training
        )
    
    elif data_source == 'csv':
        csv_path = kwargs.get('csv_path')
        if not csv_path:
            raise ValueError("CSV path required for CSV data source")
        
        from mi_training.data_io import load_csv
        return load_csv(
            path=csv_path,
            fs=kwargs.get('fs', 250),
            marker_col=kwargs.get('marker_col', 'marker')
        )
    
    else:
        raise ValueError(f"Unknown data source: {data_source}")


def preprocess_test_data(data: np.ndarray, markers: list, fs: int, 
                        config: Optional[Dict] = None) -> tuple:
    """Preprocess test data using same pipeline as training"""
    
    # Use training config if available, otherwise use defaults
    if config:
        notch_hz = config.get('notch_hz', 60)
        bp_low = config.get('bp_low', 8.0)
        bp_high = config.get('bp_high', 30.0)
        resample = config.get('resample')
    else:
        notch_hz = 60
        bp_low = 8.0
        bp_high = 30.0
        resample = None
    
    # Apply same preprocessing as training
    filtered_data = apply_filters(
        data=data,
        fs=fs,
        notch=notch_hz,
        bp_low=bp_low,
        bp_high=bp_high,
        resample=resample
    )
    
    # Update fs if resampling was applied
    if resample and resample != fs:
        fs = resample
    
    # Build epochs
    epochs, labels, time_vector = build_epochs(
        data=filtered_data,
        markers=markers,
        fs=fs,
        tmin=-0.2,
        tmax=4.0
    )
    
    # Apply baseline correction
    epochs = baseline_correct(
        epochs=epochs,
        fs=fs,
        tmin=-0.2,
        tmax=4.0,
        baseline=(-0.2, 0.0)
    )
    
    return epochs, labels, time_vector, fs


def evaluate_predictions(predictions: list, true_labels: np.ndarray) -> Dict[str, float]:
    """Evaluate prediction accuracy and confidence statistics"""
    
    predicted_labels = [pred['predicted_class'] for pred in predictions]
    probabilities = [(pred['prob_left'], pred['prob_right']) for pred in predictions]
    confidences = [pred['confidence'] for pred in predictions]
    
    # Calculate accuracy
    accuracy = np.mean(np.array(predicted_labels) == true_labels)
    
    # Calculate per-class accuracy
    left_mask = true_labels == 0
    right_mask = true_labels == 1
    
    left_accuracy = np.mean(np.array(predicted_labels)[left_mask] == true_labels[left_mask]) if np.any(left_mask) else 0
    right_accuracy = np.mean(np.array(predicted_labels)[right_mask] == true_labels[right_mask]) if np.any(right_mask) else 0
    
    # Confidence statistics
    mean_confidence = np.mean(confidences)
    confident_predictions = np.mean([pred['is_confident'] for pred in predictions])
    
    # Accuracy of confident predictions
    confident_mask = [pred['is_confident'] for pred in predictions]
    if np.any(confident_mask):
        confident_accuracy = np.mean(np.array(predicted_labels)[confident_mask] == true_labels[confident_mask])
    else:
        confident_accuracy = 0
    
    return {
        'overall_accuracy': float(accuracy),
        'left_accuracy': float(left_accuracy),
        'right_accuracy': float(right_accuracy),
        'mean_confidence': float(mean_confidence),
        'confident_predictions_ratio': float(confident_predictions),
        'confident_accuracy': float(confident_accuracy),
        'n_predictions': len(predictions)
    }


def print_prediction_summary(predictions: list, evaluation: Dict[str, float]):
    """Print a summary of prediction results"""
    
    print("\n" + "="*60)
    print("MOTOR IMAGERY PREDICTION SUMMARY")
    print("="*60)
    
    # Overall statistics
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Total predictions: {evaluation['n_predictions']}")
    print(f"   Overall accuracy: {evaluation['overall_accuracy']:.3f}")
    print(f"   LEFT accuracy: {evaluation['left_accuracy']:.3f}")
    print(f"   RIGHT accuracy: {evaluation['right_accuracy']:.3f}")
    
    print(f"\n🎯 CONFIDENCE ANALYSIS:")
    print(f"   Mean confidence: {evaluation['mean_confidence']:.3f}")
    print(f"   Confident predictions: {evaluation['confident_predictions_ratio']:.1%}")
    print(f"   Accuracy when confident: {evaluation['confident_accuracy']:.3f}")
    
    # Show some example predictions
    print(f"\n🔍 EXAMPLE PREDICTIONS:")
    print("   Idx | True | Pred | Prob L/R    | Conf | Certain")
    print("   " + "-"*48)
    
    for i, pred in enumerate(predictions[:10]):  # Show first 10
        true_label = "LEFT" if i % 2 == 0 else "RIGHT"  # Simplified for display
        pred_label = pred['predicted_label']
        prob_left = pred['prob_left']
        prob_right = pred['prob_right']
        confidence = pred['confidence']
        is_confident = "✓" if pred['is_confident'] else " "
        
        print(f"   {i:3d} | {true_label:4s} | {pred_label:4s} | {prob_left:.2f}/{prob_right:.2f} | {confidence:.2f} | {is_confident}")
    
    if len(predictions) > 10:
        print(f"   ... and {len(predictions) - 10} more predictions")
    
    print("\n" + "="*60)


def realtime_simulation(predictor: MotorImageryPredictor, test_epochs: np.ndarray, 
                       test_labels: np.ndarray, delay: float = 1.0):
    """Simulate real-time prediction with delays"""
    
    print("\n" + "="*60)
    print("REAL-TIME SIMULATION")
    print("="*60)
    print("Simulating real-time motor imagery classification...")
    print("Press Ctrl+C to stop\n")
    
    try:
        for i in range(len(test_epochs)):
            epoch = test_epochs[i]
            true_label = 'LEFT' if test_labels[i] == 0 else 'RIGHT'
            
            # Simulate processing time
            start_time = time.time()
            prediction = predictor.predict_epoch(epoch)
            processing_time = time.time() - start_time
            
            # Display prediction
            pred_label = prediction['predicted_label']
            prob_left = prediction['prob_left']
            prob_right = prediction['prob_right']
            confidence = prediction['confidence']
            
            # Color coding for terminal (if supported)
            correct = pred_label == true_label
            status = "✓" if correct else "✗"
            
            print(f"Trial {i+1:2d}: True={true_label:5s} | Pred={pred_label:5s} {status} | "
                  f"P(L/R)={prob_left:.2f}/{prob_right:.2f} | Conf={confidence:.2f} | "
                  f"Time={processing_time*1000:.1f}ms")
            
            # Wait before next prediction
            time.sleep(max(0, delay - processing_time))
            
    except KeyboardInterrupt:
        print("\nReal-time simulation stopped by user")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Motor Imagery Model Runtime Preview",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test model on synthetic data
  python runtime_preview.py --model models/mi_lda_demo.joblib --fake
  
  # Test model on CSV file
  python runtime_preview.py --model models/mi_lda_alice.joblib --csv test_data.csv
  
  # Real-time simulation
  python runtime_preview.py --model models/mi_svm_bob.joblib --fake --realtime --delay 2.0
        """
    )
    
    # Required parameters
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.joblib file)')
    
    # Data source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--fake', action='store_true',
                             help='Use synthetic test data')
    source_group.add_argument('--csv', type=str,
                             help='Path to CSV file with test data')
    
    # Optional parameters
    parser.add_argument('--metadata', type=str,
                       help='Path to training metadata (.json file)')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='Number of test trials per class for synthetic data (default: 20)')
    parser.add_argument('--marker-col', type=str, default='marker',
                       help='Marker column name for CSV data (default: marker)')
    
    # Real-time simulation
    parser.add_argument('--realtime', action='store_true',
                       help='Enable real-time simulation mode')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between predictions in seconds (default: 1.0)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed for synthetic data (default: 123)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser


def main():
    """Main function"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug)
    
    logging.info("Starting Motor Imagery model runtime preview")
    
    try:
        # Load predictor
        logging.info(f"Loading model: {args.model}")
        predictor = MotorImageryPredictor(
            model_path=args.model,
            metadata_path=args.metadata
        )
        
        # Print model information
        model_info = predictor.get_model_info()
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        print(f"Model path: {model_info['model_path']}")
        print(f"Model type: {model_info['model_type']}")
        
        if model_info['has_metadata']:
            print(f"Classifier: {model_info.get('classifier', 'Unknown')}")
            print(f"CSP components: {model_info.get('csp_components', 'Unknown')}")
            print(f"Training accuracy: {model_info.get('training_accuracy', 'Unknown'):.3f}")
            print(f"Motor channels: {model_info.get('motor_channels', 'Unknown')}")
            if model_info.get('training_date'):
                print(f"Training date: {model_info['training_date']}")
        else:
            print("No metadata available")
        
        # Load test data
        logging.info("Loading test data")
        if args.fake:
            data, markers, fs, ch_names = load_test_data(
                'fake',
                n_trials=args.n_trials,
                fs=250,
                seed=args.seed
            )
        elif args.csv:
            data, markers, fs, ch_names = load_test_data(
                'csv',
                csv_path=args.csv,
                fs=250,
                marker_col=args.marker_col
            )
        
        print(f"\nTest data loaded: {len(markers)} trials, {len(ch_names)} channels")
        
        # Preprocess test data
        config = predictor.metadata.get('config') if predictor.metadata else None
        epochs, labels, time_vector, fs = preprocess_test_data(data, markers, fs, config)
        
        # Select motor channels (same as training)
        if predictor.metadata and 'motor_channels_used' in predictor.metadata:
            motor_ch_names = predictor.metadata['motor_channels_used']
            motor_indices = []
            for ch_name in motor_ch_names:
                if ch_name in ch_names:
                    motor_indices.append(ch_names.index(ch_name))
            
            if motor_indices:
                motor_epochs = epochs[:, motor_indices, :]
                logging.info(f"Using {len(motor_indices)} motor channels: {motor_ch_names}")
            else:
                logging.warning("Motor channels from training not found, using all channels")
                motor_epochs = epochs
        else:
            # Fallback: try to find standard motor channels
            motor_indices, motor_ch_names = resolve_motor_channels(ch_names)
            motor_epochs = epochs[:, motor_indices, :]
            logging.info(f"Using default motor channels: {motor_ch_names}")
        
        print(f"Preprocessed data: {motor_epochs.shape[0]} epochs, {motor_epochs.shape[1]} channels")
        
        # Make predictions
        logging.info("Making predictions")
        predictions = predictor.predict_epochs(motor_epochs)
        
        # Evaluate predictions
        evaluation = evaluate_predictions(predictions, labels)
        
        # Print results
        print_prediction_summary(predictions, evaluation)
        
        # Real-time simulation if requested
        if args.realtime:
            realtime_simulation(predictor, motor_epochs, labels, args.delay)
        
        logging.info("Runtime preview completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logging.info("Preview interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Preview failed: {e}")
        if args.debug:
            raise
        return 1


if __name__ == '__main__':
    sys.exit(main())
