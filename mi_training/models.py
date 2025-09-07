"""
Machine learning models for Motor Imagery classification

This module builds and trains classification pipelines combining CSP/FBCSP
feature extraction with LDA or SVM classifiers. It also handles cross-validation
and model evaluation.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
import joblib

from .features import make_csp_transformer, make_fbcsp_transformer


def build_pipeline_csp_lda(
    n_components: int = 6,
    random_state: int = 42
) -> Pipeline:
    """
    Build CSP + LDA classification pipeline
    
    This is the classic motor imagery BCI pipeline:
    1. CSP: Extract spatial features that maximize class discrimination
    2. StandardScaler: Normalize features (important for LDA)
    3. LDA: Linear classifier that assumes Gaussian class distributions
    
    Why this combination?
    - CSP provides optimal spatial features for motor imagery
    - LDA works well with CSP features (typically Gaussian distributed)
    - Fast training and prediction
    - Interpretable decision boundaries
    - Proven performance in BCI competitions
    
    Args:
        n_components: Number of CSP components to extract
        random_state: Random seed for reproducibility
        
    Returns:
        Configured sklearn Pipeline
    """
    logging.info(f"Building CSP+LDA pipeline with {n_components} CSP components")
    
    pipeline = Pipeline([
        ('csp', make_csp_transformer(n_components=n_components, random_state=random_state)),
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ])
    
    return pipeline


def build_pipeline_csp_svm(
    n_components: int = 6,
    C: float = 1.0,
    random_state: int = 42
) -> Pipeline:
    """
    Build CSP + SVM classification pipeline
    
    Alternative to LDA using Support Vector Machine:
    1. CSP: Extract spatial features
    2. StandardScaler: Normalize features (critical for SVM)
    3. SVM: Non-linear classifier with RBF kernel
    
    Why SVM?
    - Can capture non-linear decision boundaries
    - Robust to outliers
    - Good generalization with proper regularization
    - Probability estimates available for confidence measures
    
    Args:
        n_components: Number of CSP components to extract
        C: SVM regularization parameter (higher = less regularization)
        random_state: Random seed for reproducibility
        
    Returns:
        Configured sklearn Pipeline
    """
    logging.info(f"Building CSP+SVM pipeline with {n_components} CSP components, C={C}")
    
    pipeline = Pipeline([
        ('csp', make_csp_transformer(n_components=n_components, random_state=random_state)),
        ('scaler', StandardScaler()),
        ('svm', SVC(C=C, kernel='rbf', probability=True, random_state=random_state))
    ])
    
    return pipeline


def build_pipeline_fbcsp_lda(
    bands: List[Tuple[float, float]],
    n_components: int = 6,
    fs: int = 250,
    random_state: int = 42
) -> Pipeline:
    """
    Build Filter Bank CSP + LDA classification pipeline
    
    Advanced pipeline using multiple frequency bands:
    1. FBCSP: Extract spatial features from multiple frequency bands
    2. StandardScaler: Normalize the concatenated features
    3. LDA: Linear classifier
    
    Why FBCSP?
    - Captures frequency-specific spatial patterns
    - More robust to individual differences in peak frequencies
    - Often achieves higher accuracy than single-band CSP
    - Standard method in BCI competitions
    
    Args:
        bands: List of (low, high) frequency bands in Hz
        n_components: Number of CSP components per frequency band
        fs: Sampling frequency in Hz
        random_state: Random seed for reproducibility
        
    Returns:
        Configured sklearn Pipeline
    """
    logging.info(f"Building FBCSP+LDA pipeline with {len(bands)} bands, {n_components} components per band")
    
    pipeline = Pipeline([
        ('fbcsp', make_fbcsp_transformer(bands=bands, n_components=n_components, fs=fs, random_state=random_state)),
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ])
    
    return pipeline


def build_pipeline_fbcsp_svm(
    bands: List[Tuple[float, float]],
    n_components: int = 6,
    fs: int = 250,
    C: float = 1.0,
    random_state: int = 42
) -> Pipeline:
    """
    Build Filter Bank CSP + SVM classification pipeline
    
    Advanced pipeline combining FBCSP with SVM classifier.
    
    Args:
        bands: List of (low, high) frequency bands in Hz
        n_components: Number of CSP components per frequency band
        fs: Sampling frequency in Hz
        C: SVM regularization parameter
        random_state: Random seed for reproducibility
        
    Returns:
        Configured sklearn Pipeline
    """
    logging.info(f"Building FBCSP+SVM pipeline with {len(bands)} bands, {n_components} components per band, C={C}")
    
    pipeline = Pipeline([
        ('fbcsp', make_fbcsp_transformer(bands=bands, n_components=n_components, fs=fs, random_state=random_state)),
        ('scaler', StandardScaler()),
        ('svm', SVC(C=C, kernel='rbf', probability=True, random_state=random_state))
    ])
    
    return pipeline


def cross_validate(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    seed: int = 42
) -> Tuple[List[float], np.ndarray, np.ndarray]:
    """
    Perform stratified cross-validation on the pipeline
    
    Cross-validation provides an unbiased estimate of model performance
    by training and testing on different data splits. Stratified CV
    ensures balanced class representation in each fold.
    
    Why cross-validation?
    - Prevents overfitting assessment
    - Uses all data for both training and testing
    - Provides confidence intervals on performance
    - Standard practice in machine learning
    
    Args:
        pipeline: Sklearn pipeline to evaluate
        X: Feature data [n_epochs x n_features] or raw epochs for CSP pipelines
        y: Labels [n_epochs]
        cv: Number of cross-validation folds
        seed: Random seed for fold generation
        
    Returns:
        Tuple of (fold_scores, y_true_all, y_pred_all) for detailed analysis
    """
    logging.info(f"Starting {cv}-fold stratified cross-validation")
    
    # Create stratified folds to ensure balanced classes in each fold
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    
    # Perform cross-validation scoring
    cv_scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring='accuracy')
    
    # Get predictions for confusion matrix
    y_pred = cross_val_predict(pipeline, X, y, cv=cv_splitter)
    
    # Log results
    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)
    
    logging.info(f"Cross-validation complete:")
    logging.info(f"  Mean accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    logging.info(f"  Fold scores: {[f'{score:.3f}' for score in cv_scores]}")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y, y_pred, target_names=['LEFT', 'RIGHT']))
    
    return cv_scores.tolist(), y, y_pred


def fit_and_save(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    model_path: str
) -> None:
    """
    Fit pipeline on all data and save to disk
    
    After cross-validation confirms good performance, we train the final
    model on all available data. This maximizes the model's learning
    from the training set.
    
    Why fit on all data?
    - Uses maximum information for final model
    - Cross-validation already provided unbiased performance estimate
    - Standard practice in machine learning deployment
    
    Args:
        pipeline: Sklearn pipeline to fit and save
        X: Training data [n_epochs x n_features] or raw epochs
        y: Training labels [n_epochs]
        model_path: Path to save the fitted model
    """
    logging.info("Fitting final model on all training data")
    
    # Fit the pipeline
    pipeline.fit(X, y)
    
    # Evaluate on training data (for reference, not performance estimate)
    train_accuracy = pipeline.score(X, y)
    logging.info(f"Training accuracy: {train_accuracy:.3f}")
    
    # Save the fitted model
    joblib.dump(pipeline, model_path)
    logging.info(f"Model saved to: {model_path}")
    
    # Verify model can be loaded
    try:
        loaded_model = joblib.load(model_path)
        test_accuracy = loaded_model.score(X, y)
        if abs(test_accuracy - train_accuracy) < 1e-6:
            logging.info("✓ Model save/load verification successful")
        else:
            logging.warning("⚠ Model save/load verification failed - accuracy mismatch")
    except Exception as e:
        logging.error(f"✗ Model save/load verification failed: {e}")


def evaluate_model_performance(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    class_names: Optional[List[str]] = None
) -> dict:
    """
    Comprehensive model performance evaluation
    
    This function provides detailed performance metrics beyond simple accuracy,
    including per-class performance and confidence measures.
    
    Args:
        pipeline: Fitted sklearn pipeline
        X: Test data
        y: True labels
        class_names: Names for classes (default: ['LEFT', 'RIGHT'])
        
    Returns:
        Dictionary with performance metrics
    """
    if class_names is None:
        class_names = ['LEFT', 'RIGHT']
    
    logging.info("Evaluating model performance")
    
    # Get predictions and probabilities
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    
    # Per-class accuracy
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        class_mask = y == i
        if np.any(class_mask):
            class_acc = accuracy_score(y[class_mask], y_pred[class_mask])
            class_accuracies[class_name] = class_acc
    
    # Confidence statistics
    max_probas = np.max(y_proba, axis=1)
    mean_confidence = np.mean(max_probas)
    
    # Prediction certainty (how often model is very confident)
    high_confidence_mask = max_probas > 0.8
    high_confidence_ratio = np.mean(high_confidence_mask)
    
    results = {
        'overall_accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'mean_confidence': mean_confidence,
        'high_confidence_ratio': high_confidence_ratio,
        'n_samples': len(y),
        'n_correct': np.sum(y == y_pred)
    }
    
    # Log results
    logging.info(f"Performance evaluation results:")
    logging.info(f"  Overall accuracy: {accuracy:.3f}")
    for class_name, class_acc in class_accuracies.items():
        logging.info(f"  {class_name} accuracy: {class_acc:.3f}")
    logging.info(f"  Mean confidence: {mean_confidence:.3f}")
    logging.info(f"  High confidence ratio: {high_confidence_ratio:.3f}")
    
    return results
