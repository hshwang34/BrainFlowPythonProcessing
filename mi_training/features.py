"""
Feature extraction for Motor Imagery classification

This module implements CSP and Filter Bank CSP (FBCSP) feature extraction.
These are the gold standard features for motor imagery BCI because they
capture the spatial patterns of brain activity that differ between classes.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from mne.decoding import CSP
from scipy import signal


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """
    Filter Bank Common Spatial Patterns (FBCSP) feature extractor
    
    FBCSP extends regular CSP by applying it to multiple frequency bands
    and concatenating the results. This captures frequency-specific spatial
    patterns that may be missed by single-band CSP.
    
    Why FBCSP?
    - Motor imagery affects multiple frequency bands (mu, beta)
    - Different subjects may have peak activity in different frequencies
    - Combining bands improves robustness and performance
    - Standard method in motor imagery BCI competitions
    
    The process:
    1. Filter data into multiple frequency bands
    2. Apply CSP to each band separately
    3. Extract log-variance features from each band
    4. Concatenate all features into final feature vector
    
    Mathematical foundation:
    CSP finds spatial filters W that maximize the ratio of variance between classes:
    W = argmax(W^T * C1 * W) / (W^T * C2 * W)
    where C1, C2 are covariance matrices for each class.
    """
    
    def __init__(
        self,
        bands: List[Tuple[float, float]],
        n_components: int = 6,
        fs: int = 250,
        random_state: Optional[int] = None
    ):
        """
        Initialize FBCSP transformer
        
        Args:
            bands: List of (low, high) frequency bands in Hz
            n_components: Number of CSP components per band
            fs: Sampling frequency in Hz
            random_state: Random seed for reproducibility
        """
        self.bands = bands
        self.n_components = n_components
        self.fs = fs
        self.random_state = random_state
        self.csp_transformers = []
        self.band_filters = []
        
    def _design_bandpass_filter(self, low: float, high: float) -> Tuple[np.ndarray, np.ndarray]:
        """Design Butterworth bandpass filter for a frequency band"""
        nyquist = self.fs / 2
        low_norm = low / nyquist
        high_norm = high / nyquist
        
        # Ensure frequencies are within valid range
        low_norm = max(low_norm, 0.01)  # Avoid DC
        high_norm = min(high_norm, 0.99)  # Avoid Nyquist
        
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        return b, a
    
    def _filter_data(self, X: np.ndarray, band_idx: int) -> np.ndarray:
        """Apply bandpass filter to data for specific frequency band"""
        b, a = self.band_filters[band_idx]
        filtered_X = np.zeros_like(X)
        
        for epoch in range(X.shape[0]):
            for ch in range(X.shape[1]):
                filtered_X[epoch, ch, :] = signal.filtfilt(b, a, X[epoch, ch, :])
        
        return filtered_X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FilterBankCSP':
        """
        Fit FBCSP transformers on training data
        
        Args:
            X: Training data [n_epochs x n_channels x n_samples]
            y: Training labels [n_epochs]
            
        Returns:
            Self for method chaining
        """
        logging.info(f"Fitting FBCSP with {len(self.bands)} frequency bands")
        logging.info(f"Bands: {self.bands}")
        logging.info(f"CSP components per band: {self.n_components}")
        
        self.csp_transformers = []
        self.band_filters = []
        
        for i, (low, high) in enumerate(self.bands):
            logging.info(f"Processing band {i+1}/{len(self.bands)}: {low}-{high} Hz")
            
            # Design bandpass filter for this band
            b, a = self._design_bandpass_filter(low, high)
            self.band_filters.append((b, a))
            
            # Filter training data to this frequency band
            X_filtered = self._filter_data(X, i)
            
            # Fit CSP transformer for this band
            csp = CSP(
                n_components=self.n_components,
                reg=None,
                log=True,  # Apply log transform to features
                norm_trace=False,
                random_state=self.random_state
            )
            
            csp.fit(X_filtered, y)
            self.csp_transformers.append(csp)
            
            logging.info(f"Band {i+1} CSP fitted successfully")
        
        logging.info("FBCSP fitting complete")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted FBCSP
        
        Args:
            X: Data to transform [n_epochs x n_channels x n_samples]
            
        Returns:
            FBCSP features [n_epochs x (n_bands * n_components)]
        """
        if not self.csp_transformers:
            raise ValueError("FBCSP must be fitted before transform")
        
        all_features = []
        
        for i, csp in enumerate(self.csp_transformers):
            # Filter data to this frequency band
            X_filtered = self._filter_data(X, i)
            
            # Apply CSP transform
            band_features = csp.transform(X_filtered)
            all_features.append(band_features)
        
        # Concatenate features from all bands
        fbcsp_features = np.concatenate(all_features, axis=1)
        
        logging.info(f"FBCSP transform: {X.shape[0]} epochs -> {fbcsp_features.shape[1]} features")
        return fbcsp_features


def make_csp_transformer(
    n_components: int = 6,
    reg: Optional[str] = None,
    random_state: int = 42
) -> CSP:
    """
    Create a standard CSP transformer
    
    Common Spatial Patterns (CSP) is the most widely used feature extraction
    method for motor imagery BCI. It finds spatial filters that maximize
    the difference in variance between two classes.
    
    Why CSP works for motor imagery:
    - Motor imagery causes event-related desynchronization (ERD) in mu/beta rhythms
    - ERD affects different brain regions for left vs right hand imagery
    - CSP finds the spatial patterns that best separate these differences
    
    Args:
        n_components: Number of CSP components to extract
        reg: Regularization method ('auto', 'shrinkage', or None)
        random_state: Random seed for reproducibility
        
    Returns:
        Configured CSP transformer
    """
    logging.info(f"Creating CSP transformer with {n_components} components")
    
    csp = CSP(
        n_components=n_components,
        reg=reg,
        log=True,  # Apply log transform to variance features
        norm_trace=False,  # Don't normalize trace (preserves absolute power)
        random_state=random_state
    )
    
    return csp


def make_fbcsp_transformer(
    bands: List[Tuple[float, float]],
    n_components: int = 6,
    fs: int = 250,
    random_state: int = 42
) -> FilterBankCSP:
    """
    Create a Filter Bank CSP transformer
    
    FBCSP is an extension of CSP that applies the algorithm to multiple
    frequency bands and combines the results. This captures frequency-specific
    spatial patterns and often improves classification performance.
    
    Default frequency bands are chosen based on motor imagery research:
    - 8-12 Hz: Classical mu rhythm
    - 12-16 Hz: Low beta
    - 16-20 Hz: Mid beta
    - 20-26 Hz: High beta
    - 26-30 Hz: Very high beta
    
    Args:
        bands: List of (low_freq, high_freq) tuples in Hz
        n_components: Number of CSP components per frequency band
        fs: Sampling frequency in Hz
        random_state: Random seed for reproducibility
        
    Returns:
        Configured FBCSP transformer
    """
    logging.info(f"Creating FBCSP transformer with {len(bands)} bands")
    logging.info(f"Frequency bands: {bands}")
    
    fbcsp = FilterBankCSP(
        bands=bands,
        n_components=n_components,
        fs=fs,
        random_state=random_state
    )
    
    return fbcsp


def extract_spectral_features(
    epochs: np.ndarray,
    fs: int,
    freq_bands: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Extract spectral power features from epochs
    
    This function computes average power in specific frequency bands
    for each channel and epoch. While not as powerful as CSP features,
    spectral features are interpretable and can complement spatial features.
    
    Args:
        epochs: Epoched data [n_epochs x n_channels x n_samples]
        fs: Sampling frequency in Hz
        freq_bands: List of (low, high) frequency bands, None for default
        
    Returns:
        Spectral features [n_epochs x (n_channels * n_bands)]
    """
    if freq_bands is None:
        freq_bands = [(8, 12), (13, 30)]  # Mu and beta bands
    
    logging.info(f"Extracting spectral features for {len(freq_bands)} frequency bands")
    
    n_epochs, n_channels, n_samples = epochs.shape
    n_bands = len(freq_bands)
    
    # Initialize feature array
    spectral_features = np.zeros((n_epochs, n_channels * n_bands))
    
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            # Compute power spectral density using Welch's method
            freqs, psd = signal.welch(
                epochs[epoch_idx, ch_idx, :],
                fs=fs,
                nperseg=min(256, n_samples),
                noverlap=None,
                window='hann'
            )
            
            # Extract power in each frequency band
            for band_idx, (low, high) in enumerate(freq_bands):
                # Find frequency indices for this band
                freq_mask = (freqs >= low) & (freqs <= high)
                
                if np.any(freq_mask):
                    # Average power in this band
                    band_power = np.mean(psd[freq_mask])
                    
                    # Store in feature vector
                    feature_idx = ch_idx * n_bands + band_idx
                    spectral_features[epoch_idx, feature_idx] = np.log(band_power + 1e-10)
    
    logging.info(f"Extracted {spectral_features.shape[1]} spectral features")
    return spectral_features


def compute_csp_patterns(csp_transformer: CSP) -> np.ndarray:
    """
    Compute CSP spatial patterns for visualization
    
    CSP filters (what the algorithm learns) are not directly interpretable
    because they are designed to maximize discrimination, not to represent
    brain activity. CSP patterns show the spatial distribution of brain
    activity that is captured by each filter.
    
    Mathematical relationship:
    patterns = (cov_matrix @ filters) / diagonal(filters^T @ cov_matrix @ filters)
    
    Args:
        csp_transformer: Fitted CSP transformer
        
    Returns:
        CSP patterns [n_channels x n_components]
    """
    if not hasattr(csp_transformer, 'filters_'):
        raise ValueError("CSP transformer must be fitted before computing patterns")
    
    # Get the spatial filters
    filters = csp_transformer.filters_
    
    # Compute average covariance matrix across both classes
    # This is an approximation - ideally we'd use the original covariance matrices
    cov_avg = np.eye(filters.shape[0])  # Identity as approximation
    
    # Compute patterns using the standard formula
    patterns = cov_avg @ filters
    
    # Normalize patterns
    for i in range(patterns.shape[1]):
        norm_factor = np.sqrt(filters[:, i].T @ cov_avg @ filters[:, i])
        patterns[:, i] /= norm_factor
    
    return patterns
