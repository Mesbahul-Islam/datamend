"""
Statistical Anomaly Detection Module

This module provides statistical methods for detecting anomalies in data
without using machine learning approaches. It focuses on:
- Z-score based anomaly detection
- Interquartile Range (IQR) method
- Modified Z-score using median absolute deviation
- Time series anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Data class to store anomaly detection results"""
    method: str
    anomaly_indices: List[int]
    anomaly_values: List[Any]
    anomaly_scores: List[float]
    threshold: float
    total_anomalies: int
    anomaly_percentage: float


@dataclass
class StatisticalSummary:
    """Statistical summary for anomaly detection"""
    mean: float
    median: float
    std: float
    mad: float  # Median Absolute Deviation
    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection engine using various statistical methods.
    """
    
    def __init__(self, z_threshold: float = 2.0, iqr_multiplier: float = 1.5, 
                 modified_z_threshold: float = 3.5):
        """
        Initialize the anomaly detector with configurable thresholds.
        
        Args:
            z_threshold: Z-score threshold for standard deviation method
            iqr_multiplier: Multiplier for IQR method (typically 1.5 or 3.0)
            modified_z_threshold: Threshold for modified z-score method
        """
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.modified_z_threshold = modified_z_threshold
        
        logger.info(f"Anomaly detector initialized with z_threshold={z_threshold}, "
                   f"iqr_multiplier={iqr_multiplier}, modified_z_threshold={modified_z_threshold}")
    
    def detect_anomalies(self, df: pd.DataFrame, methods: List[str] = None) -> Dict[str, Dict[str, AnomalyResult]]:
        """
        Detect anomalies in a DataFrame across all numeric columns.
        
        Args:
            df: Input DataFrame
            methods: List of methods to use. Options: ['zscore', 'iqr', 'modified_zscore', 'isolation']
            
        Returns:
            Dictionary mapping column names to their anomaly detection results
        """
        if methods is None:
            methods = ['zscore', 'iqr', 'modified_zscore']
        
        results = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            try:
                series = df[column].dropna()
                if len(series) > 0:
                    column_results = self.detect_anomalies_multiple_methods(series, methods)
                    results[column] = column_results
                    logger.debug(f"Completed anomaly detection for column: {column}")
            except Exception as e:
                logger.error(f"Error detecting anomalies in column {column}: {str(e)}")
                results[column] = {}
        
        return results
    
    def detect_anomalies_multiple_methods(self, series: pd.Series, 
                                        methods: List[str] = None) -> Dict[str, AnomalyResult]:
        """
        Detect anomalies using multiple statistical methods.
        
        Args:
            series: Pandas Series with numeric data
            methods: List of methods to use. Options: ['zscore', 'iqr', 'modified_zscore', 'isolation']
            
        Returns:
            Dictionary mapping method names to their anomaly results
        """
        if methods is None:
            methods = ['zscore', 'iqr', 'modified_zscore']
        
        # Remove null values
        clean_series = series.dropna()
        if len(clean_series) == 0:
            logger.warning("Series contains no valid data for anomaly detection")
            return {}
        
        # Calculate statistical summary
        stats_summary = self._calculate_statistical_summary(clean_series)
        
        results = {}
        
        for method in methods:
            try:
                if method == 'zscore':
                    results[method] = self._detect_zscore_anomalies(clean_series, stats_summary)
                elif method == 'iqr':
                    results[method] = self._detect_iqr_anomalies(clean_series, stats_summary)
                elif method == 'modified_zscore':
                    results[method] = self._detect_modified_zscore_anomalies(clean_series, stats_summary)
                elif method == 'isolation':
                    results[method] = self._detect_isolation_anomalies(clean_series)
                else:
                    logger.warning(f"Unknown anomaly detection method: {method}")
            except Exception as e:
                logger.error(f"Error in {method} anomaly detection: {str(e)}")
        
        return results
    
    def _calculate_statistical_summary(self, series: pd.Series) -> StatisticalSummary:
        """
        Calculate comprehensive statistical summary for the series.
        
        Args:
            series: Clean numeric series (no nulls)
            
        Returns:
            StatisticalSummary object with all relevant statistics
        """
        mean_val = float(series.mean())
        median_val = float(series.median())
        std_val = float(series.std())
        
        # Median Absolute Deviation
        mad_val = float(np.median(np.abs(series - median_val)))
        
        # Quartiles and IQR
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        
        # IQR-based bounds
        lower_bound = q1 - (self.iqr_multiplier * iqr)
        upper_bound = q3 + (self.iqr_multiplier * iqr)
        
        return StatisticalSummary(
            mean=mean_val,
            median=median_val,
            std=std_val,
            mad=mad_val,
            q1=q1,
            q3=q3,
            iqr=iqr,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )
    
    def _detect_zscore_anomalies(self, series: pd.Series, 
                                stats_summary: StatisticalSummary) -> AnomalyResult:
        """
        Detect anomalies using standard Z-score method.
        
        Args:
            series: Clean numeric series
            stats_summary: Pre-calculated statistical summary
            
        Returns:
            AnomalyResult for Z-score method
        """
        if stats_summary.std == 0:
            # No variation in data, no anomalies
            return AnomalyResult(
                method='zscore',
                anomaly_indices=[],
                anomaly_values=[],
                anomaly_scores=[],
                threshold=self.z_threshold,
                total_anomalies=0,
                anomaly_percentage=0.0
            )
        
        # Calculate Z-scores
        z_scores = np.abs((series - stats_summary.mean) / stats_summary.std)
        
        # Identify anomalies
        anomaly_mask = z_scores > self.z_threshold
        anomaly_indices = series.index[anomaly_mask].tolist()
        anomaly_values = series[anomaly_mask].tolist()
        anomaly_scores = z_scores[anomaly_mask].tolist()
        
        return AnomalyResult(
            method='zscore',
            anomaly_indices=anomaly_indices,
            anomaly_values=anomaly_values,
            anomaly_scores=anomaly_scores,
            threshold=self.z_threshold,
            total_anomalies=len(anomaly_indices),
            anomaly_percentage=(len(anomaly_indices) / len(series)) * 100
        )
    
    def _detect_iqr_anomalies(self, series: pd.Series, 
                             stats_summary: StatisticalSummary) -> AnomalyResult:
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        Args:
            series: Clean numeric series
            stats_summary: Pre-calculated statistical summary
            
        Returns:
            AnomalyResult for IQR method
        """
        # Identify anomalies outside IQR bounds
        anomaly_mask = (series < stats_summary.lower_bound) | (series > stats_summary.upper_bound)
        anomaly_indices = series.index[anomaly_mask].tolist()
        anomaly_values = series[anomaly_mask].tolist()
        
        # Calculate "scores" as distance from nearest bound
        anomaly_scores = []
        for val in anomaly_values:
            if val < stats_summary.lower_bound:
                score = (stats_summary.lower_bound - val) / stats_summary.iqr
            else:
                score = (val - stats_summary.upper_bound) / stats_summary.iqr
            anomaly_scores.append(float(score))
        
        return AnomalyResult(
            method='iqr',
            anomaly_indices=anomaly_indices,
            anomaly_values=anomaly_values,
            anomaly_scores=anomaly_scores,
            threshold=self.iqr_multiplier,
            total_anomalies=len(anomaly_indices),
            anomaly_percentage=(len(anomaly_indices) / len(series)) * 100
        )
    
    def _detect_modified_zscore_anomalies(self, series: pd.Series, 
                                         stats_summary: StatisticalSummary) -> AnomalyResult:
        """
        Detect anomalies using Modified Z-score based on Median Absolute Deviation.
        This method is more robust to outliers than standard Z-score.
        
        Args:
            series: Clean numeric series
            stats_summary: Pre-calculated statistical summary
            
        Returns:
            AnomalyResult for Modified Z-score method
        """
        if stats_summary.mad == 0:
            # No variation in data, no anomalies
            return AnomalyResult(
                method='modified_zscore',
                anomaly_indices=[],
                anomaly_values=[],
                anomaly_scores=[],
                threshold=self.modified_z_threshold,
                total_anomalies=0,
                anomaly_percentage=0.0
            )
        
        # Calculate modified Z-scores
        # Modified Z-score = 0.6745 * (x - median) / MAD
        modified_z_scores = np.abs(0.6745 * (series - stats_summary.median) / stats_summary.mad)
        
        # Identify anomalies
        anomaly_mask = modified_z_scores > self.modified_z_threshold
        anomaly_indices = series.index[anomaly_mask].tolist()
        anomaly_values = series[anomaly_mask].tolist()
        anomaly_scores = modified_z_scores[anomaly_mask].tolist()
        
        return AnomalyResult(
            method='modified_zscore',
            anomaly_indices=anomaly_indices,
            anomaly_values=anomaly_values,
            anomaly_scores=anomaly_scores,
            threshold=self.modified_z_threshold,
            total_anomalies=len(anomaly_indices),
            anomaly_percentage=(len(anomaly_indices) / len(series)) * 100
        )
    
    def _detect_isolation_anomalies(self, series: pd.Series) -> AnomalyResult:
        """
        Simple statistical isolation method based on extreme percentiles.
        
        Args:
            series: Clean numeric series
            
        Returns:
            AnomalyResult for isolation method
        """
        # Define extreme percentiles (bottom 2% and top 2%)
        lower_percentile = series.quantile(0.02)
        upper_percentile = series.quantile(0.98)
        
        # Identify anomalies
        anomaly_mask = (series < lower_percentile) | (series > upper_percentile)
        anomaly_indices = series.index[anomaly_mask].tolist()
        anomaly_values = series[anomaly_mask].tolist()
        
        # Calculate scores based on percentile distance
        anomaly_scores = []
        for val in anomaly_values:
            if val < lower_percentile:
                score = (lower_percentile - val) / (series.max() - series.min())
            else:
                score = (val - upper_percentile) / (series.max() - series.min())
            anomaly_scores.append(float(score))
        
        return AnomalyResult(
            method='isolation',
            anomaly_indices=anomaly_indices,
            anomaly_values=anomaly_values,
            anomaly_scores=anomaly_scores,
            threshold=0.02,  # 2% threshold
            total_anomalies=len(anomaly_indices),
            anomaly_percentage=(len(anomaly_indices) / len(series)) * 100
        )
    
    def detect_time_series_anomalies(self, series: pd.Series, 
                                   window_size: int = 10) -> AnomalyResult:
        """
        Detect anomalies in time series data using rolling statistics.
        
        Args:
            series: Time series data
            window_size: Window size for rolling statistics
            
        Returns:
            AnomalyResult for time series method
        """
        if len(series) < window_size * 2:
            logger.warning(f"Series too short for time series anomaly detection with window_size={window_size}")
            return AnomalyResult(
                method='time_series',
                anomaly_indices=[],
                anomaly_values=[],
                anomaly_scores=[],
                threshold=self.z_threshold,
                total_anomalies=0,
                anomaly_percentage=0.0
            )
        
        # Calculate rolling mean and std
        rolling_mean = series.rolling(window=window_size, center=True).mean()
        rolling_std = series.rolling(window=window_size, center=True).std()
        
        # Calculate z-scores relative to rolling statistics
        z_scores = np.abs((series - rolling_mean) / rolling_std)
        
        # Remove NaN values from edges
        valid_mask = ~z_scores.isnull()
        z_scores_clean = z_scores[valid_mask]
        series_clean = series[valid_mask]
        
        # Identify anomalies
        anomaly_mask = z_scores_clean > self.z_threshold
        anomaly_indices = series_clean.index[anomaly_mask].tolist()
        anomaly_values = series_clean[anomaly_mask].tolist()
        anomaly_scores = z_scores_clean[anomaly_mask].tolist()
        
        return AnomalyResult(
            method='time_series',
            anomaly_indices=anomaly_indices,
            anomaly_values=anomaly_values,
            anomaly_scores=anomaly_scores,
            threshold=self.z_threshold,
            total_anomalies=len(anomaly_indices),
            anomaly_percentage=(len(anomaly_indices) / len(series_clean)) * 100
        )
    
    def get_anomaly_summary(self, results: Dict[str, AnomalyResult]) -> Dict[str, Any]:
        """
        Generate a summary of anomaly detection results across multiple methods.
        
        Args:
            results: Dictionary of anomaly results from different methods
            
        Returns:
            Summary dictionary with key metrics
        """
        if not results:
            return {}
        
        summary = {
            'total_methods': len(results),
            'method_results': {},
            'consensus_anomalies': [],
            'high_confidence_anomalies': []
        }
        
        # Summarize each method
        for method, result in results.items():
            summary['method_results'][method] = {
                'total_anomalies': result.total_anomalies,
                'anomaly_percentage': result.anomaly_percentage,
                'threshold': result.threshold
            }
        
        # Find consensus anomalies (detected by multiple methods)
        if len(results) > 1:
            all_indices = [set(result.anomaly_indices) for result in results.values()]
            
            # Anomalies detected by at least 2 methods
            consensus = set.intersection(*all_indices) if len(all_indices) > 1 else set()
            summary['consensus_anomalies'] = list(consensus)
            
            # High confidence: detected by majority of methods
            majority_threshold = len(results) // 2 + 1
            index_counts = {}
            for indices in all_indices:
                for idx in indices:
                    index_counts[idx] = index_counts.get(idx, 0) + 1
            
            high_confidence = [idx for idx, count in index_counts.items() 
                             if count >= majority_threshold]
            summary['high_confidence_anomalies'] = high_confidence
        
        return summary
