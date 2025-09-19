"""
Data Quality Engine Core Module

This module provides the main data quality engine that performs:
- Multithreaded data profiling with chunking
- Proactive data quality issue detection
- Statistical analysis of data columns
"""

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import warnings
from multiprocessing import cpu_count

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    """Data class to store column profiling results"""
    column_name: str
    data_type: str
    total_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    duplicate_count: int
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mode: Optional[Any] = None
    outliers_count: int = 0
    data_quality_issues: List[str] = None
    sample_values: List[Any] = None

    def __post_init__(self):
        if self.data_quality_issues is None:
            self.data_quality_issues = []
        if self.sample_values is None:
            self.sample_values = []


@dataclass
class DataQualityReport:
    """Data class to store complete data quality report"""
    dataset_name: str
    total_rows: int
    total_columns: int
    profiling_timestamp: datetime
    column_profiles: Dict[str, ColumnProfile]
    overall_quality_score: float
    critical_issues: List[str]
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class DataQualityEngine:
    """
    Main data quality engine that performs comprehensive data profiling
    and quality assessment using multithreading and chunking for scalability.
    """
    
    def __init__(self, chunk_size: int = 10000, max_workers: int = 4, anomaly_threshold: float = 2.0):
        """
        Initialize the data quality engine.
        
        Args:
            chunk_size: Size of chunks for processing large datasets
            max_workers: Maximum number of threads for parallel processing
            anomaly_threshold: Z-score threshold for anomaly detection
        """
        self.chunk_size = chunk_size
        self.max_workers = max(1, min(max_workers, cpu_count()))
        self.anomaly_threshold = anomaly_threshold
        
        logger.info(f"Data Quality Engine initialized with chunk_size={chunk_size}, "
                   f"max_workers={max_workers}, anomaly_threshold={anomaly_threshold}")
    
    def profile_dataset(self, df: pd.DataFrame, dataset_name: str = "Unknown") -> DataQualityReport:
        """
        Profile an entire dataset using multithreading and chunking.
        
        Args:
            df: Input DataFrame to profile
            dataset_name: Name of the dataset for reporting
            
        Returns:
            DataQualityReport: Comprehensive quality report
        """
        logger.info(f"Starting data profiling for dataset: {dataset_name}")
        logger.info(f"Dataset shape: {df.shape}")
        
        start_time = datetime.now()
        
        # Choose processing strategy based on dataset size
        if len(df) > 100000:
            logger.info(f"Using multithreaded processing for large dataset ({len(df)} rows)")
            column_profiles = self._profile_columns_parallel(df)
        else:
            logger.info(f"Using sequential processing for small dataset ({len(df)} rows)")
            column_profiles = self._profile_columns_sequential(df)
        
        # Calculate overall quality metrics
        overall_score = self._calculate_overall_quality_score(column_profiles)
        critical_issues = self._identify_critical_issues(column_profiles)
        
        report = DataQualityReport(
            dataset_name=dataset_name,
            total_rows=len(df),
            total_columns=len(df.columns),
            profiling_timestamp=datetime.now(),
            column_profiles=column_profiles,
            overall_quality_score=overall_score,
            critical_issues=critical_issues
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Data profiling completed in {processing_time:.2f} seconds")
        logger.info(f"Overall quality score: {overall_score:.2f}")
        
        return report
    
    def _profile_columns_parallel(self, df: pd.DataFrame) -> Dict[str, ColumnProfile]:
        """
        Profile all columns in parallel using ThreadPoolExecutor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column names to their profiles
        """
        column_profiles = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit profiling tasks for each column
            future_to_column = {
                executor.submit(self._profile_single_column, df[col], col): col
                for col in df.columns
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_column):
                column_name = future_to_column[future]
                try:
                    profile = future.result()
                    column_profiles[column_name] = profile
                    logger.debug(f"Completed profiling for column: {column_name}")
                except Exception as e:
                    logger.error(f"Error profiling column {column_name}: {str(e)}")
                    # Create a basic error profile
                    column_profiles[column_name] = ColumnProfile(
                        column_name=column_name,
                        data_type="error",
                        total_count=len(df[column_name]),
                        null_count=0,
                        null_percentage=0.0,
                        unique_count=0,
                        duplicate_count=0,
                        data_quality_issues=[f"Profiling error: {str(e)}"]
                    )
        
        return column_profiles
    
    def _profile_columns_sequential(self, df: pd.DataFrame) -> Dict[str, ColumnProfile]:
        """
        Profile all columns sequentially (for smaller datasets).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column names to their profiles
        """
        column_profiles = {}
        
        for col in df.columns:
            try:
                profile = self._profile_single_column(df[col], col)
                column_profiles[col] = profile
                logger.debug(f"Completed profiling for column: {col}")
            except Exception as e:
                logger.error(f"Error profiling column {col}: {str(e)}")
                # Create a basic error profile
                column_profiles[col] = ColumnProfile(
                    column_name=col,
                    data_type="error",
                    total_count=len(df[col]),
                    null_count=0,
                    null_percentage=0.0,
                    unique_count=0,
                    duplicate_count=0,
                    data_quality_issues=[f"Profiling error: {str(e)}"]
                )
        
        return column_profiles
    
    def _profile_single_column(self, series: pd.Series, column_name: str) -> ColumnProfile:
        """
        Profile a single column with chunking for large datasets.
        
        Args:
            series: Pandas Series to profile
            column_name: Name of the column
            
        Returns:
            ColumnProfile: Profile results for the column
        """
        logger.debug(f"Profiling column: {column_name}")
        
        # Basic statistics
        total_count = len(series)
        null_count = series.isnull().sum()
        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
        
        # Remove null values for further analysis
        non_null_series = series.dropna()
        unique_count = len(non_null_series.unique()) if len(non_null_series) > 0 else 0
        duplicate_count = total_count - unique_count
        
        # Determine data type
        data_type = self._determine_data_type(series)
        
        # Initialize profile
        profile = ColumnProfile(
            column_name=column_name,
            data_type=data_type,
            total_count=total_count,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            duplicate_count=duplicate_count,
            sample_values=non_null_series.head(5).tolist() if len(non_null_series) > 0 else []
        )
        
        # Additional statistics for numeric columns
        if data_type in ['integer', 'float'] and len(non_null_series) > 0:
            profile.mean = float(non_null_series.mean())
            profile.median = float(non_null_series.median())
            profile.std = float(non_null_series.std())
            profile.min_value = float(non_null_series.min())
            profile.max_value = float(non_null_series.max())
            
            # Detect outliers using chunking for large datasets
            profile.outliers_count = self._detect_outliers_chunked(non_null_series)
        
        # Mode for all types
        if len(non_null_series) > 0:
            mode_values = non_null_series.mode()
            profile.mode = mode_values.iloc[0] if len(mode_values) > 0 else None
        
        # Identify data quality issues
        profile.data_quality_issues = self._identify_column_issues(profile, non_null_series)
        
        return profile
    
    def _determine_data_type(self, series: pd.Series) -> str:
        """
        Determine the effective data type of a series.
        
        Args:
            series: Pandas Series to analyze
            
        Returns:
            String representation of the data type
        """
        dtype = series.dtype
        
        if pd.api.types.is_integer_dtype(dtype):
            return 'integer'
        elif pd.api.types.is_float_dtype(dtype):
            return 'float'
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return 'datetime'
        elif pd.api.types.is_bool_dtype(dtype):
            return 'boolean'
        elif pd.api.types.is_categorical_dtype(dtype):
            return 'categorical'
        else:
            return 'string'
    
    def _detect_outliers_chunked(self, series: pd.Series) -> int:
        """
        Detect outliers using z-score method with chunking for large datasets.
        
        Args:
            series: Numeric series to analyze
            
        Returns:
            Count of outliers detected
        """
        if len(series) == 0:
            return 0
        
        outliers_count = 0
        
        # Process in chunks for memory efficiency
        for i in range(0, len(series), self.chunk_size):
            chunk = series.iloc[i:i + self.chunk_size]
            
            # Calculate z-scores for the chunk
            mean_val = series.mean()  # Use overall mean
            std_val = series.std()    # Use overall std
            
            if std_val == 0:
                continue
                
            z_scores = np.abs((chunk - mean_val) / std_val)
            outliers_count += (z_scores > self.anomaly_threshold).sum()
        
        return outliers_count
    
    def _identify_column_issues(self, profile: ColumnProfile, series: pd.Series) -> List[str]:
        """
        Identify data quality issues for a column.
        
        Args:
            profile: Column profile with basic statistics
            series: Non-null series data
            
        Returns:
            List of identified issues
        """
        issues = []
        
        # High null percentage
        if profile.null_percentage > 50:
            issues.append(f"HIGH_NULL_RATE: {profile.null_percentage:.1f}% null values")
        elif profile.null_percentage > 10:
            issues.append(f"MODERATE_NULL_RATE: {profile.null_percentage:.1f}% null values")
        
        # Low uniqueness (potential data quality issue)
        if profile.total_count > 0:
            uniqueness_ratio = profile.unique_count / profile.total_count
            if uniqueness_ratio < 0.01 and profile.data_type != 'categorical':
                issues.append(f"LOW_UNIQUENESS: Only {uniqueness_ratio:.2%} unique values")
        
        # High outlier rate for numeric columns
        if profile.data_type in ['integer', 'float'] and profile.outliers_count > 0:
            outlier_rate = profile.outliers_count / len(series) if len(series) > 0 else 0
            if outlier_rate > 0.05:  # More than 5% outliers
                issues.append(f"HIGH_OUTLIER_RATE: {outlier_rate:.2%} outliers detected")
        
        # Detect potential formatting issues for string columns
        if profile.data_type == 'string' and len(series) > 0:
            # Check for inconsistent casing
            if len(series.str.lower().unique()) != len(series.unique()):
                issues.append("INCONSISTENT_CASING: Mixed case values detected")
            
            # Check for leading/trailing whitespace
            if (series.str.strip() != series).any():
                issues.append("WHITESPACE_ISSUES: Leading/trailing whitespace detected")
        
        # Check for potential ID columns with gaps
        if profile.data_type == 'integer' and profile.unique_count == profile.total_count:
            if len(series) > 1:
                expected_range = profile.max_value - profile.min_value + 1
                if expected_range != profile.unique_count:
                    issues.append("ID_GAPS: Potential gaps in ID sequence")
        
        return issues
    
    def _calculate_overall_quality_score(self, column_profiles: Dict[str, ColumnProfile]) -> float:
        """
        Calculate an overall data quality score (0-100).
        
        Args:
            column_profiles: Dictionary of column profiles
            
        Returns:
            Overall quality score
        """
        if not column_profiles:
            return 0.0
        
        total_score = 0.0
        
        for profile in column_profiles.values():
            column_score = 100.0
            
            # Penalize for null values
            column_score -= profile.null_percentage * 0.5
            
            # Penalize for data quality issues
            column_score -= len(profile.data_quality_issues) * 10
            
            # Ensure score doesn't go below 0
            column_score = max(0.0, column_score)
            total_score += column_score
        
        return total_score / len(column_profiles)
    
    def _identify_critical_issues(self, column_profiles: Dict[str, ColumnProfile]) -> List[str]:
        """
        Identify critical data quality issues across the dataset.
        
        Args:
            column_profiles: Dictionary of column profiles
            
        Returns:
            List of critical issues
        """
        critical_issues = []
        
        # Count columns with high null rates
        high_null_columns = [
            profile.column_name for profile in column_profiles.values()
            if profile.null_percentage > 50
        ]
        
        if high_null_columns:
            critical_issues.append(
                f"High null rate in {len(high_null_columns)} columns: {', '.join(high_null_columns[:3])}"
                + ("..." if len(high_null_columns) > 3 else "")
            )
        
        # Count columns with data quality issues
        problematic_columns = [
            profile.column_name for profile in column_profiles.values()
            if len(profile.data_quality_issues) >= 2
        ]
        
        if problematic_columns:
            critical_issues.append(
                f"Multiple issues in {len(problematic_columns)} columns: {', '.join(problematic_columns[:3])}"
                + ("..." if len(problematic_columns) > 3 else "")
            )
        
        return critical_issues
