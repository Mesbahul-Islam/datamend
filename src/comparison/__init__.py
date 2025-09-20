"""
Dataset Comparison Package

This package provides comprehensive dataset comparison functionality including:
- Metadata-based comparison for fast analysis
- Visualization components for comparison results
- Report generation and export capabilities
- Core comparison engine and orchestration

Modules:
- utils: Utility functions for dataset handling
- visualizations: Chart and visualization components
- reports: Report generation and export functions  
- engine: Core comparison logic and orchestration

Usage:
    from src.comparison.engine import run_metadata_comparison
    from src.comparison.utils import get_dataset_dataframe
"""

# Import key functions for easy access
from .utils import get_dataset_dataframe, get_dataset_source_type, get_source_metadata
from .engine import run_metadata_comparison

__all__ = [
    'get_dataset_dataframe',
    'get_dataset_source_type',
    'get_source_metadata',
    'run_metadata_comparison'
]

__version__ = "1.0.0"
__author__ = "DataMend Team"
