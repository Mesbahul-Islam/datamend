"""
Utility Functions for Dataset Comparison
Contains helper functions for dataset handling and metadata extraction
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional


def get_dataset_dataframe(dataset_name: str) -> Optional[pd.DataFrame]:
    """
    Get DataFrame from dataset name, handling both formats
    
    Args:
        dataset_name: Name of the dataset in session state
        
    Returns:
        DataFrame if found, None otherwise
    """
    if dataset_name not in st.session_state.datasets:
        return None
    
    dataset_info = st.session_state.datasets[dataset_name]
    
    if isinstance(dataset_info, dict) and 'dataframe' in dataset_info:
        return dataset_info['dataframe']
    else:
        return dataset_info


def get_dataset_source_type(dataset_name: str) -> str:
    """
    Determine the source type of a dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Source type as string (SNOWFLAKE, CSV, EXCEL, etc.)
    """
    dataset_info = st.session_state.datasets[dataset_name]
    
    if isinstance(dataset_info, dict) and 'source_type' in dataset_info:
        return dataset_info['source_type'].upper()
    elif dataset_name.startswith('snowflake_'):
        return 'SNOWFLAKE'
    elif dataset_name.endswith('.csv'):
        return 'CSV'
    elif dataset_name.endswith(('.xlsx', '.xls')):
        return 'EXCEL'
    else:
        return 'UNKNOWN'


def get_source_metadata(dataset_name: str) -> Dict[str, Any]:
    """
    Get source-specific metadata from session state
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing source metadata
    """
    if dataset_name not in st.session_state.datasets:
        return {}
    
    dataset_info = st.session_state.datasets[dataset_name]
    
    if isinstance(dataset_info, dict):
        # Extract all metadata except the dataframe
        source_metadata = {k: v for k, v in dataset_info.items() if k != 'dataframe'}
        return source_metadata
    else:
        return {}


def get_severity_emoji(severity: str) -> str:
    """
    Get emoji for severity level
    
    Args:
        severity: Severity level ('high', 'medium', 'low')
        
    Returns:
        Corresponding emoji
    """
    emojis = {
        'high': '🔴',
        'medium': '🟡', 
        'low': '🟢'
    }
    return emojis.get(severity.lower(), '⚪')


def get_change_type_emoji(change_type: str) -> str:
    """
    Get emoji for different change types
    
    Args:
        change_type: Type of change detected
        
    Returns:
        Corresponding emoji
    """
    emoji_map = {
        'row_count_change': '📊',
        'columns_added': '➕',
        'columns_removed': '➖',
        'data_type_changes': '🔄',
        'schema_structure_change': '📋',
        'quality_score_change': '📈',
        'null_count_change': '🕳️',
        'duplicate_count_change': '👥',
        'column_level_changes': '📋',
        'fingerprint_changes': '🔍'
    }
    return emoji_map.get(change_type, '🔧')
