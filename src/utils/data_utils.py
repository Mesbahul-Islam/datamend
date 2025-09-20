"""
Data Utilities Module

Utility functions for handling dataset operations and data extraction
"""

import pandas as pd
from typing import Tuple, Any


def get_dataframe_from_dataset(dataset_info: Any) -> Tuple[pd.DataFrame, str]:
    """
    Extract DataFrame and source type from dataset info.
    Handles both new 'data' key and legacy 'dataframe' key formats.
    
    Args:
        dataset_info: Dataset information (can be dict or DataFrame directly)
        
    Returns:
        Tuple of (DataFrame, source_type)
    """
    if isinstance(dataset_info, dict):
        # New format with 'data' key
        if 'data' in dataset_info:
            df = dataset_info['data']
            source_type = dataset_info.get('source_type', 'unknown')
        # Legacy format with 'dataframe' key
        elif 'dataframe' in dataset_info:
            df = dataset_info['dataframe']
            source_type = dataset_info.get('source_type', 'unknown')
        else:
            raise ValueError("Invalid dataset format: missing 'data' or 'dataframe' key")
    else:
        # Very legacy format - DataFrame stored directly
        df = dataset_info
        source_type = 'file'  # Default for legacy format
    
    return df, source_type


def update_dataset_data(dataset_name: str, new_df: pd.DataFrame) -> None:
    """
    Update the DataFrame in a dataset while preserving metadata.
    
    Args:
        dataset_name: Name of the dataset to update
        new_df: New DataFrame to store
    """
    import streamlit as st
    
    if 'datasets' not in st.session_state:
        return
    
    if dataset_name not in st.session_state.datasets:
        return
    
    dataset_info = st.session_state.datasets[dataset_name]
    
    # Update the DataFrame while preserving metadata
    if isinstance(dataset_info, dict):
        if 'data' in dataset_info:
            dataset_info['data'] = new_df
        elif 'dataframe' in dataset_info:
            dataset_info['dataframe'] = new_df
        else:
            # Convert to new format
            dataset_info['data'] = new_df
    else:
        # Convert legacy format to new format
        st.session_state.datasets[dataset_name] = {
            'data': new_df,
            'source_type': 'file'
        }
    
    # Update backward compatibility fields if this is the current dataset
    if dataset_name == st.session_state.get('current_dataset'):
        st.session_state.data = new_df


def get_all_dataset_names() -> list:
    """Get list of all loaded dataset names"""
    import streamlit as st
    
    if 'datasets' not in st.session_state:
        return []
    
    return list(st.session_state.datasets.keys())


def get_dataset_summary(dataset_name: str) -> dict:
    """
    Get summary information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with summary information
    """
    import streamlit as st
    
    if 'datasets' not in st.session_state or dataset_name not in st.session_state.datasets:
        return {}
    
    dataset_info = st.session_state.datasets[dataset_name]
    df, source_type = get_dataframe_from_dataset(dataset_info)
    
    summary = {
        'name': dataset_name,
        'rows': len(df),
        'columns': len(df.columns),
        'source_type': source_type,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().sum(),
        'data_types': df.dtypes.value_counts().to_dict()
    }
    
    # Add additional metadata if available
    if isinstance(dataset_info, dict):
        summary.update({
            'upload_time': dataset_info.get('upload_time'),
            'file_size': dataset_info.get('file_size'),
            'file_name': dataset_info.get('file_name'),
            'sheet_name': dataset_info.get('sheet_name'),
            'query': dataset_info.get('query')
        })
    
    return summary
