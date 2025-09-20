"""
Dataset Manager Module

Handles dataset display, selection, and management functionality
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from ...utils.data_utils import get_dataframe_from_dataset
from ..components import show_data_preview


def display_loaded_datasets():
    """Display loaded datasets and allow selection"""
    if st.session_state.datasets:
        st.subheader("Loaded Datasets")
        
        # Show summary of all datasets
        with st.expander("View all datasets", expanded=False):
            for dataset_name, dataset_info in st.session_state.datasets.items():
                try:
                    # Get DataFrame and determine source type using utility function
                    df, source_type = get_dataframe_from_dataset(dataset_info)
                    
                    # Override source type detection for legacy datasets
                    if source_type == 'file':
                        if dataset_name.startswith('snowflake_'):
                            source_type = 'snowflake'
                        elif dataset_name.endswith('.csv'):
                            source_type = 'csv'
                        elif dataset_name.endswith(('.xlsx', '.xls')):
                            source_type = 'excel'
                
                    st.write(f"**{dataset_name}** ({source_type.upper()}) - {len(df):,} rows × {len(df.columns)} columns")
                    
                except Exception as e:
                    st.write(f"**{dataset_name}** - Error reading dataset: {str(e)}")
        
        # Dataset selection
        dataset_names = list(st.session_state.datasets.keys())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_dataset = st.selectbox(
                "Select dataset for analysis", 
                dataset_names,
                index=dataset_names.index(st.session_state.current_dataset) if st.session_state.current_dataset in dataset_names else 0,
                key="dataset_selector"
            )
            
            # Update current dataset and backward compatibility variables
            if selected_dataset != st.session_state.current_dataset:
                st.session_state.current_dataset = selected_dataset
                dataset_info = st.session_state.datasets[selected_dataset]
                
                # Handle different dataset types using utility function
                try:
                    df, source_type = get_dataframe_from_dataset(dataset_info)
                    st.session_state.data = df
                    
                    # Only set connector if it exists (not needed for Snowflake)
                    if selected_dataset in st.session_state.get('connectors', {}):
                        st.session_state.connector = st.session_state.connectors[selected_dataset]
                    else:
                        st.session_state.connector = None
                except Exception as e:
                    st.error(f"Error loading dataset {selected_dataset}: {str(e)}")
                    return
                    st.session_state.connector = st.session_state.connectors.get(selected_dataset)
                st.rerun()
        
        with col2:
            if st.button("Remove Dataset", key="remove_dataset_button"):
                if len(st.session_state.datasets) > 1:
                    # Remove the selected dataset
                    del st.session_state.datasets[selected_dataset]
                    if selected_dataset in st.session_state.get('connectors', {}):
                        del st.session_state.connectors[selected_dataset]
                    
                    # Update current dataset to the first remaining one
                    remaining_datasets = list(st.session_state.datasets.keys())
                    if remaining_datasets:
                        st.session_state.current_dataset = remaining_datasets[0]
                        dataset_info = st.session_state.datasets[remaining_datasets[0]]
                        
                        # Handle different dataset types
                        if isinstance(dataset_info, dict) and 'dataframe' in dataset_info:
                            st.session_state.data = dataset_info['dataframe']
                        else:
                            st.session_state.data = dataset_info
                        
                        st.session_state.connector = st.session_state.get('connectors', {}).get(remaining_datasets[0])
                    else:
                        st.session_state.current_dataset = None
                        st.session_state.data = None
                        st.session_state.connector = None
                    
                    st.success(f"Removed dataset: {selected_dataset}")
                    st.rerun()
                else:
                    st.warning("Cannot remove the last remaining dataset")
        
        # Display current dataset info
        if st.session_state.current_dataset:
            dataset_info = st.session_state.datasets[st.session_state.current_dataset]
            
            # Get the DataFrame from either format
            if isinstance(dataset_info, dict) and 'dataframe' in dataset_info:
                current_df = dataset_info['dataframe']
                source_type = dataset_info.get('source_type', 'unknown')
                
                # Show additional info for database sources
                if source_type in ['snowflake']:
                    st.write(f"**Currently selected**: {st.session_state.current_dataset} ({source_type.title()})")
                    if 'query' in dataset_info:
                        with st.expander("View SQL Query"):
                            st.code(dataset_info['query'], language='sql')
                else:
                    st.write(f"**Currently selected**: {st.session_state.current_dataset}")
            else:
                # Legacy format
                current_df = dataset_info
                st.write(f"**Currently selected**: {st.session_state.current_dataset}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(current_df):,}")
            with col2:
                st.metric("Columns", len(current_df.columns))
            with col3:
                st.metric("Memory Usage", f"{current_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Show data preview
            show_data_preview(current_df)
    else:
        st.info("No datasets loaded yet. Please upload files or connect to a database.")


def get_dataset_info_summary(dataset_info):
    """Get a summary string for a dataset"""
    if isinstance(dataset_info, dict) and 'dataframe' in dataset_info:
        df = dataset_info['dataframe']
        source_type = dataset_info.get('source_type', 'unknown')
        return f"{len(df):,} rows × {len(df.columns)} columns ({source_type.upper()})"
    else:
        df = dataset_info
        return f"{len(df):,} rows × {len(df.columns)} columns"


def handle_dataset_selection():
    """Handle dataset selection interface"""
    if 'datasets' not in st.session_state or not st.session_state.datasets:
        st.info("No datasets available to select from")
        return
    
    dataset_names = list(st.session_state.datasets.keys())
    current_index = 0
    
    if st.session_state.get('current_dataset') in dataset_names:
        current_index = dataset_names.index(st.session_state.current_dataset)
    
    selected_dataset = st.selectbox(
        "Choose dataset:",
        dataset_names,
        index=current_index,
        key="dataset_selection"
    )
    
    if st.button("Switch to Selected Dataset", type="primary"):
        switch_to_dataset(selected_dataset)
        st.success(f"Switched to dataset: {selected_dataset}")
        st.rerun()


def handle_dataset_removal():
    """Handle dataset removal interface"""
    if 'datasets' not in st.session_state or not st.session_state.datasets:
        st.info("No datasets available to remove")
        return
    
    dataset_names = list(st.session_state.datasets.keys())
    
    dataset_to_remove = st.selectbox(
        "Choose dataset to remove:",
        dataset_names,
        key="dataset_removal"
    )
    
    if st.button("Remove Selected Dataset", type="secondary"):
        remove_dataset(dataset_to_remove)
        st.warning(f"Removed dataset: {dataset_to_remove}")
        st.rerun()


def switch_to_dataset(dataset_name):
    """Switch to a specific dataset"""
    if dataset_name not in st.session_state.datasets:
        st.error(f"Dataset '{dataset_name}' not found")
        return
    
    dataset_info = st.session_state.datasets[dataset_name]
    
    # Set current dataset
    st.session_state.current_dataset = dataset_name
    
    # Set data
    if isinstance(dataset_info, dict) and 'dataframe' in dataset_info:
        st.session_state.data = dataset_info['dataframe']
    else:
        st.session_state.data = dataset_info
    
    # Set connector if available
    st.session_state.connector = st.session_state.get('connectors', {}).get(dataset_name)


def remove_dataset(dataset_name):
    """Remove a dataset from session state"""
    if dataset_name in st.session_state.datasets:
        del st.session_state.datasets[dataset_name]
        
        if dataset_name in st.session_state.get('connectors', {}):
            del st.session_state.connectors[dataset_name]
        
        # Update current dataset if needed
        if st.session_state.current_dataset == dataset_name:
            remaining_datasets = list(st.session_state.datasets.keys())
            if remaining_datasets:
                st.session_state.current_dataset = remaining_datasets[0]
                dataset_info = st.session_state.datasets[remaining_datasets[0]]
                
                if isinstance(dataset_info, dict) and 'dataframe' in dataset_info:
                    st.session_state.data = dataset_info['dataframe']
                else:
                    st.session_state.data = dataset_info
                    
                st.session_state.connector = st.session_state.get('connectors', {}).get(remaining_datasets[0])
            else:
                st.session_state.current_dataset = None
                st.session_state.data = None
                st.session_state.connector = None
