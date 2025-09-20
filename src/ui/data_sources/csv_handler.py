"""
CSV Handler Module

Handles CSV file upload, processing, and data loading
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from ...connectors.data_connectors import DataConnectorFactory
from ..components import show_data_preview
from ...utils.data_utils import get_dataframe_from_dataset


def handle_csv_upload():
    """Simplified CSV file upload interface"""
    
    # File upload - supports both single and multiple files
    uploaded_files = st.file_uploader(
        "Select CSV files to upload", 
        type=['csv'], 
        accept_multiple_files=True,
        help="Choose one or more CSV files"
    )
    
    if uploaded_files:
        # Show selected files
        if len(uploaded_files) == 1:
            st.info(f"Selected: {uploaded_files[0].name}")
        else:
            st.info(f"Selected {len(uploaded_files)} files")
            with st.expander("View selected files"):
                for file in uploaded_files:
                    st.write(f"• {file.name}")
        
        # CSV configuration
        st.write("Configuration:")
        col1, col2 = st.columns(2)
        
        with col1:
            encoding = st.selectbox("File encoding", ["utf-8", "latin-1", "cp1252"], index=0)
            delimiter = st.selectbox("Delimiter", [",", ";", "\t", "|"], index=0)
        
        with col2:
            load_full = st.checkbox("Load complete dataset", value=True)
            if not load_full:
                sample_rows = st.number_input("Number of rows to load", min_value=100, max_value=1000000, value=10000)
            else:
                sample_rows = None
        
        # Load button
        if st.button("Load Files", type="primary"):
            load_csv_files(uploaded_files, encoding, delimiter, sample_rows)


def load_csv_files(uploaded_files, encoding, delimiter, sample_rows):
    """Load CSV files and store in session state"""
    successful_loads = 0
    failed_loads = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Loading {uploaded_file.name}...")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Create connector and load data
                connector = DataConnectorFactory.create_connector(
                    'csv',
                    file_path=temp_path,
                    encoding=encoding,
                    delimiter=delimiter
                )
                
                if connector.connect():
                    df = connector.get_data(limit=sample_rows)
                    
                    # Store in datasets with source type information
                    dataset_name = uploaded_file.name
                    if 'datasets' not in st.session_state:
                        st.session_state.datasets = {}
                    if 'connectors' not in st.session_state:
                        st.session_state.connectors = {}
                        
                    st.session_state.datasets[dataset_name] = {
                        'data': df,  # Use 'data' key for consistency
                        'source_type': 'csv',
                        'upload_time': datetime.now(),
                        'file_size': uploaded_file.size,
                        'file_name': uploaded_file.name
                    }
                    st.session_state.connectors[dataset_name] = connector
                    
                    # Set first file as current dataset
                    if i == 0:
                        st.session_state.data = df
                        st.session_state.connector = connector
                        st.session_state.current_dataset = dataset_name
                    
                    successful_loads += 1
                    
                    # Integrate with quality checking
                    source_info = {
                        'file_name': uploaded_file.name,
                        'upload_time': str(datetime.now()),
                        'file_size': uploaded_file.size,
                        'encoding': encoding,
                        'delimiter': delimiter
                    }
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                else:
                    failed_loads.append(f"{uploaded_file.name}: Failed to connect")
                    
        except Exception as e:
            failed_loads.append(f"{uploaded_file.name}: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show results
    if successful_loads > 0:
        st.success(f"Successfully loaded {successful_loads} out of {len(uploaded_files)} files")
        
        # Show summary of loaded datasets
        st.write("Loaded datasets:")
        for dataset_name, dataset_info in st.session_state.datasets.items():
            try:
                df, source_type = get_dataframe_from_dataset(dataset_info)
                st.write(f"• {dataset_name}: {len(df):,} rows, {len(df.columns)} columns")
            except Exception as e:
                st.write(f"• {dataset_name}: Error reading dataset - {str(e)}")
        
        # Note: Quality analysis is now available in the Data Profiling tab
    
    if failed_loads:
        st.error("Failed to load some files:")
        for error in failed_loads:
            st.write(f"• {error}")


def load_single_csv_file(uploaded_file, encoding, delimiter, sample_rows):
    """Load a single CSV file and store in session state"""
    try:
        with st.spinner("Loading CSV data..."):
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Create connector and load data
            connector = DataConnectorFactory.create_connector(
                'csv',
                file_path=temp_path,
                encoding=encoding,
                delimiter=delimiter
            )
            
            if connector.connect():
                df = connector.get_data(limit=sample_rows)
                
                # Store in session state (backward compatibility)
                st.session_state.data = df
                st.session_state.connector = connector
                
                # Also store in datasets for multi-file support with source type information
                dataset_name = uploaded_file.name
                if 'datasets' not in st.session_state:
                    st.session_state.datasets = {}
                if 'connectors' not in st.session_state:
                    st.session_state.connectors = {}
                    
                st.session_state.datasets[dataset_name] = {
                    'data': df,  # Use 'data' key for consistency
                    'source_type': 'csv',
                    'upload_time': datetime.now(),
                    'file_size': uploaded_file.size,
                    'file_name': uploaded_file.name
                }
                st.session_state.connectors[dataset_name] = connector
                st.session_state.current_dataset = dataset_name
                
                # Clean up temp file
                os.remove(temp_path)
                
                st.success(f"🟢 Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
                
                # Show data preview
                show_data_preview(df)
            else:
                st.error("🔴 Failed to connect to CSV file")
                
    except Exception as e:
        st.error(f"🔴 Error loading CSV: {str(e)}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
