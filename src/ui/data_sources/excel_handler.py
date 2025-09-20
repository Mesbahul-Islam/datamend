"""
Excel Handler Module

Handles Excel file upload, processing, and data loading
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from ...connectors.data_connectors import DataConnectorFactory
from ..components import show_data_preview
from ...utils.data_utils import get_dataframe_from_dataset


def handle_excel_upload():
    """Handle Excel file upload"""
    
    # File upload
    uploaded_file = st.file_uploader("Select Excel file to upload", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Get sheet names
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            # Sheet selection
            col1, col2 = st.columns(2)
            
            with col1:
                selected_sheet = st.selectbox("Select Sheet", sheet_names)
            with col2:
                load_full = st.checkbox("Load Full Dataset", value=True, 
                                       help="Uncheck to load only a sample for testing", key="excel_load_full")
            
            if not load_full:
                sample_rows = st.number_input("Sample Rows", min_value=100, max_value=1000000, value=10000, key="excel_sample_rows")
            else:
                sample_rows = None
            
            if st.button("Load Excel Data", type="primary", key="load_excel_button"):
                load_excel_data(uploaded_file, selected_sheet, sample_rows)
                        
        except Exception as e:
            st.error(f"🔴 Error reading Excel file: {str(e)}")


def load_excel_data(uploaded_file, selected_sheet, sample_rows):
    """Load Excel data and store in session state"""
    try:
        with st.spinner("Loading Excel data..."):
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Create connector and load data
            connector = DataConnectorFactory.create_connector(
                'excel',
                file_path=temp_path,
                sheet_name=selected_sheet
            )
            
            if connector.connect():
                df = connector.get_data(limit=sample_rows)
                
                # Store in session state with source type information
                dataset_name = f"{uploaded_file.name}_{selected_sheet}"
                if 'datasets' not in st.session_state:
                    st.session_state.datasets = {}
                if 'connectors' not in st.session_state:
                    st.session_state.connectors = {}
                    
                st.session_state.datasets[dataset_name] = {
                    'data': df,  # Use 'data' key for consistency
                    'source_type': 'excel',
                    'upload_time': datetime.now(),
                    'file_size': uploaded_file.size,
                    'file_name': uploaded_file.name,
                    'sheet_name': selected_sheet
                }
                st.session_state.connectors[dataset_name] = connector
                st.session_state.current_dataset = dataset_name
                st.session_state.data = df
                st.session_state.connector = connector
                
                # Integrate with quality checking
                source_info = {
                    'file_name': uploaded_file.name,
                    'sheet_name': selected_sheet,
                    'upload_time': str(datetime.now()),
                    'file_size': uploaded_file.size
                }
                
                # Clean up temp file
                os.remove(temp_path)
                
                st.success(f"🟢 Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
                
                # Note: Quality analysis is now available in the Data Profiling tab
                
                # Show data preview
                show_data_preview(df)
            else:
                st.error("🔴 Failed to connect to Excel file")
                
    except Exception as e:
        st.error(f"🔴 Error loading Excel: {str(e)}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
