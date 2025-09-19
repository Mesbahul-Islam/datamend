"""
Data Source Tab - Handle various data source types (CSV, Excel)
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from connectors.data_connectors import DataConnectorFactory
from ui.components import show_data_preview, display_loading_info, create_file_upload_section


def data_source_tab():
    """Data source selection and connection tab"""
    st.header("📁 Data Source")
    
    # Data source selection
    source_type = st.selectbox("Select Data Source Type", 
                               ["Multiple CSV Files", "Single CSV File", "Excel File"],
                               help="Choose your data source type")
    
    if source_type == "Multiple CSV Files":
        handle_multiple_csv_upload()
    elif source_type == "Single CSV File":
        handle_single_csv_upload()
    elif source_type == "Excel File":
        handle_excel_upload()
    
    # Display loaded datasets
    display_loaded_datasets()


def handle_single_csv_upload():
    """Handle single CSV file upload"""
    st.subheader("📄 Single CSV File Upload")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="single_csv_uploader")
    
    if uploaded_file is not None:
        # CSV configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], index=0, key="single_csv_encoding")
        with col2:
            delimiter = st.selectbox("Delimiter", [",", ";", "\t", "|"], index=0, key="single_csv_delimiter")
        with col3:
            load_full = st.checkbox("Load Full Dataset", value=True, 
                                   help="Uncheck to load only a sample for testing", key="single_csv_load_full")
        
        if not load_full:
            sample_rows = st.number_input("Sample Rows", min_value=100, max_value=10000, value=1000, key="single_csv_sample_rows")
        else:
            sample_rows = None
        
        if st.button("Load CSV Data", type="primary", key="load_single_csv_button"):
            load_single_csv_file(uploaded_file, encoding, delimiter, sample_rows)


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
                
                # Also store in datasets for multi-file support
                dataset_name = uploaded_file.name
                st.session_state.datasets[dataset_name] = df
                st.session_state.connectors[dataset_name] = connector
                st.session_state.current_dataset = dataset_name
                
                # Clean up temp file
                os.remove(temp_path)
                
                st.success(f"✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
                
                # Show data preview
                show_data_preview(df)
            else:
                st.error("❌ Failed to connect to CSV file")
                
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


def handle_multiple_csv_upload():
    """Handle multiple CSV file uploads"""
    st.subheader("📄 Multiple CSV Files Upload")
    
    # File upload for multiple files
    uploaded_files = st.file_uploader(
        "Choose CSV files", 
        type=['csv'], 
        accept_multiple_files=True,
        key="multiple_csv_uploader",
        help="You can select multiple CSV files to upload and analyze separately"
    )
    
    if uploaded_files:
        st.write(f"**Selected {len(uploaded_files)} file(s):**")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size:,} bytes)")
        
        # Common CSV configuration for all files
        st.write("**Configuration (applies to all files):**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], index=0, key="multi_csv_encoding")
        with col2:
            delimiter = st.selectbox("Delimiter", [",", ";", "\t", "|"], index=0, key="multi_csv_delimiter")
        with col3:
            load_full = st.checkbox("Load Full Dataset", value=True, 
                                   help="Uncheck to load only a sample for testing", key="multi_csv_load_full")
        
        if not load_full:
            sample_rows = st.number_input("Sample Rows", min_value=100, max_value=10000, value=1000, key="multi_csv_sample_rows")
        else:
            sample_rows = None
        
        if st.button("Load All CSV Files", type="primary", key="load_multiple_csv_button"):
            load_multiple_csv_files(uploaded_files, encoding, delimiter, sample_rows)


def load_multiple_csv_files(uploaded_files, encoding, delimiter, sample_rows):
    """Load multiple CSV files and store each in session state"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    successful_loads = 0
    failed_loads = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Loading {uploaded_file.name}...")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        try:
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
                
                # Store in datasets
                dataset_name = uploaded_file.name
                st.session_state.datasets[dataset_name] = df
                st.session_state.connectors[dataset_name] = connector
                
                # Set first file as current dataset for backward compatibility
                if i == 0:
                    st.session_state.data = df
                    st.session_state.connector = connector
                    st.session_state.current_dataset = dataset_name
                
                successful_loads += 1
                
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
        st.success(f"✅ Successfully loaded {successful_loads} out of {len(uploaded_files)} files")
        
        # Show summary of loaded datasets
        st.write("**Loaded datasets:**")
        for dataset_name, df in st.session_state.datasets.items():
            st.write(f"• **{dataset_name}**: {len(df):,} rows, {len(df.columns)} columns")
    
    if failed_loads:
        st.error("❌ Failed to load some files:")
        for error in failed_loads:
            st.write(f"• {error}")


def display_loaded_datasets():
    """Display loaded datasets and allow selection"""
    if st.session_state.datasets:
        st.subheader("📊 Loaded Datasets")
        
        # Dataset selection
        dataset_names = list(st.session_state.datasets.keys())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_dataset = st.selectbox(
                "Select Dataset for Analysis", 
                dataset_names,
                index=dataset_names.index(st.session_state.current_dataset) if st.session_state.current_dataset in dataset_names else 0,
                key="dataset_selector",
                help="Choose which dataset to analyze in the other tabs"
            )
            
            # Update current dataset and backward compatibility variables
            if selected_dataset != st.session_state.current_dataset:
                st.session_state.current_dataset = selected_dataset
                st.session_state.data = st.session_state.datasets[selected_dataset]
                st.session_state.connector = st.session_state.connectors[selected_dataset]
                st.rerun()
        
        with col2:
            if st.button("🗑️ Remove Selected Dataset", key="remove_dataset_button"):
                if len(st.session_state.datasets) > 1:
                    # Remove the selected dataset
                    del st.session_state.datasets[selected_dataset]
                    del st.session_state.connectors[selected_dataset]
                    
                    # Update current dataset to the first remaining one
                    remaining_datasets = list(st.session_state.datasets.keys())
                    if remaining_datasets:
                        st.session_state.current_dataset = remaining_datasets[0]
                        st.session_state.data = st.session_state.datasets[remaining_datasets[0]]
                        st.session_state.connector = st.session_state.connectors[remaining_datasets[0]]
                    else:
                        st.session_state.current_dataset = None
                        st.session_state.data = None
                        st.session_state.connector = None
                    
                    st.success(f"✅ Removed dataset: {selected_dataset}")
                    st.rerun()
                else:
                    st.warning("⚠️ Cannot remove the last remaining dataset")
        
        # Display current dataset info
        if st.session_state.current_dataset:
            current_df = st.session_state.datasets[st.session_state.current_dataset]
            
            st.write(f"**Currently Selected**: {st.session_state.current_dataset}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(current_df):,}")
            with col2:
                st.metric("Columns", len(current_df.columns))
            with col3:
                st.metric("Memory Usage", f"{current_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            with col4:
                st.metric("Total Datasets", len(st.session_state.datasets))
            
            # Show data preview
            show_data_preview(current_df)
    else:
        st.info("📝 No datasets loaded yet. Please upload some CSV or Excel files above.")


def handle_excel_upload():
    """Handle Excel file upload"""
    st.subheader("📊 Excel File Upload")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
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
                sample_rows = st.number_input("Sample Rows", min_value=100, max_value=10000, value=1000, key="excel_sample_rows")
            else:
                sample_rows = None
            
            if st.button("Load Excel Data", type="primary", key="load_excel_button"):
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
                            
                            # Store in session state
                            dataset_name = f"{uploaded_file.name}_{selected_sheet}"
                            st.session_state.datasets[dataset_name] = df
                            st.session_state.connectors[dataset_name] = connector
                            st.session_state.current_dataset = dataset_name
                            st.session_state.data = df
                            st.session_state.connector = connector
                            
                            # Clean up temp file
                            os.remove(temp_path)
                            
                            st.success(f"✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
                            
                            # Show data preview
                            show_data_preview(df)
                        else:
                            st.error("❌ Failed to connect to Excel file")
                            
                except Exception as e:
                    st.error(f"❌ Error loading Excel: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
        except Exception as e:
            st.error(f"❌ Error reading Excel file: {str(e)}")
