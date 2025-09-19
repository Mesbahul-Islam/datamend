"""
Data Quality Engine - Streamlit Frontend

A simplified data quality management interface focused on:
- Data upload (CSV/Excel)
- Interactive data profiling
- Basic anomaly detection
- AI-driven recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ydata_profiling import ProfileReport
from data_quality.anomaly_detector import StatisticalAnomalyDetector
from connectors.data_connectors import DataConnectorFactory
from llm.analyzer import DataQualityLLMAnalyzer, LLMConfig

# Configure Streamlit page
st.set_page_config(
    page_title="Data Quality Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}  # Dictionary to store multiple datasets
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None  # Currently selected dataset
    if 'data' not in st.session_state:
        st.session_state.data = None  # For backward compatibility
    if 'ydata_profiles' not in st.session_state:
        st.session_state.ydata_profiles = {}  # ydata-profiling reports for each dataset
    if 'ydata_profile' not in st.session_state:
        st.session_state.ydata_profile = None  # Currently selected profile
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'connectors' not in st.session_state:
        st.session_state.connectors = {}  # Connectors for each dataset
    if 'connector' not in st.session_state:
        st.session_state.connector = None  # For backward compatibility
    if 'profiling_complete' not in st.session_state:
        st.session_state.profiling_complete = False

def main():
    """Main application function"""
    st.title("🔍 Data Quality Engine")
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Engine configuration
        chunk_size = st.slider("Chunk Size", min_value=50000, max_value=900000, value=100000, step=50000,
                              help="Size of data chunks for parallel processing", key="main_chunk_size")
        max_workers = st.slider("Max Workers", min_value=1, max_value=8, value=4,
                               help="Number of parallel threads", key="main_max_workers")
        anomaly_threshold = st.slider("Anomaly Threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.1,
                                    help="Z-score threshold for anomaly detection", key="main_anomaly_threshold")
        
        # LLM configuration
        st.subheader("🤖 AI Recommendations")
        use_llm = st.checkbox("Enable AI Recommendations", value=False)
        if use_llm:
            api_key = st.text_input("LLM API Key", type="password", help="Optional: Enter your LLM API key for AI recommendations")
            model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet",], index=0)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Source", "📊 Data Profiling", "🎯 Anomaly Detection", "🤖 AI Recommendations"])
    
    with tab1:
        data_source_tab()
    
    with tab2:
        data_profiling_tab(chunk_size, max_workers, anomaly_threshold)
    
    with tab3:
        anomaly_detection_tab(anomaly_threshold)
    
    with tab4:
        ai_recommendations_tab(use_llm, api_key if 'api_key' in locals() else "", model if 'model' in locals() else "gpt-3.5-turbo")

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

def show_data_preview(df: pd.DataFrame):
    """Show a preview of the loaded data"""
    st.subheader("📋 Data Preview")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
    with col4:
        null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Null Values", f"{null_percentage:.1f}%")
    
    # Data types
    st.subheader("📝 Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(col_info, use_container_width=True)
    
    # Data sample
    st.subheader("🔍 Data Sample")
    st.dataframe(df.head(10), use_container_width=True)

def data_profiling_tab(chunk_size: int, max_workers: int, anomaly_threshold: float):
    """Data profiling tab"""
    st.header("📊 Data Profiling")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Source tab")
        return
    
    # Show current dataset info
    if st.session_state.current_dataset:
        st.info(f"📊 **Analyzing Dataset**: {st.session_state.current_dataset}")
    
    df = st.session_state.data
    
    # Profiling controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Calculate dataset size information
        dataset_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        estimated_processing_time = len(df) * 0.00005  # Rough estimate: 0.05ms per row

        # Determine processing strategy message
        if len(df) > 100000:
            processing_strategy = "🚀 Large dataset - using parallel processing"
        else:
            processing_strategy = "⚡ Standard dataset - using optimized sequential processing"
        
        st.write(f"**Dataset:** {len(df):,} rows × {len(df.columns)} columns ({dataset_size_mb:.1f} MB)")
        st.caption(f"{processing_strategy} • Est. processing time: ~{estimated_processing_time:.1f}s")
    
    with col2:
        if st.button("🔄 Run Profiling", type="primary", key="run_profiling_button"):
            run_data_profiling(df, chunk_size, max_workers, anomaly_threshold)
    
    # Show results if available
    if st.session_state.get('ydata_profile'):
        display_ydata_profiling_results(st.session_state.ydata_profile)

def run_data_profiling(df: pd.DataFrame, chunk_size: int, max_workers: int, anomaly_threshold: float):
    """Run data profiling on the dataset using ydata-profiling"""
    try:
        with st.spinner("🔍 Running comprehensive data profiling with ydata-profiling..."):
            # Run profiling with timing
            start_time = time.time()
            
            # Create profile report with ydata-profiling
            profile = ProfileReport(
                df, 
                title=f"Data Profile Report - {st.session_state.current_dataset or 'Dataset'}",
                explorative=True,
                minimal=False
            )
            
            end_time = time.time()
            profiling_time = end_time - start_time
                
            st.session_state.ydata_profile = profile
            st.session_state.profiling_time = profiling_time
            st.session_state.profiling_complete = True
            
        st.success(f"✅ Data profiling completed successfully in {st.session_state.profiling_time:.2f} seconds!")
        
    except Exception as e:
        st.error(f"❌ Error during profiling: {str(e)}")
        st.error(f"Details: {type(e).__name__}: {str(e)}")

def display_ydata_profiling_results(profile):
    """Display ydata-profiling results with interactive HTML report and summary"""
    
    # Executive Summary from ydata-profiling
    st.subheader("📈 Data Profiling Report")
    
    # Display profiling time if available
    if hasattr(st.session_state, 'profiling_time') and st.session_state.profiling_time:
        profiling_time = st.session_state.profiling_time
        if profiling_time < 1:
            time_display = f"{profiling_time*1000:.0f}ms"
            time_status = "⚡ Fast"
        elif profiling_time < 5:
            time_display = f"{profiling_time:.2f}s"
            time_status = "🟢 Quick"
        elif profiling_time < 15:
            time_display = f"{profiling_time:.1f}s"
            time_status = "🟡 Normal"
        else:
            time_display = f"{profiling_time:.1f}s"
            time_status = "🟠 Slow"
        st.info(f"**Profiling Time**: {time_display} ({time_status})")
    
    # Display options
    st.subheader("📊 Report Display Options")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Quick Summary", "📄 Detailed Report", "💾 Export Options"])
    
    with tab1:
        display_ydata_summary(profile)
    
    with tab2:
        display_enhanced_report(profile)
    
    with tab3:
        display_export_options(profile)

def display_ydata_summary(profile):
    """Display a quick summary of the ydata-profiling results"""
    st.subheader("📋 Quick Data Summary")
    
    # Get basic statistics from the profile
    description = profile.description_set
    table_stats = description.table
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_vars = table_stats.get('n_var', 0)
        st.metric("Variables", n_vars)
    
    with col2:
        n_obs = table_stats.get('n', 0)
        st.metric("Observations", f"{n_obs:,}")
    
    with col3:
        missing_cells = table_stats.get('n_cells_missing', 0)
        # Calculate total cells properly: rows * columns
        n_obs = table_stats.get('n', 0)
        n_vars = table_stats.get('n_var', 0)
        total_cells = n_obs * n_vars if n_obs > 0 and n_vars > 0 else 1
        missing_percent = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        # Alternative: use the percentage directly from ydata-profiling
        # missing_percent = table_stats.get('p_cells_missing', 0) * 100
        st.metric("Missing Cells", f"{missing_percent:.1f}%")
    
    with col4:
        duplicate_rows = table_stats.get('n_duplicates', 0)
        duplicate_percent = (duplicate_rows / n_obs) * 100 if n_obs > 0 else 0
        st.metric("Duplicate Rows", f"{duplicate_percent:.1f}%")
    
    # Variable types
    st.subheader("📊 Variable Types")
    
    types_summary = table_stats.get('types', {})
    if types_summary:
        types_df = pd.DataFrame([
            {"Type": type_name.replace('_', ' ').title(), "Count": count}
            for type_name, count in types_summary.items()
        ])
        st.dataframe(types_df, use_container_width=True, hide_index=True)
    
    # Warnings and alerts
    st.subheader("⚠️ Data Quality Alerts")
    
    alerts = []
    
    if missing_percent > 10:
        alerts.append(f"🔸 High missing data: {missing_percent:.1f}% of cells are missing")
    
    if duplicate_percent > 5:
        alerts.append(f"🔸 Duplicate rows detected: {duplicate_percent:.1f}% of rows are duplicates")
    
    n_constant = table_stats.get('n_constant', 0)
    if n_constant > 0:
        alerts.append(f"🔸 Constant variables: {n_constant} variables have only one unique value")
    
    if not alerts:
        st.success("✅ No major data quality issues detected!")
    else:
        for alert in alerts:
            st.warning(alert)

def display_full_html_report(profile):
    """Display the full HTML report from ydata-profiling"""
    st.subheader("� Full Profiling Report")
    
    try:
        # Generate the HTML report
        html_report = profile.to_html()
        
        # Display in Streamlit using components
        st.components.v1.html(html_report, height=800, scrolling=True)
        
    except Exception as e:
        st.error(f"Error generating HTML report: {str(e)}")
        st.info("Try using the 'Show Quick Summary' option instead")

def download_html_report(profile):
    """Provide download link for the HTML report"""
    st.subheader("💾 Download Report")
    
    try:
        # Generate the HTML report
        html_report = profile.to_html()
        
        # Create download button
        dataset_name = st.session_state.current_dataset or "dataset"
        filename = f"data_profile_report_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        st.download_button(
            label="📥 Download HTML Report",
            data=html_report,
            file_name=filename,
            mime="text/html",
            help="Download the complete profiling report as an HTML file"
        )
        
        st.success("Report ready for download! Click the button above to save the HTML file.")
        
    except Exception as e:
        st.error(f"Error preparing download: {str(e)}")

def display_enhanced_report(profile):
    """Display an enhanced report with better formatting"""
    st.subheader("📊 Detailed Data Analysis")
    
    try:
        # Get description for detailed analysis
        description = profile.description_set
        table_stats = description.table
        variables = description.variables
        
        # Dataset Overview Section
        st.markdown("### 📈 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Variables", table_stats.get('n_var', 0))
        with col2:
            st.metric("Total Observations", f"{table_stats.get('n', 0):,}")
        with col3:
            missing_cells = table_stats.get('n_cells_missing', 0)
            # Calculate total cells properly: rows * columns
            n_obs = table_stats.get('n', 0)
            n_vars = table_stats.get('n_var', 0)
            total_cells = n_obs * n_vars if n_obs > 0 and n_vars > 0 else 1
            missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        with col4:
            duplicates = table_stats.get('n_duplicates', 0)
            duplicate_pct = (duplicates / table_stats.get('n', 1)) * 100 if table_stats.get('n', 0) > 0 else 0
            st.metric("Duplicate Rows", f"{duplicate_pct:.1f}%")
        
        # Variable Types Analysis
        st.markdown("### 🔢 Variable Types Distribution")
        types_summary = table_stats.get('types', {})
        if types_summary:
            types_df = pd.DataFrame([
                {"Variable Type": type_name.replace('_', ' ').title(), "Count": count, "Percentage": f"{(count/sum(types_summary.values()))*100:.1f}%"}
                for type_name, count in types_summary.items()
            ])
            st.dataframe(types_df, use_container_width=True, hide_index=True)
        
        # Variable Details Analysis
        st.markdown("### 📋 Variable Analysis")
        
        if variables:
            # Create detailed variable analysis
            variable_details = []
            
            for var_name, var_info in variables.items():
                var_type = var_info.get('type', 'Unknown')
                n_missing = var_info.get('n_missing', 0)
                n_total = var_info.get('n', 0)
                missing_pct = (n_missing / n_total * 100) if n_total > 0 else 0
                n_unique = var_info.get('n_unique', 0)
                
                # Determine status
                if missing_pct > 50:
                    status = "🔴 Critical"
                elif missing_pct > 20:
                    status = "🟠 Warning"
                elif missing_pct > 5:
                    status = "🟡 Attention"
                else:
                    status = "🟢 Good"
                
                # Get key statistics based on type
                key_stats = ""
                if var_type == "Numeric":
                    mean_val = var_info.get('mean', 0)
                    std_val = var_info.get('std', 0)
                    key_stats = f"Mean: {mean_val:.2f}, Std: {std_val:.2f}" if mean_val and std_val else "Basic stats available"
                elif var_type == "Categorical":
                    n_categories = var_info.get('n_distinct', 0)
                    key_stats = f"{n_categories} categories"
                elif var_type == "Text":
                    max_length = var_info.get('max_length', 0)
                    key_stats = f"Max length: {max_length}" if max_length else "Text analysis available"
                
                variable_details.append({
                    "Variable": var_name,
                    "Type": var_type,
                    "Status": status,
                    "Missing %": f"{missing_pct:.1f}%",
                    "Unique Values": f"{n_unique:,}",
                    "Key Statistics": key_stats
                })
            
            # Display variable details table
            variables_df = pd.DataFrame(variable_details)
            st.dataframe(variables_df, use_container_width=True, hide_index=True)
        
        # Data Quality Issues
        st.markdown("### ⚠️ Data Quality Assessment")
        
        quality_issues = []
        
        # Check for high missing data
        high_missing_vars = [name for name, info in variables.items() 
                           if (info.get('n_missing', 0) / info.get('n', 1)) > 0.1]
        if high_missing_vars:
            quality_issues.append(f"🔸 High missing data in {len(high_missing_vars)} variables: {', '.join(high_missing_vars[:3])}{'...' if len(high_missing_vars) > 3 else ''}")
        
        # Check for duplicate rows
        if duplicate_pct > 5:
            quality_issues.append(f"🔸 Duplicate rows detected: {duplicate_pct:.1f}% of the dataset")
        
        # Check for constant variables
        constant_vars = [name for name, info in variables.items() 
                        if info.get('n_distinct', 0) <= 1]
        if constant_vars:
            quality_issues.append(f"🔸 Constant variables found: {', '.join(constant_vars)}")
        
        if quality_issues:
            for issue in quality_issues:
                st.warning(issue)
        else:
            st.success("✅ No major data quality issues detected!")
        
        # Full HTML Report Option
        st.markdown("### 📄 Complete Interactive Report")
        st.info("For the most comprehensive analysis with interactive visualizations, you can generate the full HTML report.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔍 Generate Interactive Report", type="primary", use_container_width=True):
                with st.spinner("Generating comprehensive HTML report..."):
                    html_report = profile.to_html()
                    # Use a larger iframe for better display and remove width restriction
                    st.components.v1.html(html_report, width=None, height=1200, scrolling=True)
        
        with col2:
            if st.button("💾 Download for Best View", type="secondary", use_container_width=True):
                html_report = profile.to_html()
                dataset_name = st.session_state.current_dataset or "dataset"
                filename = f"data_profile_report_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                st.download_button(
                    label="📥 Download HTML Report",
                    data=html_report,
                    file_name=filename,
                    mime="text/html",
                    help="Download for optimal viewing experience",
                    use_container_width=True
                )
        
    except Exception as e:
        st.error(f"Error displaying enhanced report: {str(e)}")
        st.info("Please try the Quick Summary view or download the report for offline viewing.")

def display_export_options(profile):
    """Provide various export and sharing options"""
    st.subheader("💾 Export & Share Options")
    
    try:
        # Generate the HTML report
        html_report = profile.to_html()
        
        # Create download button
        dataset_name = st.session_state.current_dataset or "dataset"
        filename = f"data_profile_report_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.download_button(
                label="📥 Download Complete HTML Report",
                data=html_report,
                file_name=filename,
                mime="text/html",
                help="Download the complete profiling report as an HTML file",
                use_container_width=True
            )
        
        with col2:
            file_size_mb = len(html_report.encode('utf-8')) / (1024 * 1024)
            st.metric("Report Size", f"{file_size_mb:.1f} MB")
        
        st.success("📄 Report ready for download! The HTML file contains:")
        
        features = [
            "🔍 Interactive data exploration",
            "📊 Comprehensive statistical analysis", 
            "📈 Distribution plots and histograms",
            "🔗 Correlation matrices",
            "⚠️ Data quality warnings",
            "📋 Missing data patterns",
            "🎯 Outlier detection results",
            "📱 Mobile-responsive design"
        ]
        
        col1, col2 = st.columns(2)
        for i, feature in enumerate(features):
            if i % 2 == 0:
                col1.write(feature)
            else:
                col2.write(feature)
        
        st.info("💡 **Tip**: Open the downloaded HTML file in your browser for the best viewing experience with full interactivity.")
        
    except Exception as e:
        st.error(f"Error preparing export options: {str(e)}")
    
    # Quality interpretation
    quality_score = report.overall_quality_score
    if quality_score >= 90:
        quality_status = "🟢 **Excellent**"
        quality_message = "Your data is in excellent condition with minimal issues."
    elif quality_score >= 75:
        quality_status = "🟡 **Good**"
        quality_message = "Your data quality is good but has some areas for improvement."
    elif quality_score >= 50:
        quality_status = "🟠 **Fair**"
        quality_message = "Your data has moderate quality issues that should be addressed."
    else:
        quality_status = "🔴 **Poor**"
        quality_message = "Your data has significant quality issues requiring immediate attention."
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Overall Quality", f"{quality_score:.1f}/100", delta=quality_status.split()[1])
    
    with col2:
        critical_issues = len(report.critical_issues)
        st.metric("Critical Issues", critical_issues, delta="Fix Immediately" if critical_issues > 0 else "✅ None")
    
    with col3:
        complete_columns = sum(1 for profile in report.column_profiles.values() if profile.null_percentage < 5)
        st.metric("Complete Columns", f"{complete_columns}/{len(report.column_profiles)}")
    
    with col4:
        problematic_columns = sum(1 for profile in report.column_profiles.values() if len(profile.data_quality_issues or []) > 0)
        st.metric("Columns with Issues", problematic_columns)
        
    with col5:
        # Display profiling time if available
        if hasattr(st.session_state, 'profiling_time') and st.session_state.profiling_time:
            profiling_time = st.session_state.profiling_time
            if profiling_time < 1:
                time_display = f"{profiling_time*1000:.0f}ms"
                time_status = "⚡ Fast"
            elif profiling_time < 5:
                time_display = f"{profiling_time:.2f}s"
                time_status = "🟢 Quick"
            elif profiling_time < 15:
                time_display = f"{profiling_time:.1f}s"
                time_status = "🟡 Normal"
            else:
                time_display = f"{profiling_time:.1f}s"
                time_status = "🟠 Slow"
            st.metric("Profiling Time", time_display, delta=time_status)
        else:
            st.metric("Profiling Time", "N/A")
    
    st.info(f"**Assessment**: {quality_message}")
    
    # Critical Issues Alert
    if report.critical_issues:
        st.error("🚨 **Critical Issues Requiring Immediate Attention:**")
        for i, issue in enumerate(report.critical_issues, 1):
            st.write(f"{i}. {issue}")
        st.write("")
    
    # Column Health Overview
    st.subheader("📋 Column Health Assessment")
    
    if report.column_profiles:
        columns_summary = []
        for col_name, col_profile in report.column_profiles.items():
            # Calculate column health score
            col_health_score = 100
            if col_profile.null_percentage > 0:
                col_health_score -= min(col_profile.null_percentage * 0.8, 40)  # Max 40 points deduction
            if col_profile.data_quality_issues:
                col_health_score -= len(col_profile.data_quality_issues) * 15  # 15 points per issue
            col_health_score = max(0, col_health_score)
            
            # Health status
            if col_health_score >= 85:
                health_icon = "🟢"
                health_status = "Healthy"
            elif col_health_score >= 65:
                health_icon = "🟡"
                health_status = "Minor Issues"
            elif col_health_score >= 40:
                health_icon = "🟠" 
                health_status = "Needs Attention"
            else:
                health_icon = "🔴"
                health_status = "Critical"
            
            # Key insights about the column
            insights = []
            if col_profile.null_percentage > 20:
                insights.append(f"High missing data ({col_profile.null_percentage:.1f}%)")
            elif col_profile.null_percentage > 5:
                insights.append(f"Some missing data ({col_profile.null_percentage:.1f}%)")
            
            if col_profile.outliers_count > 0:
                insights.append(f"{col_profile.outliers_count} outliers detected")
            
            if col_profile.data_quality_issues:
                for issue in col_profile.data_quality_issues:
                    if "INCONSISTENT_CASING" in issue:
                        insights.append("Mixed text casing")
                    elif "WHITESPACE" in issue:
                        insights.append("Extra spaces detected")
                    elif "LOW_UNIQUENESS" in issue:
                        insights.append("Low data diversity")
            
            # Uniqueness insight
            if col_profile.unique_count == col_profile.total_count:
                uniqueness_note = "All unique"
            elif col_profile.unique_count == 1:
                uniqueness_note = "Single value"
            else:
                uniqueness_rate = (col_profile.unique_count / col_profile.total_count) * 100
                uniqueness_note = f"{uniqueness_rate:.1f}% unique"
            
            columns_summary.append({
                'Column': f"{health_icon} **{col_name}**",
                'Type': col_profile.data_type.title(),
                'Health Status': health_status,
                'Completeness': f"{100-col_profile.null_percentage:.1f}%",
                'Uniqueness': uniqueness_note,
                'Key Insights': "; ".join(insights) if insights else "No issues detected"
            })
        
        columns_df = pd.DataFrame(columns_summary)
        st.dataframe(columns_df, use_container_width=True, hide_index=True)
        
        # Data Quality Insights
        st.subheader("� Key Data Quality Insights")
        
        # Missing data analysis
        high_missing_cols = [col for col, profile in report.column_profiles.items() 
                           if profile.null_percentage > 10]
        if high_missing_cols:
            st.warning(f"**Missing Data Concern**: {len(high_missing_cols)} columns have >10% missing values: {', '.join(high_missing_cols)}")
        
        # Data type distribution
        type_counts = {}
        for profile in report.column_profiles.values():
            type_counts[profile.data_type] = type_counts.get(profile.data_type, 0) + 1
        
        st.info(f"**Data Composition**: {type_counts}")
        
        # Outlier summary
        total_outliers = sum(profile.outliers_count for profile in report.column_profiles.values())
        if total_outliers > 0:
            outlier_cols = [col for col, profile in report.column_profiles.items() if profile.outliers_count > 0]
            st.warning(f"**Outliers Detected**: {total_outliers} outliers found across {len(outlier_cols)} columns")
        else:
            st.success("**No Statistical Outliers**: No obvious outliers detected in numeric columns")
        
        # Quick action recommendations
        st.subheader("🎯 Immediate Action Items")
        
        actions = []
        if high_missing_cols:
            actions.append(f"🔸 Investigate missing data in: {', '.join(high_missing_cols[:3])}")
        
        inconsistent_cols = [col for col, profile in report.column_profiles.items() 
                           if profile.data_quality_issues and any("INCONSISTENT" in issue for issue in profile.data_quality_issues)]
        if inconsistent_cols:
            actions.append(f"🔸 Standardize formatting in: {', '.join(inconsistent_cols[:3])}")
        
        if total_outliers > 0:
            actions.append(f"🔸 Review outliers in numeric columns (check Anomaly Detection tab)")
        
        if not actions:
            st.success("✅ **No immediate actions required** - Your data quality looks good!")
        else:
            for action in actions:
                st.write(action)

def anomaly_detection_tab(anomaly_threshold: float):
    """Anomaly detection tab"""
    st.header("🎯 Anomaly Detection")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Source tab")
        return
    
    # Show current dataset info
    if st.session_state.current_dataset:
        st.info(f"🎯 **Analyzing Dataset**: {st.session_state.current_dataset}")
    
    df = st.session_state.data
    
    # Anomaly detection controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        detection_method = st.selectbox(
            "Detection Method",
            ["Z-Score", "IQR", "Modified Z-Score", "All Methods"],
            help="Choose the statistical method for anomaly detection"
        )
    
    with col2:
        if st.button("🔍 Detect Anomalies", type="primary", key="detect_anomalies_button"):
            st.session_state.anomaly_method = detection_method
            st.session_state.run_anomaly_detection = True
    
    # Display results outside of columns for full width
    if getattr(st.session_state, 'run_anomaly_detection', False):
        run_anomaly_detection(df, st.session_state.anomaly_method, anomaly_threshold)
        st.session_state.run_anomaly_detection = False

def run_anomaly_detection(df: pd.DataFrame, method: str, threshold: float):
    """Run anomaly detection"""
    try:
        with st.spinner("🔍 Detecting anomalies..."):
            detector = StatisticalAnomalyDetector(z_threshold=threshold)
            
            # Start timing
            start_time = time.time()
            
            if method == "All Methods":
                results = detector.detect_anomalies(df)
            else:
                # Map method names to detector methods
                method_map = {
                    "Z-Score": "zscore",
                    "IQR": "iqr", 
                    "Modified Z-Score": "modified_zscore"
                }
                results = detector.detect_anomalies(df, methods=[method_map[method]])
            
            # End timing
            end_time = time.time()
            anomaly_detection_time = end_time - start_time
            
            # Store timing in session state
            st.session_state.anomaly_detection_time = anomaly_detection_time
            
            display_anomaly_results(results, df)
            
    except Exception as e:
        st.error(f"❌ Error during anomaly detection: {str(e)}")

def create_anomaly_visualizations(df: pd.DataFrame, column: str, anomaly_result: Any):
    """Create scatter plot visualization for anomalies in a specific column"""
    
    # Set style for better looking plots
    plt.style.use('default')
    
    if not hasattr(anomaly_result, 'anomaly_indices') or not anomaly_result.anomaly_indices:
        return None
        
    # Get data for the column
    column_data = df[column].dropna()
    if len(column_data) == 0:
        return None
        
    # Create a single scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(f'Anomaly Detection: {column}', fontsize=16, fontweight='bold')
    
    # Get anomaly indices and values
    anomaly_indices = anomaly_result.anomaly_indices
    normal_indices = [i for i in column_data.index if i not in anomaly_indices]
    
    # Scatter plot with highlighted anomalies
    ax.scatter(normal_indices, column_data.loc[normal_indices], 
               alpha=0.6, c='blue', label='Normal Data', s=30)
    if anomaly_indices:
        ax.scatter(anomaly_indices, column_data.loc[anomaly_indices], 
                   alpha=0.9, c='red', label='Anomalies', s=80, marker='X')
    
    ax.set_title('Data Points with Anomalies Highlighted', fontweight='bold')
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel(f'{column} Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add some statistics as text
    if anomaly_indices:
        anomaly_count = len(anomaly_indices)
        total_count = len(column_data)
        anomaly_percentage = (anomaly_count / total_count) * 100
        
        stats_text = f'Anomalies: {anomaly_count}/{total_count} ({anomaly_percentage:.1f}%)'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                verticalalignment='top', fontweight='bold')
    
    plt.tight_layout()
    return fig

def display_anomaly_results(results: Dict[str, Dict[str, Any]], df: pd.DataFrame):
    """Display anomaly detection results with meaningful insights"""
    
    # Use a container for full width display
    with st.container():
        st.subheader("🎯 Anomaly Detection Assessment")
        
        # Display timing information if available
        if hasattr(st.session_state, 'anomaly_detection_time') and st.session_state.anomaly_detection_time:
            detection_time = st.session_state.anomaly_detection_time
            if detection_time < 0.1:
                time_display = f"⚡ Completed in {detection_time*1000:.0f}ms"
            elif detection_time < 1:
                time_display = f"🟢 Completed in {detection_time*1000:.0f}ms"
            elif detection_time < 5:
                time_display = f"🟡 Completed in {detection_time:.2f}s"
            else:
                time_display = f"🟠 Completed in {detection_time:.1f}s"
            st.info(time_display)
        
        if not results:
            st.info("No numeric columns found for anomaly detection")
            return
        
        # Calculate comprehensive anomaly statistics
        total_anomalies = 0
        anomaly_by_column = {}
        severity_analysis = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
        
        for column, column_results in results.items():
            column_anomalies = set()
            max_severity_score = 0
            
            for method, anomaly_result in column_results.items():
                if hasattr(anomaly_result, 'anomaly_indices'):
                    column_anomalies.update(anomaly_result.anomaly_indices)
                    # Calculate severity based on anomaly scores
                    if hasattr(anomaly_result, 'anomaly_scores') and anomaly_result.anomaly_scores:
                        max_score = max(anomaly_result.anomaly_scores) if anomaly_result.anomaly_scores else 0
                        max_severity_score = max(max_severity_score, max_score)
            
            anomaly_count = len(column_anomalies)
            anomaly_by_column[column] = {
                'count': anomaly_count,
                'percentage': (anomaly_count / len(df)) * 100 if len(df) > 0 else 0,
                'severity_score': max_severity_score
            }
            total_anomalies += anomaly_count
            
            # Categorize severity
            if anomaly_count > 0:
                anomaly_rate = (anomaly_count / len(df)) * 100
                if anomaly_rate > 10:
                    severity_analysis["Critical"] += 1
                elif anomaly_rate > 5:
                    severity_analysis["High"] += 1
                elif anomaly_rate > 1:
                    severity_analysis["Medium"] += 1
                else:
                    severity_analysis["Low"] += 1
        
        # Executive Summary
        total_anomaly_rate = (total_anomalies / len(df)) * 100 if len(df) > 0 else 0
        
        if total_anomaly_rate == 0:
            assessment_status = "🟢 **Excellent**"
            assessment_message = "No anomalies detected. Your data appears statistically normal."
        elif total_anomaly_rate < 1:
            assessment_status = "🟡 **Good**"  
            assessment_message = "Very few anomalies detected. This is typically expected in real data."
        elif total_anomaly_rate < 5:
            assessment_status = "🟠 **Moderate**"
            assessment_message = "Some anomalies detected. Review these values to determine if they're valid or errors."
        else:
            assessment_status = "🔴 **High**"
            assessment_message = "Many anomalies detected. This may indicate data quality issues or unusual patterns."
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Anomalies", total_anomalies, delta=f"{total_anomaly_rate:.2f}% of data")
        
        with col2:
            affected_columns = len([col for col, data in anomaly_by_column.items() if data['count'] > 0])
            st.metric("Affected Columns", f"{affected_columns}/{len(results)}")
        
        with col3:
            if severity_analysis["Critical"] > 0:
                severity_status = "Critical"
                severity_color = "🔴"
            elif severity_analysis["High"] > 0:
                severity_status = "High"
                severity_color = "🟠"
            elif severity_analysis["Medium"] > 0:
                severity_status = "Medium"
                severity_color = "🟡"
            else:
                severity_status = "Low"
                severity_color = "🟢"
            st.metric("Max Severity", f"{severity_color} {severity_status}")
        
        with col4:
            # Data reliability score based on anomaly rate
            reliability = max(0, 100 - (total_anomaly_rate * 4))  # Reduce by 4 points per % anomaly
            st.metric("Data Reliability", f"{reliability:.0f}/100")
        
        st.info(f"**Assessment**: {assessment_message}")
        
        # Detailed Findings
        if total_anomalies > 0:
            st.subheader("🔍 Detailed Anomaly Analysis")
            
            # Priority columns (sorted by severity)
            priority_columns = []
            for col, data in anomaly_by_column.items():
                if data['count'] > 0:
                    priority_columns.append((col, data))
            
            priority_columns.sort(key=lambda x: x[1]['percentage'], reverse=True)
            
            if priority_columns:
                st.write("**Columns requiring attention (ordered by severity):**")
                
                for i, (column, data) in enumerate(priority_columns[:5], 1):  # Show top 5
                    # Determine priority level
                    if data['percentage'] > 10:
                        priority_icon = "🚨"
                        priority_level = "CRITICAL"
                    elif data['percentage'] > 5:
                        priority_icon = "⚠️"
                        priority_level = "HIGH"
                    elif data['percentage'] > 1:
                        priority_icon = "🟡"
                        priority_level = "MEDIUM"
                    else:
                        priority_icon = "🔵"
                        priority_level = "LOW"
                    
                    st.write(f"{i}. {priority_icon} **{column}** - {data['count']} anomalies ({data['percentage']:.2f}%) - Priority: {priority_level}")
            
            # Method-specific insights
            st.subheader("📊 Detection Method Results")
            
            method_summary = {}
            for column, column_results in results.items():
                for method, anomaly_result in column_results.items():
                    if hasattr(anomaly_result, 'total_anomalies'):
                        if method not in method_summary:
                            method_summary[method] = {'columns': 0, 'total_anomalies': 0}
                        if anomaly_result.total_anomalies > 0:
                            method_summary[method]['columns'] += 1
                            method_summary[method]['total_anomalies'] += anomaly_result.total_anomalies
            
            if method_summary:
                for method, stats in method_summary.items():
                    method_name = method.replace('_', ' ').title()
                    if stats['total_anomalies'] > 0:
                        st.write(f"**{method_name}**: Found {stats['total_anomalies']} anomalies across {stats['columns']} columns")
                    else:
                        st.write(f"**{method_name}**: No anomalies detected")
            
            # Actionable recommendations
            st.subheader("🎯 Recommended Actions")
            
            recommendations = []
            
            if severity_analysis["Critical"] > 0:
                recommendations.append("🚨 **Immediate Action**: Investigate critical anomalies - they may indicate data corruption or system errors")
            
            if severity_analysis["High"] > 0:
                recommendations.append("⚠️ **High Priority**: Review high-severity anomalies to determine if they represent valid edge cases or errors")
            
            high_anomaly_columns = [col for col, data in anomaly_by_column.items() if data['percentage'] > 5]
            if high_anomaly_columns:
                recommendations.append(f"🔍 **Data Validation**: Columns with >5% anomalies need validation: {', '.join(high_anomaly_columns[:3])}")
            
            if total_anomaly_rate > 2:
                recommendations.append("📊 **Process Review**: High overall anomaly rate suggests reviewing data collection or processing procedures")
            
            if recommendations:
                for rec in recommendations:
                    st.write(rec)
            else:
                st.success("✅ **No immediate action required** - Anomaly levels are within normal ranges")
            
            # Detailed breakdown for top problematic columns
            critical_columns = [col for col, data in anomaly_by_column.items() if data['percentage'] > 5]
            if critical_columns:
                st.subheader("🔬 Critical Column Analysis")
                
                for column in critical_columns[:3]:  # Show top 3 critical columns
                    with st.expander(f"🔍 **{column}** - Detailed Analysis"):
                        column_data = anomaly_by_column[column]
                        st.write(f"**Anomaly Rate**: {column_data['percentage']:.2f}% ({column_data['count']} out of {len(df)} records)")
                        
                        # Show results by method for this column
                        column_results = results[column]
                        best_method_result = None
                        best_method_name = None
                        max_anomalies = 0
                        
                        for method, anomaly_result in column_results.items():
                            if hasattr(anomaly_result, 'total_anomalies') and anomaly_result.total_anomalies > 0:
                                st.write(f"**{method.upper()}**: {anomaly_result.total_anomalies} anomalies")
                                
                                # Track the method with most anomalies for visualization
                                if anomaly_result.total_anomalies > max_anomalies:
                                    max_anomalies = anomaly_result.total_anomalies
                                    best_method_result = anomaly_result
                                    best_method_name = method
                                
                                # Show sample anomalous values with context
                                if hasattr(anomaly_result, 'anomaly_values') and anomaly_result.anomaly_values:
                                    sample_values = anomaly_result.anomaly_values[:3]
                                    st.write(f"Sample anomalous values: {sample_values}")
                                    
                                    # Show typical values for comparison
                                    normal_mask = ~df.index.isin(anomaly_result.anomaly_indices)
                                    if normal_mask.any():
                                        normal_sample = df.loc[normal_mask, column].dropna().head(3).tolist()
                                        st.write(f"Typical values for comparison: {normal_sample}")
                        
                        # Add detailed visualization for this column
                        if best_method_result:
                            st.write(f"**📊 Detailed Visualization (using {best_method_name.upper()} method)**")
                            
                            try:
                                col_fig = create_anomaly_visualizations(df, column, best_method_result)
                                if col_fig:
                                    st.pyplot(col_fig, use_container_width=True)
                                    plt.close(col_fig)  # Close to free memory
                                else:
                                    st.info("Unable to create visualization for this column.")
                            except Exception as viz_error:
                                st.warning(f"Could not create visualization: {str(viz_error)}")
        
        else:
            st.success("🎉 **Excellent News!** No statistical anomalies detected in your numeric data.")
            st.info("This suggests your data has consistent patterns and no obvious outliers or data quality issues.")
            
            # Show what was analyzed
            analyzed_columns = list(results.keys())
            if analyzed_columns:
                st.write(f"**Columns analyzed**: {', '.join(analyzed_columns)}")
                st.write("**Methods used**: Z-score, IQR (Interquartile Range), Modified Z-score")


def ai_recommendations_tab(use_llm: bool, api_key: str, model: str):
    """AI recommendations tab"""
    st.header("🤖 AI Recommendations")
    
    if not use_llm:
        st.info("💡 Enable AI Recommendations in the sidebar to get intelligent suggestions")
        return
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Source tab")
        return
    
    # Show current dataset info
    if st.session_state.current_dataset:
        st.info(f"🤖 **Analyzing Dataset**: {st.session_state.current_dataset}")
    
    if not st.session_state.get('ydata_profile'):
        st.warning("⚠️ Please run data profiling first")
        return
    
    # Recommendations controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Generate AI-powered recommendations based on your data quality analysis")
    
    with col2:
        if st.button("🤖 Get Recommendations", type="primary", key="get_recommendations_button"):
            generate_recommendations(api_key, model)
    
    # Show recommendations if available
    if st.session_state.recommendations:
        display_recommendations(st.session_state.recommendations)

def generate_recommendations(api_key: str, model: str):
    """Generate AI recommendations"""
    try:
        with st.spinner("🤖 Generating AI recommendations..."):
            if not api_key:
                # Use mock recommendations if no API key
                mock_recommendations = {
                    "recommendations": [
                        {
                            "type": "data_quality",
                            "priority": "high",
                            "title": "Address Missing Values",
                            "description": "Several columns have significant missing values that could impact analysis quality.",
                            "suggested_actions": [
                                "Consider imputation strategies for numerical columns",
                                "Investigate the root cause of missing data",
                                "Document data collection processes"
                            ]
                        },
                        {
                            "type": "data_validation",
                            "priority": "medium", 
                            "title": "Standardize Data Types",
                            "description": "Some columns may benefit from consistent data type formatting.",
                            "suggested_actions": [
                                "Convert string numbers to numeric types",
                                "Standardize date formats",
                                "Review categorical variable encoding"
                            ]
                        }
                    ],
                    "summary": "Your dataset shows good overall quality with some areas for improvement in completeness and consistency."
                }
                st.session_state.recommendations = mock_recommendations
            else:
                # Use actual LLM API
                config = LLMConfig(
                    provider="openai",
                    model=model,
                    api_key=api_key
                )
                
                analyzer = DataQualityLLMAnalyzer(config)
                recommendations = analyzer.analyze_data_quality(
                    st.session_state.data,
                    st.session_state.ydata_profile
                )
                st.session_state.recommendations = recommendations
        
        st.success("✅ AI recommendations generated successfully!")
        
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {str(e)}")

def display_recommendations(recommendations: Dict[str, Any]):
    """Display AI recommendations"""
    st.subheader("💡 AI-Powered Recommendations")
    
    # Summary
    if 'summary' in recommendations:
        st.info(f"📋 **Summary:** {recommendations['summary']}")
    
    # Individual recommendations
    if 'recommendations' in recommendations:
        for i, rec in enumerate(recommendations['recommendations'], 1):
            priority = rec.get('priority', 'medium')
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "🟡")
            
            with st.expander(f"{priority_color} {rec.get('title', f'Recommendation {i}')}"):
                st.write(f"**Priority:** {priority.title()}")
                st.write(f"**Type:** {rec.get('type', 'General').replace('_', ' ').title()}")
                
                if 'description' in rec:
                    st.write(f"**Description:** {rec['description']}")
                
                if 'suggested_actions' in rec and rec['suggested_actions']:
                    st.write("**Suggested Actions:**")
                    for action in rec['suggested_actions']:
                        st.write(f"• {action}")

if __name__ == "__main__":
    main()
