"""
Data Source Tab - Handle various data source types (CSV, Excel, Snowflake)

This module has been refactored into smaller, more manageable modules.
The main functionality is now in src/ui/data_sources/
"""

import streamlit as st
import pandas as pd
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the new modular structure
from src.ui.data_sources import render_data_sources_page


def data_source_tab():
    """Main data source tab - now using modular structure"""
    render_data_sources_page()


def data_overview_tab():
    """Data overview tab - shows current dataset information"""
    if 'data' not in st.session_state or st.session_state.data is None:
        st.info("No data loaded. Please load data from the Data Sources tab first.")
        return
    
    df = st.session_state.data
    current_name = st.session_state.get('current_dataset', 'Unnamed Dataset')
    
    st.header(f"📊 Data Overview: {current_name}")
    
    # Basic information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
    with col4:
        missing_count = df.isnull().sum().sum()
        st.metric("Missing Values", missing_count)
    
    # Column information
    st.subheader("📋 Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(col_info, width='stretch')
    
    # Data preview
    st.subheader("👀 Data Preview")
    preview_rows = st.slider("Rows to preview", 5, min(100, len(df)), 10)
    st.dataframe(df.head(preview_rows), width='stretch')
    
    # Data types summary
    st.subheader("📊 Data Types Summary")
    dtype_summary = df.dtypes.value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(dtype_summary)
    with col2:
        for dtype, count in dtype_summary.items():
            st.write(f"**{dtype}:** {count} columns")


# Keep backward compatibility - these functions are now handled by the modular system
# but maintained here to prevent import errors in existing code

def display_loaded_datasets():
    """Backwards compatibility - now handled by data_sources.dataset_manager"""
    from src.ui.data_sources import display_loaded_datasets as new_display_loaded_datasets
    return new_display_loaded_datasets()


def handle_csv_upload():
    """Backwards compatibility - now handled by data_sources.csv_handler"""
    from src.ui.data_sources import handle_csv_upload as new_handle_csv_upload
    return new_handle_csv_upload()


def handle_excel_upload():
    """Backwards compatibility - now handled by data_sources.excel_handler"""
    from src.ui.data_sources import handle_excel_upload as new_handle_excel_upload
    return new_handle_excel_upload()


def handle_snowflake_connection():
    """Backwards compatibility - now handled by data_sources.snowflake_handler"""
    from src.ui.data_sources import handle_snowflake_connection as new_handle_snowflake_connection
    return new_handle_snowflake_connection()


# Note: The original data_source.py file was over 1300 lines and has been 
# successfully modularized into the following structure:
#
# src/ui/data_sources/
# ├── __init__.py              # Package exports
# ├── main.py                  # Main coordination and interface
# ├── dataset_manager.py       # Dataset display and management
# ├── csv_handler.py           # CSV file upload and processing
# ├── excel_handler.py         # Excel file upload and processing  
# ├── snowflake_handler.py     # Snowflake database connection
# └── sidebar_handlers.py      # Sidebar functionality
#
# This modular structure provides:
# - Better code organization and maintainability
# - Easier testing and debugging
# - Clear separation of concerns
# - Improved readability and navigation
# - Simplified future enhancements
