"""
Data Source Tab - Handle various data source types (CSV, Excel, Snowflake)

This file provides backwards compatibility for the original data source handlers.
The actual implementations have been moved to the data_sources package for better modularity.
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
    """Data overview tab - shows all loaded datasets and allows selection"""
    import pandas as pd
    
    # Check if any datasets are loaded
    if 'datasets' not in st.session_state or not st.session_state.datasets:
        st.info("No data loaded. Please load data from the Data Sources tab first.")
        return
    
    # Show summary of all loaded datasets
    st.subheader("Loaded Datasets Summary")
    
    # Use caching for expensive summary calculations
    @st.cache_data
    def create_datasets_summary(datasets_hash):
        """Create summary table with caching to avoid recalculation"""
        summary_data = []
        datasets = st.session_state.datasets
        
        for dataset_name, dataset_info in datasets.items():
            # Handle both new 'data' key and legacy 'dataframe' key
            if isinstance(dataset_info, dict):
                if 'data' in dataset_info:
                    df = dataset_info['data']
                elif 'dataframe' in dataset_info:
                    df = dataset_info['dataframe']
                else:
                    continue  # Skip invalid entries
                source_type = dataset_info.get('source_type', 'unknown')
            else:
                # Legacy format - dataframe stored directly
                df = dataset_info
                source_type = 'file'
            
            # Expensive calculations
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            missing_count = df.isnull().sum().sum()
            
            summary_data.append({
                'Dataset Name': dataset_name,
                'Source Type': source_type.upper(),
                'Rows': f"{len(df):,}",
                'Columns': len(df.columns),
                'Memory (MB)': f"{memory_usage:.1f}",
                'Missing Values': f"{missing_count:,}",
                'Missing %': f"{(missing_count / (len(df) * len(df.columns)) * 100):.1f}%"
            })
        
        return summary_data
    
    # Create a hash of the datasets to enable caching
    datasets_hash = str(hash(str(list(st.session_state.datasets.keys()))))
    summary_data = create_datasets_summary(datasets_hash)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width='stretch')
        
        # Use caching for total statistics to avoid recalculation
        @st.cache_data
        def calculate_total_stats(datasets_hash):
            """Calculate total statistics with caching"""
            datasets = st.session_state.datasets
            total_datasets = len(datasets)
            # Handle both key formats for total calculations
            total_rows = sum(
                len(info['data']) if isinstance(info, dict) and 'data' in info 
                else len(info['dataframe']) if isinstance(info, dict) and 'dataframe' in info 
                else len(info) 
                for info in datasets.values()
            )
            total_memory = sum(
                (info['data'] if isinstance(info, dict) and 'data' in info 
                 else info['dataframe'] if isinstance(info, dict) and 'dataframe' in info 
                 else info).memory_usage(deep=True).sum() / 1024**2
                for info in datasets.values()
            )
            return total_datasets, total_rows, total_memory
        
        total_datasets, total_rows, total_memory = calculate_total_stats(datasets_hash)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Datasets", total_datasets)
        with col2:
            st.metric("Total Rows", f"{total_rows:,}")
        with col3:
            st.metric("Total Memory", f"{total_memory:.1f} MB")
    
    # Dataset selection and detailed view
    st.markdown("---")
    st.subheader("Dataset Details")
    
    # Dataset selector
    dataset_names = list(st.session_state.datasets.keys())
    if dataset_names:
        selected_dataset = st.selectbox(
            "Select dataset to view details:",
            dataset_names,
            index=dataset_names.index(st.session_state.current_dataset) if st.session_state.current_dataset in dataset_names else 0,
            key="overview_dataset_selector"
        )
        
        # Update current dataset if selection changed
        if selected_dataset != st.session_state.current_dataset:
            st.session_state.current_dataset = selected_dataset
            # Update backward compatibility 'data' field
            dataset_info = st.session_state.datasets[selected_dataset]
            if isinstance(dataset_info, dict):
                if 'data' in dataset_info:
                    st.session_state.data = dataset_info['data']
                elif 'dataframe' in dataset_info:
                    st.session_state.data = dataset_info['dataframe']
            else:
                st.session_state.data = dataset_info
            # No need for st.rerun() here - Streamlit will handle the update automatically
        
        # Show detailed information for selected dataset
        dataset_info = st.session_state.datasets[selected_dataset]
        if isinstance(dataset_info, dict):
            if 'data' in dataset_info:
                df = dataset_info['data']
            elif 'dataframe' in dataset_info:
                df = dataset_info['dataframe']
            else:
                return  # Skip invalid entries
            source_type = dataset_info.get('source_type', 'unknown')
            
            # Show additional metadata if available
            if 'query' in dataset_info:
                with st.expander("Source Query"):
                    st.code(dataset_info['query'], language='sql')
        else:
            df = dataset_info
            source_type = 'file'
        
        st.write(f"Dataset: {selected_dataset} ({source_type.upper()})")
        
        # Detailed metrics for selected dataset
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")
        with col4:
            missing_count = df.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_count:,}")
        
        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(col_info, width='stretch')
        


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


def handle_sidebar_csv_upload():
    """Backwards compatibility - CSV upload functionality for sidebar"""
    from src.ui.data_sources import handle_csv_upload
    return handle_csv_upload()


def handle_sidebar_excel_upload():
    """Backwards compatibility - Excel upload functionality for sidebar"""
    from src.ui.data_sources import handle_excel_upload
    return handle_excel_upload()


def handle_sidebar_snowflake_connection():
    """Backwards compatibility - Snowflake connection functionality for sidebar"""
    from src.ui.data_sources import handle_snowflake_connection, handle_snowflake_data_loading
    
    # If not connected, show connection interface
    if not st.session_state.get('snowflake_connected', False):
        handle_snowflake_connection()
    else:
        # If connected, show data loading interface
        st.success("✅ Connected to Snowflake")
        handle_snowflake_data_loading()
        
        # Add disconnect option
        st.markdown("---")
        if st.button("Disconnect from Snowflake", key="sidebar_snowflake_disconnect"):
            st.session_state.snowflake_connected = False
            if 'snowflake_connector' in st.session_state:
                try:
                    st.session_state.snowflake_connector.disconnect()
                except:
                    pass
                del st.session_state.snowflake_connector
            st.rerun()



