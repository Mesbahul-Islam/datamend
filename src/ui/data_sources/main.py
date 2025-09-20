"""
Main Data Sources Module

Coordinates all data source handlers and provides the main interface
"""

import streamlit as st
from .dataset_manager import display_loaded_datasets, handle_dataset_selection, handle_dataset_removal
from .csv_handler import handle_csv_upload
from .excel_handler import handle_excel_upload
from .snowflake_handler import handle_snowflake_connection, handle_snowflake_data_loading
from .sidebar_handlers import render_full_sidebar


def render_data_sources_page():
    """Render the complete data sources page"""
    st.title("📊 Data Sources")
    st.markdown("Load and manage your data from various sources")
    
    # Render sidebar
    render_full_sidebar()
    
    # Main content tabs
    tabs = st.tabs([
        "📄 Files (CSV/Excel)", 
        "🏔️ Snowflake", 
        "📋 Loaded Datasets"
    ])
    
    # File upload tab (CSV/Excel)
    with tabs[0]:
        st.header("📁 File Upload")
        
        file_type = st.radio(
            "Select file type:",
            ["CSV Files", "Excel Files"],
            horizontal=True
        )
        
        if file_type == "CSV Files":
            handle_csv_upload()
        else:
            handle_excel_upload()
    
    # Snowflake tab
    with tabs[1]:
        if not st.session_state.get('snowflake_connected', False):
            handle_snowflake_connection()
        else:
            st.success("✅ Connected to Snowflake")
            
            # Data loading section
            handle_snowflake_data_loading()
            
            # Disconnect option
            st.markdown("---")
            if st.button("Disconnect from Snowflake"):
                st.session_state.snowflake_connected = False
                if 'snowflake_connector' in st.session_state:
                    try:
                        st.session_state.snowflake_connector.disconnect()
                    except:
                        pass
                    del st.session_state.snowflake_connector
                st.rerun()
    
    # Loaded datasets tab
    with tabs[2]:
        st.header("📋 Dataset Management")
        
        # Display all loaded datasets
        display_loaded_datasets()
        
        # Dataset selection and removal
        if 'datasets' in st.session_state and st.session_state.datasets:
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔄 Switch Dataset")
                handle_dataset_selection()
            
            with col2:
                st.subheader("🗑️ Remove Dataset")
                handle_dataset_removal()
        
        # Dataset statistics
        if 'datasets' in st.session_state and st.session_state.datasets:
            st.markdown("---")
            st.subheader("📊 Dataset Statistics")
            
            total_datasets = len(st.session_state.datasets)
            total_rows = sum(len(info['dataframe']) for info in st.session_state.datasets.values())
            total_memory = sum(
                info['dataframe'].memory_usage(deep=True).sum() 
                for info in st.session_state.datasets.values()
            ) / 1024**2  # Convert to MB
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Datasets", total_datasets)
            with col2:
                st.metric("Total Rows", f"{total_rows:,}")
            with col3:
                st.metric("Memory Usage", f"{total_memory:.1f} MB")


def get_current_dataset_info():
    """Get information about the currently active dataset"""
    if 'data' not in st.session_state or st.session_state.data is None:
        return None
    
    return {
        'dataframe': st.session_state.data,
        'name': st.session_state.get('current_dataset', 'Unnamed Dataset'),
        'rows': len(st.session_state.data),
        'columns': len(st.session_state.data.columns),
        'memory_mb': st.session_state.data.memory_usage(deep=True).sum() / 1024**2
    }


def is_any_database_connected():
    """Check if any database is currently connected"""
    return st.session_state.get('snowflake_connected', False)


def get_connected_databases():
    """Get list of currently connected databases"""
    connected = []
    
    if st.session_state.get('snowflake_connected', False):
        connected.append('Snowflake')
    
    return connected


def validate_data_sources():
    """Validate the current state of data sources"""
    issues = []
    
    # Check for datasets without proper metadata
    if 'datasets' in st.session_state:
        for name, info in st.session_state.datasets.items():
            if 'dataframe' not in info:
                issues.append(f"Dataset '{name}' missing dataframe")
            if 'source_type' not in info:
                issues.append(f"Dataset '{name}' missing source type")
    
    # Check for orphaned connectors
    if 'connectors' in st.session_state:
        for name in st.session_state.connectors:
            if name not in st.session_state.get('datasets', {}):
                issues.append(f"Orphaned connector for '{name}'")
    
    return issues


# Export main functions for use in other modules
__all__ = [
    'render_data_sources_page',
    'get_current_dataset_info', 
    'is_any_database_connected',
    'get_connected_databases',
    'validate_data_sources'
]
