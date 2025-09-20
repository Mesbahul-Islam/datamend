"""
Sidebar Handlers Module

Handles sidebar functionality for data source management
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional


def display_sidebar_datasets():
    """Display loaded datasets in the sidebar"""
    if 'datasets' in st.session_state and st.session_state.datasets:
        st.sidebar.subheader("📊 Loaded Datasets")
        
        for dataset_name, dataset_info in st.session_state.datasets.items():
            with st.sidebar.expander(f"📄 {dataset_name}", expanded=False):
                # Dataset information
                df = dataset_info['dataframe']
                st.write(f"**Rows:** {len(df)}")
                st.write(f"**Columns:** {len(df.columns)}")
                st.write(f"**Source:** {dataset_info.get('source_type', 'Unknown')}")
                
                if dataset_info.get('load_time'):
                    st.write(f"**Loaded:** {dataset_info['load_time'].strftime('%H:%M:%S')}")
                
                # Switch to this dataset
                if st.button(f"Use {dataset_name}", key=f"use_{dataset_name}"):
                    st.session_state.data = df
                    st.session_state.current_dataset = dataset_name
                    
                    # Set connector if available
                    if dataset_name in st.session_state.get('connectors', {}):
                        st.session_state.connector = st.session_state.connectors[dataset_name]
                    
                    # Streamlit will handle the update automatically
                
                # Remove dataset
                if st.button(f"Remove", key=f"remove_{dataset_name}", 
                           help=f"Remove {dataset_name} from memory"):
                    remove_dataset(dataset_name)
                    # Only rerun when actually removing something
    else:
        st.sidebar.info("No datasets loaded yet")


def display_sidebar_connections():
    """Display active database connections in the sidebar"""
    st.sidebar.subheader("🔗 Database Connections")
    
    connections = []
    
    # Check for active connections
    if st.session_state.get('snowflake_connected', False):
        connections.append(("Snowflake", "🏔️", "snowflake_connected"))
    
    if connections:
        for name, icon, key in connections:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(f"{icon} {name}")
            with col2:
                if st.button("❌", key=f"disconnect_{key}", help=f"Disconnect from {name}"):
                    disconnect_database(key)
                    # Only rerun for actual disconnection
    else:
        st.sidebar.info("No active database connections")


def display_sidebar_current_dataset():
    """Display information about the currently active dataset"""
    if 'data' in st.session_state and st.session_state.data is not None:
        st.sidebar.subheader("📋 Current Dataset")
        
        df = st.session_state.data
        current_name = st.session_state.get('current_dataset', 'Unnamed Dataset')
        
        st.sidebar.success(f"**Active:** {current_name}")
        st.sidebar.metric("Rows", len(df))
        st.sidebar.metric("Columns", len(df.columns))
        
        # Dataset actions
        if st.sidebar.button("Clear Current Dataset", help="Remove current dataset from active memory"):
            clear_current_dataset()
            st.rerun()
    else:
        st.sidebar.info("No active dataset")


def display_sidebar_quick_stats():
    """Display quick statistics about the current dataset"""
    if 'data' in st.session_state and st.session_state.data is not None:
        df = st.session_state.data
        
        with st.sidebar.expander("📈 Quick Stats", expanded=False):
            # Memory usage
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")
            
            # Data types
            st.write("**Column Types:**")
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"• {dtype}: {count}")
            
            # Missing values
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                st.metric("Missing Values", missing_count)
            else:
                st.write("✅ No missing values")


def display_sidebar_export_options():
    """Display export options in the sidebar"""
    if 'data' in st.session_state and st.session_state.data is not None:
        with st.sidebar.expander("💾 Export Options", expanded=False):
            df = st.session_state.data
            current_name = st.session_state.get('current_dataset', 'dataset')
            
            # CSV export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name=f"{current_name}.csv",
                mime="text/csv"
            )
            
            # Excel export
            try:
                import io
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_data,
                    file_name=f"{current_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except ImportError:
                st.info("Install openpyxl for Excel export")


def remove_dataset(dataset_name: str):
    """Remove a dataset from session state"""
    if 'datasets' in st.session_state and dataset_name in st.session_state.datasets:
        del st.session_state.datasets[dataset_name]
    
    if 'connectors' in st.session_state and dataset_name in st.session_state.connectors:
        del st.session_state.connectors[dataset_name]
    
    # If this was the current dataset, clear it
    if st.session_state.get('current_dataset') == dataset_name:
        clear_current_dataset()


def clear_current_dataset():
    """Clear the currently active dataset"""
    if 'data' in st.session_state:
        del st.session_state.data
    if 'current_dataset' in st.session_state:
        del st.session_state.current_dataset
    if 'connector' in st.session_state:
        del st.session_state.connector


def disconnect_database(connection_key: str):
    """Disconnect from a database"""
    if connection_key in st.session_state:
        st.session_state[connection_key] = False
    
    # Clean up connector references
    if connection_key == 'snowflake_connected':
        if 'snowflake_connector' in st.session_state:
            try:
                st.session_state.snowflake_connector.disconnect()
            except:
                pass
            del st.session_state.snowflake_connector


def display_sidebar_help():
    """Display help information in the sidebar"""
    with st.sidebar.expander("ℹ️ Help & Tips", expanded=False):
        st.markdown("""
        ### Quick Guide
        
        **1. Load Data:**
        - Use the Data Sources tab to load CSV, Excel, or database data
        - Multiple datasets can be loaded simultaneously
        
        **2. Analyze Data:**
        - Switch between datasets using the sidebar
        - Use various analysis tools in the main interface
        
        **3. Export Results:**
        - Download processed data as CSV or Excel
        - Save analysis results for later use
        
        **4. Database Connections:**
        - Connect to Snowflake
        - Use environment variables for secure credentials
        - Manage connections through the sidebar
        
        ### Keyboard Shortcuts
        - `Ctrl + R`: Refresh page
        - `Ctrl + S`: Save current state
        
        ### Performance Tips
        - Limit large datasets to avoid memory issues
        - Use database queries to filter data at source
        - Export large results rather than viewing all rows
        """)


def render_full_sidebar():
    """Render the complete sidebar with all components"""
    # Current dataset info
    display_sidebar_current_dataset()
    
    # All loaded datasets
    display_sidebar_datasets()
    
    # Database connections
    display_sidebar_connections()
    
    # Quick stats
    display_sidebar_quick_stats()
    
    # Export options
    display_sidebar_export_options()
    
    # Help section
    display_sidebar_help()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Data Analysis Dashboard*")
