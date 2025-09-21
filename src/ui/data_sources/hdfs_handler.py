"""
HDFS Handler Module

Handles HDFS (Hadoop Distributed File System) connection and data loading
"""

import streamlit as st
import pandas as pd
import os
import logging
from datetime import datetime
from ...connectors.data_connectors import DataConnectorFactory
from ..components import show_data_preview

logger = logging.getLogger(__name__)


def handle_hdfs_connection():
    """Handle HDFS connection and data loading"""
    st.subheader("HDFS Connection")
    
    # Load environment variables for HDFS
    env_namenode_url = os.getenv('HDFS_NAMENODE_URL', '')
    env_username = os.getenv('HDFS_USERNAME', 'hdfs')
    env_timeout = os.getenv('HDFS_TIMEOUT', '30')
    
    # Check if required environment variables are set
    env_configured = bool(env_namenode_url)
    
    if env_configured:
        st.info("Using HDFS configuration from environment variables")
        
        # Auto-connect button
        if st.button("Connect with Environment Settings", type="primary"):
            connect_to_hdfs(
                env_namenode_url,
                env_username,
                int(env_timeout)
            )
    else:
        st.warning("HDFS environment variables not configured. Please use manual connection below.")
    
    # Manual connection form (always available as fallback)
    with st.expander("Manual Connection (Override Environment Settings)", expanded=not env_configured):
        with st.form("hdfs_connection"):
            st.markdown("### HDFS Connection Details")
            
            # HDFS configuration with examples
            st.markdown("**HDFS NameNode URL Examples:**")
            st.code("""
# Standard HDFS:
http://namenode-host:9870
# or
http://namenode-host:50070

# Secure HDFS (HTTPS):
https://namenode-host:9871

# High Availability HDFS:
http://hdfs-cluster:8020
            """)
            
            namenode_url = st.text_input(
                "NameNode URL",
                value=env_namenode_url,
                help="HDFS NameNode web interface URL (e.g., http://namenode:9870)"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                username = st.text_input(
                    "Username", 
                    value=env_username,
                    help="HDFS username (default: hdfs)"
                )
            with col2:
                timeout = st.number_input(
                    "Timeout (seconds)",
                    value=int(env_timeout) if env_timeout else 30,
                    min_value=5,
                    max_value=300,
                    help="Connection timeout in seconds"
                )
            
            # Kerberos authentication (optional)
            st.markdown("### Kerberos Authentication (Optional)")
            st.caption("For secure Hadoop clusters with Kerberos authentication")
            
            col3, col4 = st.columns(2)
            with col3:
                kerberos_principal = st.text_input(
                    "Kerberos Principal",
                    help="e.g., user@REALM.COM (leave empty for simple auth)"
                )
            with col4:
                kerberos_keytab = st.text_input(
                    "Keytab File Path",
                    help="Path to keytab file (leave empty for simple auth)"
                )
            
            # Connection test button
            connect_button = st.form_submit_button("Connect to HDFS", type="primary")
            
            if connect_button:
                if not namenode_url:
                    st.error("Please provide the NameNode URL")
                    return
                
                connect_to_hdfs(
                    namenode_url,
                    username,
                    timeout,
                    kerberos_principal if kerberos_principal else None,
                    kerberos_keytab if kerberos_keytab else None
                )


def connect_to_hdfs(namenode_url, username, timeout, kerberos_principal=None, kerberos_keytab=None):
    """Connect to HDFS with provided credentials"""
    with st.spinner("Connecting to HDFS..."):
        try:
            connector = DataConnectorFactory.create_connector(
                'hdfs',
                namenode_url=namenode_url,
                username=username,
                timeout=timeout,
                kerberos_principal=kerberos_principal,
                kerberos_keytab=kerberos_keytab
            )
            
            if connector.connect():
                st.success("Successfully connected to HDFS!")
                
                # Store connection in session state
                st.session_state.hdfs_connector = connector
                st.session_state.hdfs_connected = True
                
                # Get and display available directories
                directories = connector.get_directories('/')
                files = connector.get_files('/')
                
                if directories or files:
                    st.info(f"Found {len(directories)} directories and {len(files)} files in HDFS root")
                else:
                    st.warning("No files or directories found in HDFS root")
                
                # Rerun to immediately show the data loading interface
                st.rerun()
            else:
                st.error("Failed to connect to HDFS. Please check your connection settings.")
                
        except ImportError:
            st.error("""
            **HDFS connector not available!**
            
            To connect to HDFS, you need to install the HDFS client library:
            
            ```bash
            pip install hdfs3
            # or
            pip install hdfs
            ```
            """)
        except Exception as e:
            error_msg = str(e).lower()
            
            if "connection refused" in error_msg or "connection failed" in error_msg:
                st.error("**Connection refused:** Please check if HDFS NameNode is running and accessible.")
            elif "timeout" in error_msg:
                st.error("**Connection timeout:** Please check network connectivity and increase timeout if needed.")
            elif "authentication" in error_msg or "unauthorized" in error_msg:
                st.error("**Authentication failed:** Please check your username or Kerberos credentials.")
            else:
                st.error(f"**Connection failed:** {str(e)}")
                
            st.info("""
            **Troubleshooting Tips for HDFS:**
            1. Ensure HDFS NameNode is running and accessible
            2. Verify the NameNode URL (usually port 9870 for Hadoop 3.x or 50070 for Hadoop 2.x)
            3. Check network connectivity and firewall settings
            4. For secure clusters, ensure Kerberos credentials are correct
            5. Verify HDFS username has appropriate permissions
            """)
                
            logger.error(f"HDFS connection error: {str(e)}")


def handle_hdfs_data_loading():
    """Handle data loading from HDFS"""
    if not st.session_state.get('hdfs_connected', False):
        st.warning("Please connect to HDFS first")
        return
    
    connector = st.session_state.get('hdfs_connector')
    if not connector:
        st.error("HDFS connector not found in session state")
        return
    
    st.subheader("Load Data from HDFS")
    
    # File browser approach
    st.markdown("### Browse HDFS Files")
    
    # Directory navigation
    if 'hdfs_current_path' not in st.session_state:
        st.session_state.hdfs_current_path = '/'
    
    current_path = st.session_state.hdfs_current_path
    
    # Show current path
    st.text(f"Current path: {current_path}")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📁 Root", help="Go to root directory"):
            st.session_state.hdfs_current_path = '/'
            st.rerun()
    
    with col2:
        if current_path != '/' and st.button("⬆️ Parent", help="Go to parent directory"):
            parent_path = '/'.join(current_path.rstrip('/').split('/')[:-1])
            st.session_state.hdfs_current_path = parent_path if parent_path else '/'
            st.rerun()
    
    # Get directories and files in current path
    directories = connector.get_directories(current_path)
    files = connector.get_files(current_path)
    
    # Filter files to show only data files
    data_files = [f for f in files if f.lower().endswith(('.csv', '.parquet', '.pq', '.json', '.xlsx', '.xls'))]
    
    # Show directories
    if directories:
        st.markdown("**Directories:**")
        for directory in directories[:20]:  # Limit display
            dir_name = directory.split('/')[-1]
            if st.button(f"📁 {dir_name}", key=f"dir_{directory}"):
                st.session_state.hdfs_current_path = directory
                st.rerun()
    
    # Show data files
    if data_files:
        st.markdown("**Data Files:**")
        selected_file = st.selectbox(
            "Select a file to load:",
            data_files,
            help="Choose a data file to load into the analysis tool"
        )
        
        if selected_file:
            # Show file info
            file_info = connector.get_file_info(selected_file)
            if file_info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("File Size", f"{file_info.get('size_mb', 'Unknown')} MB")
                with col2:
                    st.metric("File Type", selected_file.split('.')[-1].upper())
                with col3:
                    st.metric("Owner", file_info.get('owner', 'Unknown'))
            
            # Sample size selection
            sample_size = st.slider(
                "Number of rows to load:",
                min_value=100,
                max_value=100000,
                value=10000,
                step=100,
                help="Choose how many rows to load (limited for performance)"
            )
            
            # Load data button
            if st.button("Load File Data", type="primary"):
                load_hdfs_file(connector, selected_file, sample_size)
    else:
        if not directories:
            st.info("No data files or directories found in current path")
        else:
            st.info("No data files found in current path. Navigate to a directory containing CSV, Parquet, JSON, or Excel files.")
    
    # Manual file path option
    st.markdown("---")
    st.markdown("### Direct File Path")
    manual_path = st.text_input(
        "Enter HDFS file path directly:",
        placeholder="/path/to/your/data.csv",
        help="Enter the full HDFS path to your data file"
    )
    
    if manual_path and st.button("Load File by Path", type="secondary"):
        load_hdfs_file(connector, manual_path, 10000)


def load_hdfs_file(connector, file_path, limit):
    """Load data from HDFS file"""
    try:
        with st.spinner(f"Loading {limit:,} rows from {file_path}..."):
            df = connector.get_data(file_path, limit)
            
            if df is not None and not df.empty:
                # Store in session state
                file_name = file_path.split('/')[-1]
                dataset_name = f"hdfs_{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                if 'datasets' not in st.session_state:
                    st.session_state.datasets = {}
                
                st.session_state.datasets[dataset_name] = df
                st.session_state.data = df
                st.session_state.data_source = 'hdfs'
                st.session_state.current_dataset = dataset_name
                
                st.success(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns from {file_name}")
                
                # Show preview
                show_data_preview(df, f"HDFS File: {file_name}")
                
            else:
                st.warning("No data returned from the file")
                
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        logger.error(f"HDFS data loading error: {str(e)}")
