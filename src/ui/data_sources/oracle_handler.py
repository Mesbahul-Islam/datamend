"""
Oracle Handler Module

Handles Oracle Cloud database connection and data loading
"""

import streamlit as st
import pandas as pd
import os
import logging
from datetime import datetime
from ...connectors.data_connectors import DataConnectorFactory
from ..components import show_data_preview

logger = logging.getLogger(__name__)


def handle_oracle_connection():
    """Handle Oracle database connection and data loading"""
    st.subheader("Oracle Cloud Database Connection")
    
    # Load environment variables for Oracle
    env_connection_string = os.getenv('ORACLE_CONNECTION_STRING', '')
    env_username = os.getenv('ORACLE_USERNAME', '')
    env_password = os.getenv('ORACLE_PASSWORD', '')
    env_wallet_location = os.getenv('ORACLE_WALLET_LOCATION', '')
    env_wallet_password = os.getenv('ORACLE_WALLET_PASSWORD', '')
    
    # Check if all required environment variables are set
    env_configured = all([env_connection_string, env_username, env_password])
    
    if env_configured:
        st.info("Using Oracle configuration from environment variables")
        
        # Auto-connect button
        if st.button("Connect with Environment Settings", type="primary"):
            connect_to_oracle(
                env_connection_string,
                env_username, 
                env_password,
                env_wallet_location if env_wallet_location else None,
                env_wallet_password if env_wallet_password else None
            )
    else:
        st.warning("Oracle environment variables not fully configured. Please use manual connection below.")
    
    # Manual connection form (always available as fallback)
    with st.expander("Manual Connection (Override Environment Settings)", expanded=not env_configured):
        with st.form("oracle_connection"):
            st.markdown("### Connection Details")
            
            # Connection string input with examples
            st.markdown("**Oracle Cloud Connection String Examples:**")
            st.code("""
# Oracle Cloud Autonomous Database (most common):
(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.eu-stockholm-1.oraclecloud.com))(connect_data=(service_name=gaaaae3991ccb74_testing_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))

# Standard Oracle Database:
host:port/service_name
# Example: mydb.us-east-1.rds.amazonaws.com:1521/ORCL

# TNS Name (if using tnsnames.ora):
MY_DB_TNS_NAME
            """)
            
            connection_string = st.text_area(
                "Connection String",
                value=env_connection_string,
                height=120,
                help="Enter Oracle connection string. For Oracle Cloud Autonomous DB, use the TNS connection string from your cloud console."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                username = st.text_input("Username", value=env_username)
            with col2:
                password = st.text_input("Password", type="password", value=env_password)
            
            # Oracle Cloud wallet configuration (optional)
            st.markdown("### Oracle Cloud Wallet (Optional)")
            st.caption("Only required if NOT using built-in security in connection string. Most Oracle Cloud Autonomous DB connections include security settings and don't need separate wallet files.")
            
            col3, col4 = st.columns(2)
            with col3:
                wallet_location = st.text_input("Wallet Directory", 
                                              value=env_wallet_location,
                                              help="Path to wallet directory (usually not needed for Oracle Cloud Autonomous DB)")
            with col4:
                wallet_password = st.text_input("Wallet Password", 
                                               type="password",
                                               value=env_wallet_password,
                                               help="Wallet password (usually not needed for Oracle Cloud Autonomous DB)")
            
            # Connection test button
            connect_button = st.form_submit_button("Connect to Oracle", type="primary")
            
            if connect_button:
                if not all([connection_string, username, password]):
                    st.error("Please fill in connection string, username, and password")
                    return
                
                connect_to_oracle(
                    connection_string,
                    username, 
                    password,
                    wallet_location if wallet_location else None,
                    wallet_password if wallet_password else None
                )


def connect_to_oracle(connection_string, username, password, wallet_location=None, wallet_password=None):
    """Connect to Oracle with provided credentials"""
    with st.spinner("Connecting to Oracle Database..."):
        try:
            connector = DataConnectorFactory.create_connector(
                'oracle',
                connection_string=connection_string,
                username=username,
                password=password,
                wallet_location=wallet_location,
                wallet_password=wallet_password
            )
            
            if connector.connect():
                st.success("Successfully connected to Oracle Database!")
                
                # Store connection in session state
                st.session_state.oracle_connector = connector
                st.session_state.oracle_connected = True
                
                # Get and display available tables
                tables = connector.get_tables()
                if tables:
                    st.info(f"Found {len(tables)} tables in your schema")
                else:
                    st.warning("No tables found in the current schema")
                
                # Rerun to immediately show the data loading interface
                st.rerun()
            else:
                st.error("Failed to connect to Oracle Database. Please check your credentials.")
                
        except ImportError:
            st.error("""
            **Oracle connector not available!**
            
            To connect to Oracle databases, you need to install the Oracle client library:
            
            ```bash
            pip install oracledb
            ```
            
            For Oracle Cloud Autonomous Database, no additional Oracle client installation is needed.
            """)
        except Exception as e:
            error_msg = str(e).lower()
            
            if "invalid username/password" in error_msg or "login denied" in error_msg:
                st.error("**Authentication failed:** Please check your username and password.")
            elif "network adapter" in error_msg or "connection refused" in error_msg:
                st.error("**Network connection failed:** Please check your connection string and network connectivity.")
            elif "service name" in error_msg or "sid" in error_msg:
                st.error("**Service not found:** Please verify your service name in the connection string.")
            else:
                st.error(f"**Connection failed:** {str(e)}")
                
            st.info("""
            **Troubleshooting Tips for Oracle Cloud:**
            1. Ensure your connection string is copied exactly from Oracle Cloud Console
            2. Verify your username and password are correct
            3. Check that your IP address is allowlisted in Oracle Cloud (if required)
            4. For Autonomous Database, ensure you're using the correct service level (high, medium, low)
            """)
                
            logger.error(f"Oracle connection error: {str(e)}")


def handle_oracle_data_loading():
    """Handle data loading from Oracle Database"""
    if not st.session_state.get('oracle_connected', False):
        st.warning("Please connect to Oracle first")
        return
    
    connector = st.session_state.get('oracle_connector')
    if not connector:
        st.error("Oracle connector not found in session state")
        return
    
    st.subheader("Load Data from Oracle")
    
    # Get available tables
    tables = connector.get_tables()
    
    if not tables:
        st.warning("No tables found in your Oracle schema")
        return
    
    # Data loading options
    loading_method = st.radio(
        "Choose loading method:",
        ["Select from Tables", "Custom SQL Query"],
        help="Choose how you want to load data"
    )
    
    if loading_method == "Select from Tables":
        # Table selection
        selected_table = st.selectbox(
            "Select a table:",
            tables,
            help="Choose a table to load data from"
        )
        
        if selected_table:
            # Show table info
            table_info = connector.get_table_info(selected_table)
            if table_info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{table_info.get('row_count', 'Unknown'):,}")
                with col2:
                    st.metric("Columns", table_info.get('column_count', 'Unknown'))
                with col3:
                    st.metric("Table", selected_table)
                
                # Sample size selection
                sample_size = st.slider(
                    "Number of rows to load:",
                    min_value=100,
                    max_value=min(50000, table_info.get('row_count', 50000)),
                    value=min(1000, table_info.get('row_count', 1000)),
                    step=100,
                    help="Choose how many rows to load (limited for performance)"
                )
                
                # Load data button
                if st.button("Load Table Data", type="primary"):
                    load_oracle_table(connector, selected_table, sample_size)
    
    else:  # Custom SQL Query
        st.markdown("### Custom SQL Query")
        st.caption("Write your own SQL query to load specific data")
        
        query = st.text_area(
            "SQL Query:",
            value="SELECT * FROM your_table_name WHERE ROWNUM <= 1000",
            height=100,
            help="Enter your SQL query. Use ROWNUM for limiting results in Oracle."
        )
        
        # Query execution button
        if st.button("Execute Query", type="primary"):
            if query.strip():
                load_oracle_query(connector, query)
            else:
                st.error("Please enter a SQL query")


def load_oracle_table(connector, table_name, limit):
    """Load data from Oracle table"""
    try:
        with st.spinner(f"Loading {limit:,} rows from {table_name}..."):
            query = f"SELECT * FROM {table_name} WHERE ROWNUM <= {limit}"
            df = connector.get_data(query)
            
            if df is not None and not df.empty:
                # Store in session state
                dataset_name = f"oracle_{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                if 'datasets' not in st.session_state:
                    st.session_state.datasets = {}
                
                st.session_state.datasets[dataset_name] = df
                st.session_state.data = df
                st.session_state.data_source = 'oracle'
                st.session_state.current_dataset = dataset_name
                
                st.success(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns from {table_name}")
                
                # Show preview
                show_data_preview(df, f"Oracle Table: {table_name}")
                
            else:
                st.warning("No data returned from the table")
                
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        logger.error(f"Oracle data loading error: {str(e)}")


def load_oracle_query(connector, query):
    """Load data using custom SQL query"""
    try:
        with st.spinner("Executing query..."):
            df = connector.get_data(query)
            
            if df is not None and not df.empty:
                # Store in session state
                dataset_name = f"oracle_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                if 'datasets' not in st.session_state:
                    st.session_state.datasets = {}
                
                st.session_state.datasets[dataset_name] = df
                st.session_state.data = df
                st.session_state.data_source = 'oracle'
                st.session_state.current_dataset = dataset_name
                
                st.success(f"Query executed successfully! Loaded {len(df):,} rows and {len(df.columns)} columns")
                
                # Show preview
                show_data_preview(df, "Oracle Query Result")
                
            else:
                st.warning("Query returned no data")
                
    except Exception as e:
        st.error(f"Query execution failed: {str(e)}")
        logger.error(f"Oracle query error: {str(e)}")
