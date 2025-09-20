"""
Snowflake Handler Module

Handles Snowflake database connection and data loading
"""

import streamlit as st
import pandas as pd
import os
import logging
from datetime import datetime
from ...connectors.data_connectors import DataConnectorFactory
from ..components import show_data_preview

logger = logging.getLogger(__name__)


def handle_snowflake_connection():
    """Handle Snowflake database connection and data loading"""
    st.subheader("Snowflake Database Connection")
    
    # Load environment variables for Snowflake
    env_account = os.getenv('SNOWFLAKE_ACCOUNT', '')
    env_username = os.getenv('SNOWFLAKE_USERNAME', '')
    env_password = os.getenv('SNOWFLAKE_PASSWORD', '')
    env_warehouse = os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH')
    env_database = os.getenv('SNOWFLAKE_DATABASE', '')
    env_schema = os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC')
    
    # Check if all required environment variables are set
    env_configured = all([env_account, env_username, env_password, env_database])
    
    if env_configured:
        st.info("Using Snowflake configuration from environment variables")
        
        # Auto-connect button
        if st.button("Connect with Environment Settings", type="primary"):
            connect_to_snowflake(env_account, env_username, env_password, env_warehouse, env_database, env_schema)
    else:
        st.warning("🟡 Snowflake environment variables not fully configured. Please use manual connection below.")
    
    # Manual connection form (always available as fallback)
    with st.expander("Manual Connection (Override Environment Settings)", expanded=not env_configured):
        with st.form("snowflake_connection"):
            st.markdown("### Connection Details")
            
            col1, col2 = st.columns(2)
            with col1:
                account = st.text_input("Account Identifier", 
                                       value=env_account,
                                       help="Your Snowflake account identifier (e.g., abc12345.us-east-1)")
                username = st.text_input("Username", value=env_username)
                warehouse = st.text_input("Warehouse", value=env_warehouse)
                
            with col2:
                password = st.text_input("Password", type="password", value=env_password)
                database = st.text_input("Database", value=env_database)
                schema = st.text_input("Schema", value=env_schema)
            
            # Connection test button
            connect_button = st.form_submit_button("Connect to Snowflake", type="primary")
            
            if connect_button:
                if not all([account, username, password, warehouse, database]):
                    st.error("🔴 Please fill in all required fields")
                    return
                
                connect_to_snowflake(account, username, password, warehouse, database, schema)


def connect_to_snowflake(account, username, password, warehouse, database, schema):
    """Connect to Snowflake with provided credentials"""
    with st.spinner("Connecting to Snowflake..."):
        try:
            connector = DataConnectorFactory.create_connector(
                'snowflake',
                account=account,
                username=username,
                password=password,
                warehouse=warehouse,
                database=database,
                schema=schema
            )
            
            if connector.connect():
                st.success("🟢 Successfully connected to Snowflake!")
                
                # Store connection in session state
                st.session_state.snowflake_connector = connector
                st.session_state.snowflake_connected = True
                
                # Get and display available tables
                tables = connector.get_tables()
                if tables:
                    st.info(f"Found {len(tables)} tables in schema {schema}")
                else:
                    st.warning("No tables found in the specified schema")
                
                # Rerun to immediately show the data loading interface
                st.rerun()
            else:
                st.error("🔴 Failed to connect to Snowflake. Please check your credentials.")
                
        except Exception as e:
            st.error(f"🔴 Connection error: {str(e)}")


def handle_snowflake_data_loading():
    """Handle data loading from connected Snowflake instance"""
    st.subheader("Load Data from Snowflake")
    
    connector = st.session_state.get('snowflake_connector')
    if not connector:
        st.error("🔴 No active Snowflake connection")
        return
    
    # Get available tables
    tables = connector.get_tables()
    
    # Query input method selection
    query_method = st.radio(
        "Select Data Source Method:",
        ["Select from Tables", "Custom SQL Query"],
        help="Choose how you want to specify the data to load",
        key="snowflake_query_method"
    )
    
    if query_method == "Select from Tables":
        if not tables:
            st.warning("No tables available in the current schema")
            return
            
        # Table selection
        selected_table = st.selectbox("Select Table", tables)
        
        if selected_table:
            # Show table info
            with st.expander("Table Information", expanded=False):
                try:
                    table_info = connector.get_table_info(selected_table)
                    if table_info:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Rows", table_info.get('row_count', 'Unknown'))
                            st.write("Columns:")
                            for col in table_info.get('columns', []):
                                col_type = table_info.get('column_types', {}).get(col, 'Unknown')
                                st.write(f"• {col} ({col_type})")
                        with col2:
                            st.metric("Columns", table_info.get('column_count', 'Unknown'))
                except Exception as e:
                    st.warning(f"Could not load table info: {str(e)}")
            
            # Query configuration
            col1, col2 = st.columns(2)
            with col1:
                limit_data = st.checkbox("Limit Rows", value=True, 
                                       help="Limit the number of rows for faster loading")
            with col2:
                if limit_data:
                    row_limit = st.number_input("Max Rows", min_value=100, max_value=1000000, 
                                              value=10000, step=1000)
                else:
                    row_limit = None
            
            # Generate query
            query = f'SELECT * FROM "{connector.schema}"."{selected_table}"'
            if limit_data and row_limit:
                query += f" LIMIT {row_limit}"
    
    else:  # Custom SQL Query
        st.markdown("Enter your SQL query:")
        query = st.text_area(
            "SQL Query",
            height=150,
            placeholder="SELECT * FROM your_table WHERE condition LIMIT 1000",
            help="Write your SQL query. Be mindful of performance for large datasets."
        )
        
        # Row limit for custom queries
        limit_custom = st.checkbox("Add Row Limit", value=True, key="snowflake_custom_limit")
        if limit_custom:
            row_limit = st.number_input("Max Rows", min_value=100, max_value=1000000, 
                                      value=10000, step=1000, key="snowflake_custom_limit_value")
            if query and "LIMIT" not in query.upper():
                query = f"{query.rstrip(';')} LIMIT {row_limit}"
    
    # Load data button
    if st.button("Load Data from Snowflake", type="primary", disabled=not query):
        if not query.strip():
            st.error("🔴 Please enter a SQL query")
            return
        
        # Pass table name if it's a table selection, otherwise None for custom query
        table_name = selected_table if query_method == "Select from Tables" and 'selected_table' in locals() else None
        load_snowflake_data(connector, query, table_name)


def load_snowflake_data(connector, query, table_name=None):
    """Load data from Snowflake using the provided query"""
    try:
        with st.spinner("Loading data from Snowflake..."):
            # Execute query
            df = connector.get_data(query=query)
            
            if df.empty:
                st.warning("🟡 Query returned no data")
                return
            
            # Create descriptive dataset name
            if table_name:
                # If it's a table query, use table name
                dataset_name = f"snowflake_{table_name}_{len(st.session_state.get('datasets', {})) + 1}"
            else:
                # If it's a custom query, use generic name
                dataset_name = f"snowflake_query_{len(st.session_state.get('datasets', {})) + 1}"
            
            if 'datasets' not in st.session_state:
                st.session_state.datasets = {}
            if 'connectors' not in st.session_state:
                st.session_state.connectors = {}
            
            # Store DataFrame with source type information
            st.session_state.datasets[dataset_name] = {
                'data': df,  # Use 'data' key for consistency
                'source_type': 'snowflake',
                'load_time': datetime.now(),
                'table_name': table_name if table_name else 'custom_query',
                'query': query,
                'row_count': len(df),
                'column_count': len(df.columns)
            }
            st.session_state.connectors[dataset_name] = st.session_state.snowflake_connector
            
            # Set as current dataset
            st.session_state.data = df
            st.session_state.connector = st.session_state.snowflake_connector
            st.session_state.current_dataset = dataset_name
            st.session_state.data_source = 'snowflake'
            
            # Store lineage-related information
            if table_name:
                st.session_state.current_table = table_name
                st.session_state.current_schema = connector.schema
                st.session_state.current_database = connector.database
                st.session_state.snowflake_config = {
                    'account': connector.account,
                    'username': connector.username,
                    'password': connector.password,
                    'warehouse': connector.warehouse,
                    'database': connector.database,
                    'schema': connector.schema
                }
            
            # Integrate with quality checking
            source_info = {
                'table_name': table_name if table_name else 'custom_query',
                'query': query,
                'load_time': str(datetime.now()),
                'row_count': len(df),
                'column_count': len(df.columns)
            }
            
            # Success message
            st.success(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns from Snowflake")
            
            # Note: Quality analysis is now available in the Data Profiling tab
            
            # Show preview
            show_data_preview(df)
            
    except Exception as e:
        st.error(f"Error loading data from Snowflake: {str(e)}")
        logger.error(f"Snowflake data loading failed: {str(e)}")
        if "syntax" in str(e).lower():
            st.info("Tip: Check your SQL syntax and table/column names")
        elif "permission" in str(e).lower():
            st.info("Tip: Check your user permissions for the requested data")
