"""
Data Source Tab - Handle various data source types (CSV, Excel)
"""

import streamlit as st
import pandas as pd
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.connectors.data_connectors import DataConnectorFactory
from src.ui.components import show_data_preview, display_loading_info, create_file_upload_section


def data_source_tab():
    """Data source selection and connection tab"""
    st.header("Data Sources")
    
    # Show loaded datasets first if any exist
    if st.session_state.get('datasets'):
        display_loaded_datasets()
        st.markdown("---")
    
    # Simplified data loading interface
    st.subheader("Load New Data")
    
    # Create tabs for different data source types
    csv_tab, excel_tab, snowflake_tab = st.tabs(["CSV Files", "Excel Files", "Snowflake Database"])
    
    with csv_tab:
        handle_csv_upload()
    
    with excel_tab:
        handle_excel_upload()
    
    with snowflake_tab:
        handle_snowflake_connection()
    
    # Show data loading interface if Snowflake is connected
    if st.session_state.get('snowflake_connected', False):
        st.markdown("---")
        handle_snowflake_data_loading()


def handle_csv_upload():
    """Simplified CSV file upload interface"""
    
    # File upload - supports both single and multiple files
    uploaded_files = st.file_uploader(
        "Select CSV files to upload", 
        type=['csv'], 
        accept_multiple_files=True,
        help="Choose one or more CSV files"
    )
    
    if uploaded_files:
        # Show selected files
        if len(uploaded_files) == 1:
            st.info(f"Selected: {uploaded_files[0].name}")
        else:
            st.info(f"Selected {len(uploaded_files)} files")
            with st.expander("View selected files"):
                for file in uploaded_files:
                    st.write(f"• {file.name}")
        
        # CSV configuration
        st.write("**Configuration:**")
        col1, col2 = st.columns(2)
        
        with col1:
            encoding = st.selectbox("File encoding", ["utf-8", "latin-1", "cp1252"], index=0)
            delimiter = st.selectbox("Delimiter", [",", ";", "\t", "|"], index=0)
        
        with col2:
            load_full = st.checkbox("Load complete dataset", value=True)
            if not load_full:
                sample_rows = st.number_input("Number of rows to load", min_value=100, max_value=1000000, value=10000)
            else:
                sample_rows = None
        
        # Load button
        if st.button("Load Files", type="primary"):
            load_csv_files(uploaded_files, encoding, delimiter, sample_rows)


def load_csv_files(uploaded_files, encoding, delimiter, sample_rows):
    """Load CSV files and store in session state"""
    successful_loads = 0
    failed_loads = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Loading {uploaded_file.name}...")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
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
                    
                    # Set first file as current dataset
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
        st.success(f"Successfully loaded {successful_loads} out of {len(uploaded_files)} files")
        
        # Show summary of loaded datasets
        st.write("**Loaded datasets:**")
        for dataset_name, df in st.session_state.datasets.items():
            # Handle both DataFrame and legacy dictionary formats
            if isinstance(df, dict) and 'dataframe' in df:
                actual_df = df['dataframe']
                st.write(f"• {dataset_name}: {len(actual_df):,} rows, {len(actual_df.columns)} columns")
            else:
                st.write(f"• {dataset_name}: {len(df):,} rows, {len(df.columns)} columns")
    
    if failed_loads:
        st.error("Failed to load some files:")
        for error in failed_loads:
            st.write(f"• {error}")


def handle_single_csv_upload():
    """Handle single CSV file upload - DEPRECATED"""
    pass


def handle_multiple_csv_upload():
    """Handle multiple CSV file uploads - DEPRECATED"""
    pass


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
            sample_rows = st.number_input("Sample Rows", min_value=100, max_value=1000000, value=10000, key="multi_csv_sample_rows")
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
            # Handle both DataFrame and legacy dictionary formats
            if isinstance(df, dict) and 'dataframe' in df:
                # Legacy format with metadata
                actual_df = df['dataframe']
                st.write(f"• **{dataset_name}**: {len(actual_df):,} rows, {len(actual_df.columns)} columns")
            else:
                # Direct DataFrame format
                st.write(f"• **{dataset_name}**: {len(df):,} rows, {len(df.columns)} columns")
    
    if failed_loads:
        st.error("❌ Failed to load some files:")
        for error in failed_loads:
            st.write(f"• {error}")


def display_loaded_datasets():
    """Display loaded datasets and allow selection"""
    if st.session_state.datasets:
        st.subheader("Loaded Datasets")
        
        # Show summary of all datasets
        with st.expander("View all datasets", expanded=False):
            for dataset_name, dataset_info in st.session_state.datasets.items():
                # Get DataFrame and determine source type
                if isinstance(dataset_info, dict) and 'dataframe' in dataset_info:
                    df = dataset_info['dataframe']
                    source_type = dataset_info.get('source_type', 'unknown')
                else:
                    df = dataset_info
                    if dataset_name.startswith('snowflake_'):
                        source_type = 'snowflake'
                    elif dataset_name.endswith('.csv'):
                        source_type = 'csv'
                    elif dataset_name.endswith(('.xlsx', '.xls')):
                        source_type = 'excel'
                    else:
                        source_type = 'file'
                
                st.write(f"**{dataset_name}** ({source_type.upper()}) - {len(df):,} rows × {len(df.columns)} columns")
        
        # Dataset selection
        dataset_names = list(st.session_state.datasets.keys())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_dataset = st.selectbox(
                "Select dataset for analysis", 
                dataset_names,
                index=dataset_names.index(st.session_state.current_dataset) if st.session_state.current_dataset in dataset_names else 0,
                key="dataset_selector"
            )
            
            # Update current dataset and backward compatibility variables
            if selected_dataset != st.session_state.current_dataset:
                st.session_state.current_dataset = selected_dataset
                dataset_info = st.session_state.datasets[selected_dataset]
                
                # Handle different dataset types
                if isinstance(dataset_info, dict) and 'dataframe' in dataset_info:
                    # New format with metadata
                    st.session_state.data = dataset_info['dataframe']
                    # Only set connector if it exists (not needed for Snowflake)
                    if selected_dataset in st.session_state.get('connectors', {}):
                        st.session_state.connector = st.session_state.connectors[selected_dataset]
                    else:
                        st.session_state.connector = None
                else:
                    # Legacy format - direct DataFrame
                    st.session_state.data = dataset_info
                    st.session_state.connector = st.session_state.connectors.get(selected_dataset)
                st.rerun()
        
        with col2:
            if st.button("Remove Dataset", key="remove_dataset_button"):
                if len(st.session_state.datasets) > 1:
                    # Remove the selected dataset
                    del st.session_state.datasets[selected_dataset]
                    if selected_dataset in st.session_state.get('connectors', {}):
                        del st.session_state.connectors[selected_dataset]
                    
                    # Update current dataset to the first remaining one
                    remaining_datasets = list(st.session_state.datasets.keys())
                    if remaining_datasets:
                        st.session_state.current_dataset = remaining_datasets[0]
                        dataset_info = st.session_state.datasets[remaining_datasets[0]]
                        
                        # Handle different dataset types
                        if isinstance(dataset_info, dict) and 'dataframe' in dataset_info:
                            st.session_state.data = dataset_info['dataframe']
                        else:
                            st.session_state.data = dataset_info
                        
                        st.session_state.connector = st.session_state.get('connectors', {}).get(remaining_datasets[0])
                    else:
                        st.session_state.current_dataset = None
                        st.session_state.data = None
                        st.session_state.connector = None
                    
                    st.success(f"Removed dataset: {selected_dataset}")
                    st.rerun()
                else:
                    st.warning("Cannot remove the last remaining dataset")
        
        # Display current dataset info
        if st.session_state.current_dataset:
            dataset_info = st.session_state.datasets[st.session_state.current_dataset]
            
            # Get the DataFrame from either format
            if isinstance(dataset_info, dict) and 'dataframe' in dataset_info:
                current_df = dataset_info['dataframe']
                source_type = dataset_info.get('source_type', 'unknown')
                
                # Show additional info for Snowflake
                if source_type == 'snowflake':
                    st.write(f"**Currently selected**: {st.session_state.current_dataset} (Snowflake)")
                    if 'query' in dataset_info:
                        with st.expander("View SQL Query"):
                            st.code(dataset_info['query'], language='sql')
                else:
                    st.write(f"**Currently selected**: {st.session_state.current_dataset}")
            else:
                # Legacy format
                current_df = dataset_info
                st.write(f"**Currently selected**: {st.session_state.current_dataset}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(current_df):,}")
            with col2:
                st.metric("Columns", len(current_df.columns))
            with col3:
                st.metric("Memory Usage", f"{current_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Show data preview
            show_data_preview(current_df)
    else:
        st.info("No datasets loaded yet. Please upload files or connect to a database.")


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


def handle_snowflake_connection():
    """Handle Snowflake database connection and data loading"""
    st.subheader("🏔️ Snowflake Database Connection")
    
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
            with st.spinner("Connecting to Snowflake..."):
                try:
                    from src.connectors.data_connectors import DataConnectorFactory
                    
                    connector = DataConnectorFactory.create_connector(
                        'snowflake',
                        account=env_account,
                        username=env_username,
                        password=env_password,
                        warehouse=env_warehouse,
                        database=env_database,
                        schema=env_schema
                    )
                    
                    if connector.connect():
                        st.success(f"✅ Successfully connected to Snowflake database: {env_database}")
                        
                        # Store connection in session state
                        st.session_state.snowflake_connector = connector
                        st.session_state.snowflake_connected = True
                        
                        # Get and display available tables
                        tables = connector.get_tables()
                        if tables:
                            st.info(f"Found {len(tables)} tables in schema {env_schema}")
                        else:
                            st.warning("No tables found in the specified schema")
                    else:
                        st.error("❌ Failed to connect to Snowflake. Please check your credentials.")
                        
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")
    else:
        st.warning("⚠️ Snowflake environment variables not fully configured. Please use manual connection below.")
    
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
                    st.error("❌ Please fill in all required fields")
                    return
                
                # Test connection
                with st.spinner("Connecting to Snowflake..."):
                    try:
                        from src.connectors.data_connectors import DataConnectorFactory
                        
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
                            st.success("✅ Successfully connected to Snowflake!")
                            
                            # Store connection in session state
                            st.session_state.snowflake_connector = connector
                            st.session_state.snowflake_connected = True
                            
                            # Get and display available tables
                            tables = connector.get_tables()
                            if tables:
                                st.info(f"Found {len(tables)} tables in schema {schema}")
                            else:
                                st.warning("No tables found in the specified schema")
                        else:
                            st.error("❌ Failed to connect to Snowflake. Please check your credentials.")
                            
                    except Exception as e:
                        st.error(f"❌ Connection error: {str(e)}")


def handle_snowflake_data_loading():
    """Handle data loading from connected Snowflake instance"""
    st.subheader("📊 Load Data from Snowflake")
    
    connector = st.session_state.get('snowflake_connector')
    if not connector:
        st.error("❌ No active Snowflake connection")
        return
    
    # Get available tables
    tables = connector.get_tables()
    
    # Query input method selection
    query_method = st.radio(
        "Select Data Source Method:",
        ["Select from Tables", "Custom SQL Query"],
        help="Choose how you want to specify the data to load",
        key="snowflake_bhai"
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
                            st.write("**Columns:**")
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
        st.markdown("**Enter your SQL query:**")
        query = st.text_area(
            "SQL Query",
            height=150,
            placeholder="SELECT * FROM your_table WHERE condition LIMIT 1000",
            help="Write your SQL query. Be mindful of performance for large datasets."
        )
        
        # Row limit for custom queries
        limit_custom = st.checkbox("Add Row Limit", value=True, key="custom_limit")
        if limit_custom:
            row_limit = st.number_input("Max Rows", min_value=100, max_value=1000000, 
                                      value=10000, step=1000, key="custom_limit_value")
            if query and "LIMIT" not in query.upper():
                query = f"{query.rstrip(';')} LIMIT {row_limit}"
    
    # Load data button
    if st.button("Load Data from Snowflake", type="primary", disabled=not query):
        if not query.strip():
            st.error("❌ Please enter a SQL query")
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
                st.warning("⚠️ Query returned no data")
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
            
            # Store DataFrame directly (consistent with CSV/Excel handling)
            st.session_state.datasets[dataset_name] = df
            st.session_state.connectors[dataset_name] = st.session_state.snowflake_connector
            
            # Set as current dataset
            st.session_state.data = df
            st.session_state.connector = st.session_state.snowflake_connector
            st.session_state.current_dataset = dataset_name
            
            # Success message
            st.success(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns from Snowflake")
            
            # Show preview
            show_data_preview(df)
            
    except Exception as e:
        st.error(f"Error loading data from Snowflake: {str(e)}")
        if "syntax" in str(e).lower():
            st.info("Tip: Check your SQL syntax and table/column names")
        elif "permission" in str(e).lower():
            st.info("Tip: Check your user permissions for the requested data")


# Sidebar functions for streamlined interface
def handle_sidebar_csv_upload():
    """Handle CSV file upload in sidebar"""
    uploaded_files = st.file_uploader(
        "Upload CSV files:", 
        type=['csv'], 
        accept_multiple_files=True,
        help="Select one or more CSV files",
        key="sidebar_csv"
    )
    
    if uploaded_files:
        # Configuration in sidebar
        encoding = st.selectbox("Encoding:", ["utf-8", "latin-1", "cp1252"], index=0, key="sidebar_csv_encoding")
        delimiter = st.selectbox("Delimiter:", [",", ";", "\t", "|"], index=0, key="sidebar_csv_delimiter")
        
        load_full = st.checkbox("Load complete files", value=True, key="sidebar_csv_full")
        if not load_full:
            sample_rows = st.number_input("Rows to load:", min_value=100, max_value=1000000, value=10000, key="sidebar_csv_rows")
        else:
            sample_rows = None
        
        if st.button("Load CSV Files", type="primary", key="sidebar_csv_load"):
            load_csv_files(uploaded_files, encoding, delimiter, sample_rows)


def handle_sidebar_excel_upload():
    """Handle Excel file upload in sidebar"""
    uploaded_file = st.file_uploader(
        "Upload Excel file:", 
        type=['xlsx', 'xls'],
        help="Select an Excel file",
        key="sidebar_excel"
    )
    
    if uploaded_file is not None:
        try:
            import pandas as pd
            # Get sheet names
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            # Sheet selection
            selected_sheet = st.selectbox("Select sheet:", sheet_names, key="sidebar_excel_sheet")
            
            # Configuration
            load_full = st.checkbox("Load complete sheet", value=True, key="sidebar_excel_full")
            if not load_full:
                sample_rows = st.number_input("Rows to load:", min_value=100, max_value=1000000, value=10000, key="sidebar_excel_rows")
            else:
                sample_rows = None
            
            if st.button("Load Excel File", type="primary", key="sidebar_excel_load"):
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
                            
                            st.success(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
                        else:
                            st.error("Failed to connect to Excel file")
                            
                except Exception as e:
                    st.error(f"Error loading Excel: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")


def handle_sidebar_snowflake_connection():
    """Handle Snowflake connection in sidebar with persistent state"""
    
    # Check if already connected
    if st.session_state.get('snowflake_connected', False):
        # Show connection status and controls
        connector = st.session_state.snowflake_connector
        st.success("Connected to Snowflake")
        
        # Connection info
        with st.expander("Connection Details", expanded=False):
            st.write(f"**Database:** {connector.database}")
            st.write(f"**Schema:** {connector.schema}")
            st.write(f"**Warehouse:** {connector.warehouse}")
        
        # Disconnect option
        if st.button("Disconnect", key="sidebar_snowflake_disconnect"):
            st.session_state.snowflake_connected = False
            st.session_state.snowflake_connector = None
            st.rerun()
        
        st.markdown("---")
        st.subheader("Load Data")
        
        # Data loading interface
        try:
            tables = connector.get_tables()
            
            if tables:
                # Table selection and query options
                query_method = st.radio(
                    "Data source:",
                    ["Select Table", "Custom Query"],
                    key="sidebar_sf_method"
                )
                
                if query_method == "Select Table":
                    selected_table = st.selectbox("Table:", tables, key="sidebar_sf_table")
                    
                    # Row limit option
                    limit_rows = st.checkbox("Limit rows", value=True, key="sidebar_sf_limit")
                    if limit_rows:
                        row_limit = st.number_input("Max rows:", min_value=100, max_value=1000000, value=10000, key="sidebar_sf_rows")
                        query = f'SELECT * FROM "{connector.schema}"."{selected_table}" LIMIT {row_limit}'
                    else:
                        query = f'SELECT * FROM "{connector.schema}"."{selected_table}"'
                        row_limit = None
                    
                    # Show table info
                    try:
                        table_info = connector.get_table_info(selected_table)
                        if table_info:
                            st.caption(f"Columns: {table_info.get('column_count', 'Unknown')}")
                    except:
                        pass
                    
                    if st.button("Load Table", type="primary", key="sidebar_sf_load_table"):
                        load_snowflake_data(connector, query, selected_table)
                
                else:  # Custom Query
                    custom_query = st.text_area(
                        "SQL Query:",
                        height=100,
                        placeholder="SELECT * FROM your_table WHERE condition",
                        key="sidebar_sf_custom_query"
                    )
                    
                    # Optional row limit for custom queries
                    limit_custom = st.checkbox("Add row limit", value=True, key="sidebar_sf_custom_limit")
                    if limit_custom:
                        custom_limit = st.number_input("Max rows:", min_value=100, max_value=1000000, value=10000, key="sidebar_sf_custom_rows")
                        if custom_query and "LIMIT" not in custom_query.upper():
                            query = f"{custom_query.rstrip(';')} LIMIT {custom_limit}"
                        else:
                            query = custom_query
                    else:
                        query = custom_query
                    
                    if st.button("Execute Query", type="primary", key="sidebar_sf_load_custom", disabled=not custom_query.strip()):
                        load_snowflake_data(connector, query, None)
            else:
                st.warning("No tables found in the current schema")
                
        except Exception as e:
            st.error(f"Error accessing tables: {str(e)}")
            # Offer to reconnect
            if st.button("Reconnect", key="sidebar_sf_reconnect"):
                st.session_state.snowflake_connected = False
                st.rerun()
    
    else:
        # Not connected - show connection interface
        # Load environment variables
        env_account = os.getenv('SNOWFLAKE_ACCOUNT', '')
        env_username = os.getenv('SNOWFLAKE_USERNAME', '')
        env_password = os.getenv('SNOWFLAKE_PASSWORD', '')
        env_warehouse = os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH')
        env_database = os.getenv('SNOWFLAKE_DATABASE', '')
        env_schema = os.getenv('SNOWFLAKE_SCHEMA', 'PUBLIC')
        
        # Check if environment is configured
        env_configured = all([env_account, env_username, env_password, env_database])
        
        if env_configured:
            st.info("Environment configuration detected")
            st.write(f"**Database:** {env_database}")
            st.write(f"**Schema:** {env_schema}")
            
            if st.button("Connect to Snowflake", type="primary", key="sidebar_snowflake_connect"):
                with st.spinner("Connecting to Snowflake..."):
                    try:
                        connector = DataConnectorFactory.create_connector(
                            'snowflake',
                            account=env_account,
                            username=env_username,
                            password=env_password,
                            warehouse=env_warehouse,
                            database=env_database,
                            schema=env_schema
                        )
                        
                        if connector.connect():
                            st.session_state.snowflake_connector = connector
                            st.session_state.snowflake_connected = True
                            st.success("Connected successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to connect to Snowflake")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
        else:
            st.warning("Snowflake not configured in .env file")
            
        # Manual configuration option
        with st.expander("Manual Configuration", expanded=not env_configured):
            account = st.text_input("Account:", value=env_account, key="sidebar_sf_account")
            username = st.text_input("Username:", value=env_username, key="sidebar_sf_username")
            password = st.text_input("Password:", type="password", value=env_password, key="sidebar_sf_password")
            warehouse = st.text_input("Warehouse:", value=env_warehouse, key="sidebar_sf_warehouse")
            database = st.text_input("Database:", value=env_database, key="sidebar_sf_database")
            schema = st.text_input("Schema:", value=env_schema, key="sidebar_sf_schema")
            
            if st.button("Connect", type="primary", key="sidebar_snowflake_manual_connect"):
                if all([account, username, password, database]):
                    with st.spinner("Connecting..."):
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
                                st.session_state.snowflake_connector = connector
                                st.session_state.snowflake_connected = True
                                st.success("Connected successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to connect")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.error("Please fill all required fields")


def data_overview_tab():
    """Main data overview tab - replaces the old data source tab"""
    if st.session_state.get('datasets'):
        display_loaded_datasets()
    else:
        st.info("No data loaded yet. Use the sidebar to upload files or connect to databases.")
        
        # Quick start guide
        st.subheader("Quick Start Guide")
        st.write("1. **Upload Data**: Use the sidebar to upload CSV/Excel files or connect to Snowflake")
        st.write("2. **Explore**: View your data structure and basic statistics") 
        st.write("3. **Profile**: Generate comprehensive data quality reports")
        st.write("4. **Analyze**: Detect anomalies and data quality issues")
        st.write("5. **Recommend**: Get AI-powered data quality recommendations")
