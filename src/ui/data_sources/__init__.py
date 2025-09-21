"""
Data Sources Package

This package contains modular handlers for different data sources
"""

from .main import render_data_sources_page, get_current_dataset_info, is_any_database_connected, get_connected_databases, validate_data_sources
from .dataset_manager import display_loaded_datasets, handle_dataset_selection, handle_dataset_removal
from .csv_handler import handle_csv_upload
from .excel_handler import handle_excel_upload
from .snowflake_handler import handle_snowflake_connection, handle_snowflake_data_loading
from .oracle_handler import handle_oracle_connection, handle_oracle_data_loading
from .hdfs_handler import handle_hdfs_connection, handle_hdfs_data_loading
from .sidebar_handlers import render_full_sidebar

__all__ = [
    # Main interface
    'render_data_sources_page',
    'get_current_dataset_info',
    'is_any_database_connected', 
    'get_connected_databases',
    'validate_data_sources',
    
    # Dataset management
    'display_loaded_datasets',
    'handle_dataset_selection', 
    'handle_dataset_removal',
    
    # File handlers
    'handle_csv_upload',
    'handle_excel_upload',
    
    # Database handlers
    'handle_snowflake_connection',
    'handle_snowflake_data_loading',
    'handle_oracle_connection',
    'handle_oracle_data_loading',
    'handle_hdfs_connection',
    'handle_hdfs_data_loading',
    
    # Sidebar
    'render_full_sidebar'
]
