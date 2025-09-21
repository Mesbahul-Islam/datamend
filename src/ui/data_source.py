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

    st.markdown("""
    <style>
    div[data-testid="stSelectbox"] > div > div > div {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    /* Style the selectbox label */
    div[data-testid="stSelectbox"] label {
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    
    /* Style the selected value in the selectbox */
    div[data-testid="stSelectbox"] > div > div {
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
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
            
            # Basic metrics
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            
            # Data Quality Metrics
            total_cells = len(df) * len(df.columns)
            missing_count = df.isnull().sum().sum()
            
            # Completeness Score (0-100)
            completeness_score = ((total_cells - missing_count) / total_cells * 100) if total_cells > 0 else 0
            
            # Consistency Score - check for duplicate rows
            duplicate_rows = df.duplicated().sum()
            consistency_score = ((len(df) - duplicate_rows) / len(df) * 100) if len(df) > 0 else 100
            
            # Validity Score - check for mixed data types in object columns
            validity_issues = 0
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    # Try converting to numeric to detect mixed types
                    pd.to_numeric(df[col], errors='raise')
                except (ValueError, TypeError):
                    # Check if column has mixed numeric/text values
                    sample_values = df[col].dropna().head(100)
                    numeric_count = 0
                    for val in sample_values:
                        try:
                            float(val)
                            numeric_count += 1
                        except (ValueError, TypeError):
                            pass
                    # If 10-90% are numeric, it's likely mixed types
                    if 0.1 < numeric_count / len(sample_values) < 0.9 and len(sample_values) > 10:
                        validity_issues += 1
            
            validity_score = max(0, 100 - (validity_issues / len(df.columns) * 100)) if len(df.columns) > 0 else 100
            
            # Overall Quality Score (weighted average)
            quality_score = (completeness_score * 0.4 + consistency_score * 0.3 + validity_score * 0.3)
            
            # Quality grade
            if quality_score >= 90:
                quality_grade = "A"
            elif quality_score >= 80:
                quality_grade = "B"
            elif quality_score >= 70:
                quality_grade = "C"
            elif quality_score >= 60:
                quality_grade = "D"
            else:
                quality_grade = "F"
            
            summary_data.append({
                'Dataset Name': dataset_name,
                'Source Type': source_type.upper(),
                'Rows': f"{len(df):,}",
                'Columns': len(df.columns),
                'Memory (MB)': f"{memory_usage:.1f}",
                'Completeness': f"{completeness_score:.1f}%",
                'Consistency': f"{consistency_score:.1f}%",
                'Validity': f"{validity_score:.1f}%",
                'Quality Score': f"{quality_score:.1f}",
                'Grade': quality_grade
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
        
    st.metric("Total Datasets loaded", total_datasets)
        
    
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
            # Calculate overall data quality score
            total_cells = len(df) * len(df.columns)
            missing_count = df.isnull().sum().sum()
            completeness = ((total_cells - missing_count) / total_cells * 100) if total_cells > 0 else 0
            
            duplicate_rows = df.duplicated().sum()
            consistency = ((len(df) - duplicate_rows) / len(df) * 100) if len(df) > 0 else 100
            
            quality_score = (completeness * 0.6 + consistency * 0.4)
            st.metric("Data Quality", f"{quality_score:.1f}%")
        
        # Data Quality Breakdown
        st.subheader("Data Quality Metrics")
        
        # Calculate detailed quality metrics
        total_cells = len(df) * len(df.columns)
        missing_count = df.isnull().sum().sum()
        completeness_score = ((total_cells - missing_count) / total_cells * 100) if total_cells > 0 else 0
        
        duplicate_count = df.duplicated().sum()
        consistency_score = ((len(df) - duplicate_count) / len(df) * 100) if len(df) > 0 else 100
        
        # Check for empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        
        # Check for constant columns
        constant_columns = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_columns.append(col)
        
        # Data quality metrics display
        quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
        
        with quality_col1:
            st.metric(
                "Completeness", 
                f"{completeness_score:.1f}%",
                help="Percentage of non-missing values across all cells"
            )
        
        with quality_col2:
            st.metric(
                "Consistency", 
                f"{consistency_score:.1f}%",
                help="Percentage of unique rows (no duplicates)"
            )
        
        with quality_col3:
            validity_issues = len(empty_columns) + len(constant_columns)
            validity_score = max(0, 100 - (validity_issues / len(df.columns) * 100)) if len(df.columns) > 0 else 100
            st.metric(
                "Validity", 
                f"{validity_score:.1f}%",
                help="Percentage of columns without structural issues"
            )
        
        with quality_col4:
            overall_quality = (completeness_score * 0.4 + consistency_score * 0.3 + validity_score * 0.3)
            st.metric(
                "Overall Quality", 
                f"{overall_quality:.1f}%",
                help="Weighted average of all quality metrics"
            )
        
        # Quality Issues Alert
        issues = []
        if completeness_score < 90:
            issues.append(f"Low completeness: {missing_count:,} missing values")
        if duplicate_count > 0:
            issues.append(f"Data duplicates: {duplicate_count:,} duplicate rows")
        if empty_columns:
            issues.append(f"Empty columns: {len(empty_columns)} columns with no data")
        if constant_columns:
            issues.append(f"Constant columns: {len(constant_columns)} columns with only one value")
        
        if issues:
            st.warning("Data Quality Issues Detected:", width=500)
            for issue in issues:
                st.write(f"• {issue}")
        else:
            st.success("No major data quality issues detected!")
        
        # Column information with quality indicators
        st.subheader("Column Quality Analysis")
        col_quality_data = []
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
            unique_count = df[col].nunique()
            
            # Determine quality status
            if null_pct == 100:
                quality_status = "Empty"
            elif unique_count <= 1:
                quality_status = "Constant"
            elif null_pct > 50:
                quality_status = "Poor"
            elif null_pct > 20:
                quality_status = "Fair"
            elif null_pct > 5:
                quality_status = "Good"
            else:
                quality_status = "Excellent"
            
            col_quality_data.append({
                'Column': col,
                'Type': str(df[col].dtype),
                'Non-Null': f"{len(df) - null_count:,}",
                'Missing': f"{null_count:,}",
                'Missing %': f"{null_pct:.1f}%",
                'Unique Values': f"{unique_count:,}",
                'Quality': quality_status
            })
        
        col_quality_df = pd.DataFrame(col_quality_data)
        st.dataframe(col_quality_df, width='stretch')
        


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
        st.success("Connected to Snowflake")
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


def handle_sidebar_oracle_connection():
    """Backwards compatibility - Oracle connection functionality for sidebar"""
    from src.ui.data_sources import handle_oracle_connection, handle_oracle_data_loading
    
    # If not connected, show connection interface
    if not st.session_state.get('oracle_connected', False):
        handle_oracle_connection()
    else:
        # If connected, show data loading interface
        st.success("Connected to Oracle Database")
        handle_oracle_data_loading()
        
        # Add disconnect option
        st.markdown("---")
        if st.button("Disconnect from Oracle", key="sidebar_oracle_disconnect"):
            st.session_state.oracle_connected = False
            if 'oracle_connector' in st.session_state:
                try:
                    st.session_state.oracle_connector.close()
                except:
                    pass
                del st.session_state.oracle_connector
            st.rerun()


def handle_sidebar_hdfs_connection():
    """Backwards compatibility - HDFS connection functionality for sidebar"""
    from src.ui.data_sources import handle_hdfs_connection, handle_hdfs_data_loading
    
    # If not connected, show connection interface
    if not st.session_state.get('hdfs_connected', False):
        handle_hdfs_connection()
    else:
        # If connected, show data loading interface
        st.success("Connected to HDFS")
        handle_hdfs_data_loading()
        
        # Add disconnect option
        st.markdown("---")
        if st.button("Disconnect from HDFS", key="sidebar_hdfs_disconnect"):
            st.session_state.hdfs_connected = False
            if 'hdfs_connector' in st.session_state:
                try:
                    st.session_state.hdfs_connector.close()
                except:
                    pass
                del st.session_state.hdfs_connector
            st.rerun()