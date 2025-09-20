"""
Reusable UI Components for Data Quality Engine
"""

import streamlit as st
import pandas as pd


def show_data_preview(df: pd.DataFrame):
    """Show a preview of the loaded data"""
    st.subheader("Data Preview")
    

    # Column information
    st.write("Column Information:")
    column_info = []
    for col in df.columns:
        col_info = {
            "Column": col,
            "Type": str(df[col].dtype),
            "Non-Null Count": f"{df[col].count():,}",
            "Null Count": f"{df[col].isnull().sum():,}",
            "Unique Values": f"{df[col].nunique():,}"
        }
        column_info.append(col_info)
    
    column_df = pd.DataFrame(column_info)
    st.dataframe(column_df, width='stretch', hide_index=True)


def display_loading_info(df: pd.DataFrame, processing_time: float = None):
    """Display information about loaded dataset"""
    dataset_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"🟢 Data loaded successfully!")
        st.write(f"Shape: {len(df):,} rows × {len(df.columns)} columns")
        
    with col2:
        st.write(f"Size: {dataset_size_mb:.1f} MB")
        if processing_time:
            st.write(f"Load time: {processing_time:.2f}s")
            
    with col3:
        missing_cells = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        st.write(f"Missing data: {missing_pct:.1f}%")


def create_file_upload_section(key_suffix: str = ""):
    """Create a file upload section with configuration options"""
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type=['csv'],
        accept_multiple_files=True,
        key=f"file_uploader_{key_suffix}",
        help="Upload one or more CSV files for analysis"
    )
    
    if uploaded_files:
        st.success(f"� {len(uploaded_files)} file(s) selected")
        
        # File upload configuration
        st.subheader("Upload Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            encoding = st.selectbox(
                "File Encoding",
                ["utf-8", "latin-1", "cp1252", "utf-16"],
                index=0,
                key=f"encoding_{key_suffix}",
                help="Character encoding of the CSV files"
            )
        
        with col2:
            delimiter = st.selectbox(
                "Delimiter",
                [",", ";", "\t", "|"],
                index=0,
                format_func=lambda x: {"," : "Comma (,)", ";" : "Semicolon (;)", "\t" : "Tab", "|" : "Pipe (|)"}[x],
                key=f"delimiter_{key_suffix}",
                help="Character used to separate fields"
            )
        
        with col3:
            sample_rows = st.number_input(
                "Preview Rows",
                min_value=5,
                max_value=1000,
                value=100,
                step=50,
                key=f"sample_rows_{key_suffix}",
                help="Number of rows to show in preview"
            )
        
        return uploaded_files, encoding, delimiter, sample_rows
    
    return None, None, None, None
