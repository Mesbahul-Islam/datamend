"""
Data Comparison and Root Cause Analysis Module
Main UI interface for dataset comparison using metadata analysis for efficient change detection
"""

import streamlit as st
import pandas as pd

# Import only what we actually use
from ..comparison.utils import get_dataset_dataframe, get_dataset_source_type
from ..comparison.engine import run_metadata_comparison


def data_comparison_tab():
    """Data comparison and root cause analysis tab using metadata-based analysis"""
    st.header("🔍 Dataset Comparison & Root Cause Analysis")
    st.info("Fast metadata-based comparison to identify changes and inconsistencies between datasets")
    
    if not st.session_state.get('datasets') or len(st.session_state.datasets) < 2:
        st.warning("You need at least 2 datasets to perform comparison. Please load data using the sidebar.")
        return
    
    # Dataset selection for comparison
    st.subheader("📊 Select Datasets for Comparison")
    
    col1, col2 = st.columns(2)
    
    dataset_names = list(st.session_state.datasets.keys())
    
    with col1:
        dataset1_name = st.selectbox("Primary Dataset (Reference):", dataset_names, key="comp_dataset1")
        dataset1 = get_dataset_dataframe(dataset1_name)
        if dataset1 is not None:
            source_type1 = get_dataset_source_type(dataset1_name)
            st.caption(f"🔗 Source: {source_type1}")
            st.caption(f"📏 Shape: {dataset1.shape[0]:,} rows × {dataset1.shape[1]} columns")
    
    with col2:
        available_datasets = [name for name in dataset_names if name != dataset1_name]
        if available_datasets:
            dataset2_name = st.selectbox("Comparison Dataset:", available_datasets, key="comp_dataset2")
            dataset2 = get_dataset_dataframe(dataset2_name)
            if dataset2 is not None:
                source_type2 = get_dataset_source_type(dataset2_name)
                st.caption(f"🔗 Source: {source_type2}")
                st.caption(f"📏 Shape: {dataset2.shape[0]:,} rows × {dataset2.shape[1]} columns")
        else:
            st.warning("Please select a different dataset for comparison")
            return
    
    if dataset1 is None or dataset2 is None:
        st.error("Error loading selected datasets")
        return
    
    # Comparison configuration (simplified for fast metadata analysis)
    st.subheader("⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_metadata = st.checkbox("Export metadata to JSON", value=False,
                                     help="Export detailed metadata to JSON file")
    
    with col2:
        show_hashes = st.checkbox("Show hash details", value=False,
                                help="Display hash information (if available)")
    
    # Run comparison
    if st.button("🚀 Run Metadata Analysis", type="primary"):
        run_metadata_comparison(
            dataset1, dataset2, dataset1_name, dataset2_name,
            export_metadata, show_hashes
        )
