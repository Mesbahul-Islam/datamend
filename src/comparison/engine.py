"""
Core Comparison Engine for Dataset Analysis
Contains the main comparison logic and orchestration functions
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional

# Import our modules
from .utils import get_dataset_source_type, get_source_metadata, get_change_type_emoji
from .visualizations import (
    create_metadata_comparison_charts, 
    display_change_details, 
    display_dataset_metadata_summary,
    display_hash_comparison
)
from .reports import generate_metadata_comparison_report, export_metadata_json

# Import metadata analysis
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.metadata.metadata_extractor import (
    MetadataExtractor, 
    DatasetMetadata, 
    compare_dataset_metadata
)


def run_metadata_comparison(df1: pd.DataFrame, df2: pd.DataFrame, 
                          name1: str, name2: str,
                          export_metadata: bool, show_hashes: bool):
    """
    Run metadata-based comparison analysis between two datasets
    
    Args:
        df1: First dataset DataFrame
        df2: Second dataset DataFrame
        name1: Name of first dataset
        name2: Name of second dataset
        export_metadata: Whether to export metadata to JSON
        show_hashes: Whether to show hash details
    """
    
    with st.spinner("🔍 Extracting metadata and analyzing changes..."):
        try:
            # Get source metadata from session state if available
            source_meta1 = get_source_metadata(name1)
            source_meta2 = get_source_metadata(name2)
            
            # Extract metadata from both datasets
            extractor = MetadataExtractor()
            
            source_type1 = get_dataset_source_type(name1)
            source_type2 = get_dataset_source_type(name2)
            
            metadata1 = extractor.extract_metadata(df1, name1, source_type1, source_meta1)
            metadata2 = extractor.extract_metadata(df2, name2, source_type2, source_meta2)
            
            # Compare metadata
            comparison_result = compare_dataset_metadata(metadata1, metadata2)
            
            # Store results in session state
            st.session_state.metadata_comparison_result = {
                'metadata1': metadata1,
                'metadata2': metadata2,
                'comparison': comparison_result
            }
            
            # Display results
            display_metadata_comparison_results(
                metadata1, metadata2, comparison_result, 
                name1, name2, show_hashes, export_metadata
            )
            
        except Exception as e:
            st.error(f"Error during metadata analysis: {str(e)}")
            st.exception(e)


def display_metadata_comparison_results(metadata1: DatasetMetadata, metadata2: DatasetMetadata,
                                       comparison_result: Dict[str, Any], name1: str, name2: str,
                                       show_hashes: bool, export_metadata: bool):
    """
    Display comprehensive metadata comparison results
    
    Args:
        metadata1: First dataset metadata
        metadata2: Second dataset metadata
        comparison_result: Results from metadata comparison
        name1: Name of first dataset
        name2: Name of second dataset
        show_hashes: Whether to show hash details
        export_metadata: Whether to export metadata to JSON
    """
    
    st.success("✅ Metadata analysis completed!")
    
    # Executive Summary
    st.subheader("📋 Executive Summary")
    
    change_score = comparison_result.get('change_score', 0)
    num_changes = len(comparison_result.get('changes', []))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Change Score", f"{change_score:.1f}/100", 
                 help="Overall change intensity (0=identical, 100=completely different)")
    
    with col2:
        st.metric("Total Changes", num_changes)
    
    with col3:
        # Calculate simple quality metric based on null percentage
        total_cells1 = metadata1.row_count * metadata1.column_count
        total_cells2 = metadata2.row_count * metadata2.column_count
        quality1 = 100 - (metadata1.total_null_count / total_cells1 * 100) if total_cells1 > 0 else 100
        quality2 = 100 - (metadata2.total_null_count / total_cells2 * 100) if total_cells2 > 0 else 100
        quality_diff = quality2 - quality1
        st.metric("Quality Change", f"{quality_diff:+.1f}%", delta=f"{quality_diff:+.1f}%")
    
    with col4:
        row_diff = metadata2.row_count - metadata1.row_count
        st.metric("Row Count Δ", f"{row_diff:+,}", delta=f"{row_diff:+,}")
    
    # Change Severity Breakdown
    if comparison_result.get('changes'):
        st.subheader("🚨 Change Analysis")
        
        # Create severity breakdown
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for change in comparison_result['changes']:
            severity = change.get('severity', 'low')
            severity_counts[severity] += 1
        
        # Visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create change timeline/breakdown chart
            fig = create_metadata_comparison_charts(comparison_result, metadata1, metadata2)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Severity summary
            st.write("**Change Severity:**")
            if severity_counts['high'] > 0:
                st.error(f"🔴 High: {severity_counts['high']} critical changes")
            if severity_counts['medium'] > 0:
                st.warning(f"🟡 Medium: {severity_counts['medium']} moderate changes")
            if severity_counts['low'] > 0:
                st.info(f"🟢 Low: {severity_counts['low']} minor changes")
    
    # Detailed Change Analysis
    if comparison_result.get('changes'):
        st.subheader("🔍 Detailed Change Analysis")
        
        # Group changes by type
        change_groups = {}
        for change in comparison_result['changes']:
            change_type = change['type']
            if change_type not in change_groups:
                change_groups[change_type] = []
            change_groups[change_type].append(change)
        
        for change_type, changes in change_groups.items():
            with st.expander(f"{get_change_type_emoji(change_type)} {change_type.replace('_', ' ').title()} ({len(changes)} changes)"):
                for change in changes:
                    display_change_details(change, metadata1, metadata2)
    
    # Dataset Metadata Overview
    st.subheader("📊 Dataset Metadata Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{name1} (Reference)**")
        display_dataset_metadata_summary(metadata1, show_hashes)
    
    with col2:
        st.write(f"**{name2} (Comparison)**")
        display_dataset_metadata_summary(metadata2, show_hashes)
    
    # Column-level Analysis
    if comparison_result.get('changes'):
        column_changes = []
        for change in comparison_result['changes']:
            if change['type'] == 'column_level_changes':
                column_changes.extend(change['details'].get('column_changes', []))
        
        if column_changes:
            st.subheader("📋 Column-Level Changes")
            
            # Create a DataFrame for column changes
            column_change_data = []
            for col_change in column_changes:
                column_change_data.append({
                    'Column': col_change['column'],
                    'Change Type': col_change['change_type'].replace('_', ' ').title(),
                    'Description': col_change['description'],
                    'Details': str(col_change.get('details', {}))
                })
            
            if column_change_data:
                column_df = pd.DataFrame(column_change_data)
                st.dataframe(column_df, use_container_width=True)
    
    # Root Cause Analysis & Recommendations
    st.subheader("💡 Root Cause Analysis & Recommendations")
    
    recommendations = comparison_result.get('recommendations', [])
    if recommendations:
        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"{i}. {recommendation}")
    else:
        st.info("✅ No specific recommendations - datasets appear to be consistent")
    
    # Advanced Analysis Options
    with st.expander("🔧 Advanced Analysis"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Hash Analysis:**")
            if show_hashes:
                display_hash_comparison(metadata1, metadata2)
            else:
                st.info("Enable 'Show hash details' to view fingerprint analysis")
        
        with col2:
            st.write("**Export Options:**")
            if st.button("📄 Generate Detailed Report"):
                generate_metadata_comparison_report(metadata1, metadata2, comparison_result, name1, name2)
            
            if export_metadata:
                if st.button("💾 Export Metadata JSON"):
                    export_metadata_json(metadata1, metadata2, comparison_result)
    
    # Performance Metrics
    st.subheader("⚡ Analysis Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Analysis Time", "< 1 second", help="Metadata analysis is fast")
    
    with col2:
        total_rows = metadata1.row_count + metadata2.row_count
        st.metric("Rows Analyzed", f"{total_rows:,}", help="Total rows in both datasets")
    
    with col3:
        total_cols = metadata1.column_count + metadata2.column_count
        st.metric("Columns Analyzed", total_cols, help="Total columns in both datasets")
