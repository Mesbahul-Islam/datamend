"""
Visualization Components for Dataset Comparison
Contains all chart and visualization functions for the comparison system
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any

# Import our utilities
from .utils import get_severity_emoji, get_change_type_emoji

# Import metadata types
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.metadata.metadata_extractor import DatasetMetadata


def create_metadata_comparison_charts(comparison_result: Dict[str, Any], 
                                     metadata1: DatasetMetadata, 
                                     metadata2: DatasetMetadata):
    """
    Create visualization charts for metadata comparison
    
    Args:
        comparison_result: Results from metadata comparison
        metadata1: First dataset metadata
        metadata2: Second dataset metadata
        
    Returns:
        Plotly figure with multiple subplots
    """
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Change Types Distribution', 
            'Dataset Size Comparison',
            'Data Quality Comparison', 
            'Column Type Distribution'
        ),
        specs=[
            [{'type': 'pie'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ]
    )
    
    # 1. Change Types Distribution (Pie chart)
    if comparison_result.get('changes'):
        change_types = {}
        for change in comparison_result['changes']:
            change_type = change['type'].replace('_', ' ').title()
            change_types[change_type] = change_types.get(change_type, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(change_types.keys()),
                values=list(change_types.values()),
                hole=0.3,
                name="Change Types"
            ),
            row=1, col=1
        )
    
    # 2. Dataset Size Comparison (Bar chart)
    datasets = [metadata1.name, metadata2.name]
    row_counts = [metadata1.row_count, metadata2.row_count]
    
    fig.add_trace(
        go.Bar(
            x=datasets,
            y=row_counts,
            name="Row Count",
            marker_color=['lightblue', 'lightcoral']
        ),
        row=1, col=2
    )
    
    # 3. Data Quality Comparison
    quality_scores = [metadata1.data_quality_score, metadata2.data_quality_score]
    
    fig.add_trace(
        go.Bar(
            x=datasets,
            y=quality_scores,
            name="Quality Score",
            marker_color=['lightgreen', 'lightyellow']
        ),
        row=2, col=1
    )
    
    # 4. Column Type Distribution
    col_counts1 = [len(metadata1.numeric_columns), len(metadata1.categorical_columns), len(metadata1.datetime_columns)]
    col_counts2 = [len(metadata2.numeric_columns), len(metadata2.categorical_columns), len(metadata2.datetime_columns)]
    
    fig.add_trace(
        go.Bar(
            x=['Numeric', 'Categorical', 'DateTime'],
            y=col_counts1,
            name=metadata1.name,
            marker_color='lightblue'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=['Numeric', 'Categorical', 'DateTime'],
            y=col_counts2,
            name=metadata2.name,
            marker_color='lightcoral'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Metadata Comparison Analysis")
    
    return fig


def display_change_details(change: Dict[str, Any], metadata1: DatasetMetadata, metadata2: DatasetMetadata):
    """
    Display detailed information about a specific change
    
    Args:
        change: Change information dictionary
        metadata1: First dataset metadata
        metadata2: Second dataset metadata
    """
    
    severity_emoji = get_severity_emoji(change.get('severity', 'low'))
    
    st.write(f"{severity_emoji} **{change['description']}**")
    
    # Display change details based on type
    details = change.get('details', {})
    
    if change['type'] == 'row_count_change':
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Previous", f"{details.get('previous', 0):,}")
        with col2:
            st.metric("Current", f"{details.get('current', 0):,}")
        with col3:
            change_val = details.get('change', 0)
            st.metric("Change", f"{change_val:+,}")
    
    elif change['type'] in ['columns_added', 'columns_removed']:
        columns = details.get('columns', [])
        if columns:
            st.write("**Affected Columns:**")
            for col in columns:
                st.write(f"• `{col}`")
    
    elif change['type'] == 'data_type_changes':
        type_changes = details.get('changes', [])
        if type_changes:
            change_df = pd.DataFrame(type_changes)
            st.dataframe(change_df, use_container_width=True)
    
    elif change['type'] == 'quality_score_change':
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Previous Score", f"{details.get('previous_score', 0):.1f}%")
        with col2:
            st.metric("Current Score", f"{details.get('current_score', 0):.1f}%")
        with col3:
            change_val = details.get('change', 0)
            st.metric("Change", f"{change_val:+.1f}%")
    
    elif change['type'] == 'column_level_changes':
        column_changes = details.get('column_changes', [])
        if column_changes:
            st.write(f"**{len(column_changes)} columns affected:**")
            for col_change in column_changes[:5]:  # Show first 5
                st.write(f"• **{col_change['column']}**: {col_change['description']}")
            if len(column_changes) > 5:
                st.write(f"... and {len(column_changes) - 5} more columns")
    
    elif change['type'] == 'fingerprint_changes':
        hash_changes = details.get('hash_changes', [])
        for hash_change in hash_changes:
            st.write(f"• **{hash_change['hash_type'].title()}**: {hash_change['description']}")
            st.write(f"  *Impact: {hash_change['impact']}*")


def display_dataset_metadata_summary(metadata: DatasetMetadata, show_hashes: bool):
    """
    Display a summary of dataset metadata
    
    Args:
        metadata: Dataset metadata to display
        show_hashes: Whether to show hash details
    """
    
    # Basic information
    st.write("**Basic Information:**")
    st.write(f"• **Source**: {metadata.source_type}")
    st.write(f"• **Rows**: {metadata.row_count:,}")
    st.write(f"• **Columns**: {metadata.column_count}")
    st.write(f"• **Memory Usage**: {metadata.memory_usage / 1024 / 1024:.1f} MB")
    st.write(f"• **Created**: {metadata.creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Data quality
    st.write("**Data Quality:**")
    st.write(f"• **Quality Score**: {metadata.data_quality_score:.1f}%")
    st.write(f"• **Total Nulls**: {metadata.total_null_count:,}")
    st.write(f"• **Duplicate Rows**: {metadata.total_duplicate_rows:,}")
    
    # Column types
    st.write("**Column Types:**")
    st.write(f"• **Numeric**: {len(metadata.numeric_columns)}")
    st.write(f"• **Categorical**: {len(metadata.categorical_columns)}")
    st.write(f"• **DateTime**: {len(metadata.datetime_columns)}")
    
    if show_hashes:
        st.write("**Fingerprints:**")
        st.write(f"• **Schema**: `{metadata.schema_hash[:12]}...`")
        st.write(f"• **Structure**: `{metadata.structure_hash[:12]}...`")
        st.write(f"• **Content**: `{metadata.content_hash[:12]}...`")


def display_hash_comparison(metadata1: DatasetMetadata, metadata2: DatasetMetadata):
    """
    Display hash comparison details
    
    Args:
        metadata1: First dataset metadata
        metadata2: Second dataset metadata
    """
    
    hash_comparisons = [
        ('Schema Hash', metadata1.schema_hash, metadata2.schema_hash),
        ('Structure Hash', metadata1.structure_hash, metadata2.structure_hash),
        ('Content Hash', metadata1.content_hash, metadata2.content_hash),
        ('Stats Hash', metadata1.stats_hash, metadata2.stats_hash)
    ]
    
    for hash_name, hash1, hash2 in hash_comparisons:
        status = "✅ Match" if hash1 == hash2 else "❌ Different"
        st.write(f"**{hash_name}**: {status}")
        if hash1 != hash2:
            st.write(f"  • Dataset 1: `{hash1[:20]}...`")
            st.write(f"  • Dataset 2: `{hash2[:20]}...`")
