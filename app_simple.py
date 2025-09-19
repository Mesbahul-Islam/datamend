"""
Data Quality Engine - Streamlit Frontend

A simplified data quality management interface focused on:
- Data upload (CSV/Excel)
- Interactive data profiling
- Basic anomaly detection
- AI-driven recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_quality.engine import DataQualityEngine, DataQualityReport
from data_quality.anomaly_detector import StatisticalAnomalyDetector
from connectors.data_connectors import DataConnectorFactory
from llm.analyzer import DataQualityLLMAnalyzer, LLMConfig

# Configure Streamlit page
st.set_page_config(
    page_title="Data Quality Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'quality_report' not in st.session_state:
        st.session_state.quality_report = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'connector' not in st.session_state:
        st.session_state.connector = None
    if 'profiling_complete' not in st.session_state:
        st.session_state.profiling_complete = False

def main():
    """Main application function"""
    st.title("🔍 Data Quality Engine")
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Engine configuration
        chunk_size = st.slider("Chunk Size", min_value=500, max_value=10000, value=1000, step=500,
                              help="Size of data chunks for parallel processing")
        max_workers = st.slider("Max Workers", min_value=1, max_value=8, value=4,
                               help="Number of parallel threads")
        anomaly_threshold = st.slider("Anomaly Threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.1,
                                    help="Z-score threshold for anomaly detection")
        
        # LLM configuration
        st.subheader("🤖 AI Recommendations")
        use_llm = st.checkbox("Enable AI Recommendations", value=False)
        if use_llm:
            api_key = st.text_input("LLM API Key", type="password", help="Optional: Enter your LLM API key for AI recommendations")
            model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"], index=0)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Source", "📊 Data Profiling", "🎯 Anomaly Detection", "🤖 AI Recommendations"])
    
    with tab1:
        data_source_tab()
    
    with tab2:
        data_profiling_tab(chunk_size, max_workers, anomaly_threshold)
    
    with tab3:
        anomaly_detection_tab(anomaly_threshold)
    
    with tab4:
        ai_recommendations_tab(use_llm, api_key if 'api_key' in locals() else "", model if 'model' in locals() else "gpt-3.5-turbo")

def data_source_tab():
    """Data source selection and connection tab"""
    st.header("📁 Data Source")
    
    # Data source selection
    source_type = st.selectbox("Select Data Source Type", 
                               ["CSV File", "Excel File"],
                               help="Choose your data source type")
    
    if source_type == "CSV File":
        handle_csv_upload()
    elif source_type == "Excel File":
        handle_excel_upload()

def handle_csv_upload():
    """Handle CSV file upload"""
    st.subheader("📄 CSV File Upload")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # CSV configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], index=0)
        with col2:
            delimiter = st.selectbox("Delimiter", [",", ";", "\t", "|"], index=0)
        with col3:
            load_full = st.checkbox("Load Full Dataset", value=True, 
                                   help="Uncheck to load only a sample for testing")
        
        if not load_full:
            sample_rows = st.number_input("Sample Rows", min_value=100, max_value=10000, value=1000)
        else:
            sample_rows = None
        
        if st.button("Load CSV Data", type="primary"):
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
                        st.session_state.data = df
                        st.session_state.connector = connector
                        
                        # Clean up temp file
                        os.remove(temp_path)
                        
                        st.success(f"✅ Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
                        
                        # Show data preview
                        show_data_preview(df)
                    else:
                        st.error("❌ Failed to connect to CSV file")
                        
            except Exception as e:
                st.error(f"❌ Error loading CSV: {str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

def handle_excel_upload():
    """Handle Excel file upload"""
    st.subheader("📊 Excel File Upload")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
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
                                       help="Uncheck to load only a sample for testing")
            
            if not load_full:
                sample_rows = st.number_input("Sample Rows", min_value=100, max_value=10000, value=1000)
            else:
                sample_rows = None
            
            if st.button("Load Excel Data", type="primary"):
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

def show_data_preview(df: pd.DataFrame):
    """Show a preview of the loaded data"""
    st.subheader("📋 Data Preview")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
    with col4:
        null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Null Values", f"{null_percentage:.1f}%")
    
    # Data types
    st.subheader("📝 Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(col_info, use_container_width=True)
    
    # Data sample
    st.subheader("🔍 Data Sample")
    st.dataframe(df.head(10), use_container_width=True)

def data_profiling_tab(chunk_size: int, max_workers: int, anomaly_threshold: float):
    """Data profiling tab"""
    st.header("📊 Data Profiling")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Source tab")
        return
    
    df = st.session_state.data
    
    # Profiling controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Calculate dataset size information
        dataset_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        estimated_processing_time = len(df) * 0.001  # Rough estimate: 1ms per 1000 rows
        
        # Determine processing strategy message
        if len(df) > 100000:
            processing_strategy = "🚀 Large dataset - using parallel processing"
        else:
            processing_strategy = "⚡ Standard dataset - using optimized sequential processing"
        
        st.write(f"**Dataset:** {len(df):,} rows × {len(df.columns)} columns ({dataset_size_mb:.1f} MB)")
        st.caption(f"{processing_strategy} • Est. processing time: ~{estimated_processing_time:.1f}s")
    
    with col2:
        if st.button("🔄 Run Profiling", type="primary"):
            run_data_profiling(df, chunk_size, max_workers, anomaly_threshold)
    
    # Show results if available
    if st.session_state.quality_report:
        display_profiling_results(st.session_state.quality_report)

def run_data_profiling(df: pd.DataFrame, chunk_size: int, max_workers: int, anomaly_threshold: float):
    """Run data profiling on the dataset"""
    try:
        with st.spinner("🔍 Running comprehensive data profiling..."):
            # Initialize engine
            engine = DataQualityEngine(
                chunk_size=chunk_size,
                max_workers=max_workers,
                anomaly_threshold=anomaly_threshold
            )
            
            # Run profiling with timing
            with st.spinner("🔍 Profiling your dataset..."):
                start_time = time.time()
                report = engine.profile_dataset(df, "Dataset")
                end_time = time.time()
                
                # Calculate profiling time
                profiling_time = end_time - start_time
                
            st.session_state.quality_report = report
            st.session_state.profiling_time = profiling_time
            st.session_state.profiling_complete = True
            
        st.success(f"✅ Data profiling completed successfully in {st.session_state.profiling_time:.2f} seconds!")
        
    except Exception as e:
        st.error(f"❌ Error during profiling: {str(e)}")

def display_profiling_results(report: DataQualityReport):
    """Display profiling results with meaningful insights"""
    
    # Executive Summary
    st.subheader("📈 Data Quality Executive Summary")
    
    # Quality interpretation
    quality_score = report.overall_quality_score
    if quality_score >= 90:
        quality_status = "🟢 **Excellent**"
        quality_message = "Your data is in excellent condition with minimal issues."
    elif quality_score >= 75:
        quality_status = "🟡 **Good**"
        quality_message = "Your data quality is good but has some areas for improvement."
    elif quality_score >= 50:
        quality_status = "🟠 **Fair**"
        quality_message = "Your data has moderate quality issues that should be addressed."
    else:
        quality_status = "🔴 **Poor**"
        quality_message = "Your data has significant quality issues requiring immediate attention."
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Overall Quality", f"{quality_score:.1f}/100", delta=quality_status.split()[1])
    
    with col2:
        critical_issues = len(report.critical_issues)
        st.metric("Critical Issues", critical_issues, delta="Fix Immediately" if critical_issues > 0 else "✅ None")
    
    with col3:
        complete_columns = sum(1 for profile in report.column_profiles.values() if profile.null_percentage < 5)
        st.metric("Complete Columns", f"{complete_columns}/{len(report.column_profiles)}")
    
    with col4:
        problematic_columns = sum(1 for profile in report.column_profiles.values() if len(profile.data_quality_issues or []) > 0)
        st.metric("Columns with Issues", problematic_columns)
        
    with col5:
        # Display profiling time if available
        if hasattr(st.session_state, 'profiling_time') and st.session_state.profiling_time:
            profiling_time = st.session_state.profiling_time
            if profiling_time < 1:
                time_display = f"{profiling_time*1000:.0f}ms"
                time_status = "⚡ Fast"
            elif profiling_time < 5:
                time_display = f"{profiling_time:.2f}s"
                time_status = "🟢 Quick"
            elif profiling_time < 15:
                time_display = f"{profiling_time:.1f}s"
                time_status = "🟡 Normal"
            else:
                time_display = f"{profiling_time:.1f}s"
                time_status = "🟠 Slow"
            st.metric("Profiling Time", time_display, delta=time_status)
        else:
            st.metric("Profiling Time", "N/A")
    
    st.info(f"**Assessment**: {quality_message}")
    
    # Critical Issues Alert
    if report.critical_issues:
        st.error("🚨 **Critical Issues Requiring Immediate Attention:**")
        for i, issue in enumerate(report.critical_issues, 1):
            st.write(f"{i}. {issue}")
        st.write("")
    
    # Column Health Overview
    st.subheader("📋 Column Health Assessment")
    
    if report.column_profiles:
        columns_summary = []
        for col_name, col_profile in report.column_profiles.items():
            # Calculate column health score
            col_health_score = 100
            if col_profile.null_percentage > 0:
                col_health_score -= min(col_profile.null_percentage * 0.8, 40)  # Max 40 points deduction
            if col_profile.data_quality_issues:
                col_health_score -= len(col_profile.data_quality_issues) * 15  # 15 points per issue
            col_health_score = max(0, col_health_score)
            
            # Health status
            if col_health_score >= 85:
                health_icon = "🟢"
                health_status = "Healthy"
            elif col_health_score >= 65:
                health_icon = "🟡"
                health_status = "Minor Issues"
            elif col_health_score >= 40:
                health_icon = "🟠" 
                health_status = "Needs Attention"
            else:
                health_icon = "🔴"
                health_status = "Critical"
            
            # Key insights about the column
            insights = []
            if col_profile.null_percentage > 20:
                insights.append(f"High missing data ({col_profile.null_percentage:.1f}%)")
            elif col_profile.null_percentage > 5:
                insights.append(f"Some missing data ({col_profile.null_percentage:.1f}%)")
            
            if col_profile.outliers_count > 0:
                insights.append(f"{col_profile.outliers_count} outliers detected")
            
            if col_profile.data_quality_issues:
                for issue in col_profile.data_quality_issues:
                    if "INCONSISTENT_CASING" in issue:
                        insights.append("Mixed text casing")
                    elif "WHITESPACE" in issue:
                        insights.append("Extra spaces detected")
                    elif "LOW_UNIQUENESS" in issue:
                        insights.append("Low data diversity")
            
            # Uniqueness insight
            if col_profile.unique_count == col_profile.total_count:
                uniqueness_note = "All unique"
            elif col_profile.unique_count == 1:
                uniqueness_note = "Single value"
            else:
                uniqueness_rate = (col_profile.unique_count / col_profile.total_count) * 100
                uniqueness_note = f"{uniqueness_rate:.1f}% unique"
            
            columns_summary.append({
                'Column': f"{health_icon} **{col_name}**",
                'Type': col_profile.data_type.title(),
                'Health Status': health_status,
                'Completeness': f"{100-col_profile.null_percentage:.1f}%",
                'Uniqueness': uniqueness_note,
                'Key Insights': "; ".join(insights) if insights else "No issues detected"
            })
        
        columns_df = pd.DataFrame(columns_summary)
        st.dataframe(columns_df, use_container_width=True, hide_index=True)
        
        # Data Quality Insights
        st.subheader("� Key Data Quality Insights")
        
        # Missing data analysis
        high_missing_cols = [col for col, profile in report.column_profiles.items() 
                           if profile.null_percentage > 10]
        if high_missing_cols:
            st.warning(f"**Missing Data Concern**: {len(high_missing_cols)} columns have >10% missing values: {', '.join(high_missing_cols)}")
        
        # Data type distribution
        type_counts = {}
        for profile in report.column_profiles.values():
            type_counts[profile.data_type] = type_counts.get(profile.data_type, 0) + 1
        
        st.info(f"**Data Composition**: {type_counts}")
        
        # Outlier summary
        total_outliers = sum(profile.outliers_count for profile in report.column_profiles.values())
        if total_outliers > 0:
            outlier_cols = [col for col, profile in report.column_profiles.items() if profile.outliers_count > 0]
            st.warning(f"**Outliers Detected**: {total_outliers} outliers found across {len(outlier_cols)} columns")
        else:
            st.success("**No Statistical Outliers**: No obvious outliers detected in numeric columns")
        
        # Quick action recommendations
        st.subheader("🎯 Immediate Action Items")
        
        actions = []
        if high_missing_cols:
            actions.append(f"🔸 Investigate missing data in: {', '.join(high_missing_cols[:3])}")
        
        inconsistent_cols = [col for col, profile in report.column_profiles.items() 
                           if profile.data_quality_issues and any("INCONSISTENT" in issue for issue in profile.data_quality_issues)]
        if inconsistent_cols:
            actions.append(f"🔸 Standardize formatting in: {', '.join(inconsistent_cols[:3])}")
        
        if total_outliers > 0:
            actions.append(f"🔸 Review outliers in numeric columns (check Anomaly Detection tab)")
        
        if not actions:
            st.success("✅ **No immediate actions required** - Your data quality looks good!")
        else:
            for action in actions:
                st.write(action)

def anomaly_detection_tab(anomaly_threshold: float):
    """Anomaly detection tab"""
    st.header("🎯 Anomaly Detection")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Source tab")
        return
    
    df = st.session_state.data
    
    # Anomaly detection controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        detection_method = st.selectbox(
            "Detection Method",
            ["Z-Score", "IQR", "Modified Z-Score", "All Methods"],
            help="Choose the statistical method for anomaly detection"
        )
    
    with col2:
        if st.button("🔍 Detect Anomalies", type="primary"):
            st.session_state.anomaly_method = detection_method
            st.session_state.run_anomaly_detection = True
    
    # Display results outside of columns for full width
    if getattr(st.session_state, 'run_anomaly_detection', False):
        run_anomaly_detection(df, st.session_state.anomaly_method, anomaly_threshold)
        st.session_state.run_anomaly_detection = False

def run_anomaly_detection(df: pd.DataFrame, method: str, threshold: float):
    """Run anomaly detection"""
    try:
        with st.spinner("🔍 Detecting anomalies..."):
            detector = StatisticalAnomalyDetector(z_threshold=threshold)
            
            # Start timing
            start_time = time.time()
            
            if method == "All Methods":
                results = detector.detect_anomalies(df)
            else:
                # Map method names to detector methods
                method_map = {
                    "Z-Score": "zscore",
                    "IQR": "iqr", 
                    "Modified Z-Score": "modified_zscore"
                }
                results = detector.detect_anomalies(df, methods=[method_map[method]])
            
            # End timing
            end_time = time.time()
            anomaly_detection_time = end_time - start_time
            
            # Store timing in session state
            st.session_state.anomaly_detection_time = anomaly_detection_time
            
            display_anomaly_results(results, df)
            
    except Exception as e:
        st.error(f"❌ Error during anomaly detection: {str(e)}")

def create_anomaly_visualizations(df: pd.DataFrame, column: str, anomaly_result: Any):
    """Create matplotlib visualizations for anomalies in a specific column"""
    
    # Set style for better looking plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    if not hasattr(anomaly_result, 'anomaly_indices') or not anomaly_result.anomaly_indices:
        return None
        
    # Get data for the column
    column_data = df[column].dropna()
    if len(column_data) == 0:
        return None
        
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Anomaly Analysis for Column: {column}', fontsize=16, fontweight='bold')
    
    # Get anomaly indices and values
    anomaly_indices = anomaly_result.anomaly_indices
    normal_indices = [i for i in column_data.index if i not in anomaly_indices]
    
    # 1. Scatter plot with highlighted anomalies
    ax1 = axes[0, 0]
    ax1.scatter(normal_indices, column_data.loc[normal_indices], 
               alpha=0.6, c='blue', label='Normal Data', s=30)
    if anomaly_indices:
        ax1.scatter(anomaly_indices, column_data.loc[anomaly_indices], 
                   alpha=0.8, c='red', label='Anomalies', s=60, marker='X')
    ax1.set_title('Data Points with Anomalies Highlighted', fontweight='bold')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot showing outliers
    ax2 = axes[0, 1]
    box_data = [column_data.loc[normal_indices], column_data.loc[anomaly_indices]] if anomaly_indices else [column_data]
    labels = ['Normal Data', 'Anomalies'] if anomaly_indices else ['All Data']
    bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral'] if anomaly_indices else ['lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_title('Box Plot Comparison', fontweight='bold')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram with anomaly distribution
    ax3 = axes[1, 0]
    
    # Plot histogram for normal data
    normal_data = column_data.loc[normal_indices]
    ax3.hist(normal_data, bins=30, alpha=0.7, color='blue', label='Normal Data', density=True)
    
    # Plot histogram for anomalies if any
    if anomaly_indices:
        anomaly_data = column_data.loc[anomaly_indices]
        ax3.hist(anomaly_data, bins=15, alpha=0.8, color='red', label='Anomalies', density=True)
    
    # Add threshold lines if available
    if hasattr(anomaly_result, 'threshold'):
        mean_val = column_data.mean()
        std_val = column_data.std()
        threshold = anomaly_result.threshold
        
        ax3.axvline(mean_val - threshold * std_val, color='orange', linestyle='--', 
                   label=f'Lower Threshold', alpha=0.7)
        ax3.axvline(mean_val + threshold * std_val, color='orange', linestyle='--', 
                   label=f'Upper Threshold', alpha=0.7)
    
    ax3.set_title('Distribution Analysis', fontweight='bold')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics comparison
    ax4 = axes[1, 1]
    
    # Calculate statistics
    normal_stats = {
        'Mean': normal_data.mean(),
        'Median': normal_data.median(),
        'Std': normal_data.std(),
        'Min': normal_data.min(),
        'Max': normal_data.max()
    }
    
    if anomaly_indices:
        anomaly_data = column_data.loc[anomaly_indices]
        anomaly_stats = {
            'Mean': anomaly_data.mean(),
            'Median': anomaly_data.median(),
            'Std': anomaly_data.std(),
            'Min': anomaly_data.min(),
            'Max': anomaly_data.max()
        }
        
        # Create comparison bar chart
        x_pos = np.arange(len(normal_stats))
        width = 0.35
        
        normal_values = list(normal_stats.values())
        anomaly_values = list(anomaly_stats.values())
        
        bars1 = ax4.bar(x_pos - width/2, normal_values, width, label='Normal Data', 
                       color='blue', alpha=0.7)
        bars2 = ax4.bar(x_pos + width/2, anomaly_values, width, label='Anomalies', 
                       color='red', alpha=0.7)
        
        ax4.set_xlabel('Statistics')
        ax4.set_ylabel('Value')
        ax4.set_title('Statistical Comparison', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(normal_stats.keys(), rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    else:
        # Just show normal data statistics as text
        stats_text = "\n".join([f"{k}: {v:.2f}" for k, v in normal_stats.items()])
        ax4.text(0.5, 0.5, f"Column Statistics:\n\n{stats_text}", 
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax4.set_title('Column Statistics', fontweight='bold')
        ax4.axis('off')
    
    plt.tight_layout()
    return fig

def create_overall_anomaly_summary_plot(results: Dict[str, Dict[str, Any]], df: pd.DataFrame):
    """Create an overall summary visualization of anomalies across all columns"""
    
    # Calculate anomaly counts per column
    anomaly_counts = {}
    anomaly_percentages = {}
    
    for column, column_results in results.items():
        total_anomalies = 0
        for method, anomaly_result in column_results.items():
            if hasattr(anomaly_result, 'anomaly_indices'):
                total_anomalies += len(set(anomaly_result.anomaly_indices))
        
        anomaly_counts[column] = total_anomalies
        anomaly_percentages[column] = (total_anomalies / len(df)) * 100 if len(df) > 0 else 0
    
    if not any(anomaly_counts.values()):
        return None
    
    # Create summary plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Anomaly Detection Summary Across All Columns', fontsize=16, fontweight='bold')
    
    # 1. Bar chart of anomaly counts
    ax1 = axes[0]
    columns = list(anomaly_counts.keys())
    counts = list(anomaly_counts.values())
    
    # Color bars based on severity
    colors = []
    for count, total in zip(counts, [len(df)] * len(counts)):
        percentage = (count / total) * 100 if total > 0 else 0
        if percentage > 10:
            colors.append('red')
        elif percentage > 5:
            colors.append('orange')
        elif percentage > 1:
            colors.append('yellow')
        else:
            colors.append('green')
    
    bars = ax1.bar(columns, counts, color=colors, alpha=0.7)
    ax1.set_title('Anomaly Counts by Column', fontweight='bold')
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Number of Anomalies')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Pie chart of anomaly distribution
    ax2 = axes[1]
    
    # Filter out columns with no anomalies for pie chart
    non_zero_columns = {k: v for k, v in anomaly_counts.items() if v > 0}
    
    if non_zero_columns:
        labels = list(non_zero_columns.keys())
        sizes = list(non_zero_columns.values())
        
        # Use different colors for different severity levels
        pie_colors = []
        for col in labels:
            percentage = anomaly_percentages[col]
            if percentage > 10:
                pie_colors.append('#ff4444')  # Red
            elif percentage > 5:
                pie_colors.append('#ff8800')  # Orange
            elif percentage > 1:
                pie_colors.append('#ffdd00')  # Yellow
            else:
                pie_colors.append('#44ff44')  # Green
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=pie_colors, startangle=90)
        ax2.set_title('Anomaly Distribution by Column', fontweight='bold')
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax2.text(0.5, 0.5, 'No Anomalies\nDetected', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        ax2.set_title('Anomaly Distribution', fontweight='bold')
    
    # Add legend for severity levels
    severity_legend = [
        mpatches.Patch(color='red', label='Critical (>10%)'),
        mpatches.Patch(color='orange', label='High (5-10%)'),
        mpatches.Patch(color='yellow', label='Medium (1-5%)'),
        mpatches.Patch(color='green', label='Low (<1%)')
    ]
    
    fig.legend(handles=severity_legend, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=4, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    return fig

def display_anomaly_results(results: Dict[str, Dict[str, Any]], df: pd.DataFrame):
    """Display anomaly detection results with meaningful insights"""
    
    # Use a container for full width display
    with st.container():
        st.subheader("🎯 Anomaly Detection Assessment")
        
        # Display timing information if available
        if hasattr(st.session_state, 'anomaly_detection_time') and st.session_state.anomaly_detection_time:
            detection_time = st.session_state.anomaly_detection_time
            if detection_time < 0.1:
                time_display = f"⚡ Completed in {detection_time*1000:.0f}ms"
            elif detection_time < 1:
                time_display = f"🟢 Completed in {detection_time*1000:.0f}ms"
            elif detection_time < 5:
                time_display = f"🟡 Completed in {detection_time:.2f}s"
            else:
                time_display = f"🟠 Completed in {detection_time:.1f}s"
            st.info(time_display)
        
        if not results:
            st.info("No numeric columns found for anomaly detection")
            return
        
        # Calculate comprehensive anomaly statistics
        total_anomalies = 0
        anomaly_by_column = {}
        severity_analysis = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
        
        for column, column_results in results.items():
            column_anomalies = set()
            max_severity_score = 0
            
            for method, anomaly_result in column_results.items():
                if hasattr(anomaly_result, 'anomaly_indices'):
                    column_anomalies.update(anomaly_result.anomaly_indices)
                    # Calculate severity based on anomaly scores
                    if hasattr(anomaly_result, 'anomaly_scores') and anomaly_result.anomaly_scores:
                        max_score = max(anomaly_result.anomaly_scores) if anomaly_result.anomaly_scores else 0
                        max_severity_score = max(max_severity_score, max_score)
            
            anomaly_count = len(column_anomalies)
            anomaly_by_column[column] = {
                'count': anomaly_count,
                'percentage': (anomaly_count / len(df)) * 100 if len(df) > 0 else 0,
                'severity_score': max_severity_score
            }
            total_anomalies += anomaly_count
            
            # Categorize severity
            if anomaly_count > 0:
                anomaly_rate = (anomaly_count / len(df)) * 100
                if anomaly_rate > 10:
                    severity_analysis["Critical"] += 1
                elif anomaly_rate > 5:
                    severity_analysis["High"] += 1
                elif anomaly_rate > 1:
                    severity_analysis["Medium"] += 1
                else:
                    severity_analysis["Low"] += 1
        
        # Executive Summary
        total_anomaly_rate = (total_anomalies / len(df)) * 100 if len(df) > 0 else 0
        
        if total_anomaly_rate == 0:
            assessment_status = "🟢 **Excellent**"
            assessment_message = "No anomalies detected. Your data appears statistically normal."
        elif total_anomaly_rate < 1:
            assessment_status = "🟡 **Good**"  
            assessment_message = "Very few anomalies detected. This is typically expected in real data."
        elif total_anomaly_rate < 5:
            assessment_status = "🟠 **Moderate**"
            assessment_message = "Some anomalies detected. Review these values to determine if they're valid or errors."
        else:
            assessment_status = "🔴 **High**"
            assessment_message = "Many anomalies detected. This may indicate data quality issues or unusual patterns."
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Anomalies", total_anomalies, delta=f"{total_anomaly_rate:.2f}% of data")
        
        with col2:
            affected_columns = len([col for col, data in anomaly_by_column.items() if data['count'] > 0])
            st.metric("Affected Columns", f"{affected_columns}/{len(results)}")
        
        with col3:
            if severity_analysis["Critical"] > 0:
                severity_status = "Critical"
                severity_color = "🔴"
            elif severity_analysis["High"] > 0:
                severity_status = "High"
                severity_color = "🟠"
            elif severity_analysis["Medium"] > 0:
                severity_status = "Medium"
                severity_color = "🟡"
            else:
                severity_status = "Low"
                severity_color = "🟢"
            st.metric("Max Severity", f"{severity_color} {severity_status}")
        
        with col4:
            # Data reliability score based on anomaly rate
            reliability = max(0, 100 - (total_anomaly_rate * 4))  # Reduce by 4 points per % anomaly
            st.metric("Data Reliability", f"{reliability:.0f}/100")
        
        st.info(f"**Assessment**: {assessment_message}")
        
        # Add overall summary visualization
        st.subheader("📊 Anomaly Detection Visualizations")
        
        # Create and display overall summary plot
        summary_fig = create_overall_anomaly_summary_plot(results, df)
        if summary_fig:
            st.pyplot(summary_fig, use_container_width=True)
            plt.close(summary_fig)  # Close to free memory
        
        # Detailed Findings
        if total_anomalies > 0:
            st.subheader("🔍 Detailed Anomaly Analysis")
            
            # Priority columns (sorted by severity)
            priority_columns = []
            for col, data in anomaly_by_column.items():
                if data['count'] > 0:
                    priority_columns.append((col, data))
            
            priority_columns.sort(key=lambda x: x[1]['percentage'], reverse=True)
            
            if priority_columns:
                st.write("**Columns requiring attention (ordered by severity):**")
                
                for i, (column, data) in enumerate(priority_columns[:5], 1):  # Show top 5
                    # Determine priority level
                    if data['percentage'] > 10:
                        priority_icon = "🚨"
                        priority_level = "CRITICAL"
                    elif data['percentage'] > 5:
                        priority_icon = "⚠️"
                        priority_level = "HIGH"
                    elif data['percentage'] > 1:
                        priority_icon = "🟡"
                        priority_level = "MEDIUM"
                    else:
                        priority_icon = "🔵"
                        priority_level = "LOW"
                    
                    st.write(f"{i}. {priority_icon} **{column}** - {data['count']} anomalies ({data['percentage']:.2f}%) - Priority: {priority_level}")
            
            # Method-specific insights
            st.subheader("📊 Detection Method Results")
            
            method_summary = {}
            for column, column_results in results.items():
                for method, anomaly_result in column_results.items():
                    if hasattr(anomaly_result, 'total_anomalies'):
                        if method not in method_summary:
                            method_summary[method] = {'columns': 0, 'total_anomalies': 0}
                        if anomaly_result.total_anomalies > 0:
                            method_summary[method]['columns'] += 1
                            method_summary[method]['total_anomalies'] += anomaly_result.total_anomalies
            
            if method_summary:
                for method, stats in method_summary.items():
                    method_name = method.replace('_', ' ').title()
                    if stats['total_anomalies'] > 0:
                        st.write(f"**{method_name}**: Found {stats['total_anomalies']} anomalies across {stats['columns']} columns")
                    else:
                        st.write(f"**{method_name}**: No anomalies detected")
            
            # Actionable recommendations
            st.subheader("🎯 Recommended Actions")
            
            recommendations = []
            
            if severity_analysis["Critical"] > 0:
                recommendations.append("🚨 **Immediate Action**: Investigate critical anomalies - they may indicate data corruption or system errors")
            
            if severity_analysis["High"] > 0:
                recommendations.append("⚠️ **High Priority**: Review high-severity anomalies to determine if they represent valid edge cases or errors")
            
            high_anomaly_columns = [col for col, data in anomaly_by_column.items() if data['percentage'] > 5]
            if high_anomaly_columns:
                recommendations.append(f"🔍 **Data Validation**: Columns with >5% anomalies need validation: {', '.join(high_anomaly_columns[:3])}")
            
            if total_anomaly_rate > 2:
                recommendations.append("📊 **Process Review**: High overall anomaly rate suggests reviewing data collection or processing procedures")
            
            if recommendations:
                for rec in recommendations:
                    st.write(rec)
            else:
                st.success("✅ **No immediate action required** - Anomaly levels are within normal ranges")
            
            # Detailed breakdown for top problematic columns
            critical_columns = [col for col, data in anomaly_by_column.items() if data['percentage'] > 5]
            if critical_columns:
                st.subheader("🔬 Critical Column Analysis")
                
                for column in critical_columns[:3]:  # Show top 3 critical columns
                    with st.expander(f"🔍 **{column}** - Detailed Analysis"):
                        column_data = anomaly_by_column[column]
                        st.write(f"**Anomaly Rate**: {column_data['percentage']:.2f}% ({column_data['count']} out of {len(df)} records)")
                        
                        # Show results by method for this column
                        column_results = results[column]
                        best_method_result = None
                        best_method_name = None
                        max_anomalies = 0
                        
                        for method, anomaly_result in column_results.items():
                            if hasattr(anomaly_result, 'total_anomalies') and anomaly_result.total_anomalies > 0:
                                st.write(f"**{method.upper()}**: {anomaly_result.total_anomalies} anomalies")
                                
                                # Track the method with most anomalies for visualization
                                if anomaly_result.total_anomalies > max_anomalies:
                                    max_anomalies = anomaly_result.total_anomalies
                                    best_method_result = anomaly_result
                                    best_method_name = method
                                
                                # Show sample anomalous values with context
                                if hasattr(anomaly_result, 'anomaly_values') and anomaly_result.anomaly_values:
                                    sample_values = anomaly_result.anomaly_values[:3]
                                    st.write(f"Sample anomalous values: {sample_values}")
                                    
                                    # Show typical values for comparison
                                    normal_mask = ~df.index.isin(anomaly_result.anomaly_indices)
                                    if normal_mask.any():
                                        normal_sample = df.loc[normal_mask, column].dropna().head(3).tolist()
                                        st.write(f"Typical values for comparison: {normal_sample}")
                        
                        # Add detailed visualization for this column
                        if best_method_result:
                            st.write(f"**📊 Detailed Visualization (using {best_method_name.upper()} method)**")
                            
                            try:
                                col_fig = create_anomaly_visualizations(df, column, best_method_result)
                                if col_fig:
                                    st.pyplot(col_fig, use_container_width=True)
                                    plt.close(col_fig)  # Close to free memory
                                else:
                                    st.info("Unable to create visualization for this column.")
                            except Exception as viz_error:
                                st.warning(f"Could not create visualization: {str(viz_error)}")
        
        else:
            st.success("🎉 **Excellent News!** No statistical anomalies detected in your numeric data.")
            st.info("This suggests your data has consistent patterns and no obvious outliers or data quality issues.")
            
            # Show what was analyzed
            analyzed_columns = list(results.keys())
            if analyzed_columns:
                st.write(f"**Columns analyzed**: {', '.join(analyzed_columns)}")
                st.write("**Methods used**: Z-score, IQR (Interquartile Range), Modified Z-score")


def ai_recommendations_tab(use_llm: bool, api_key: str, model: str):
    """AI recommendations tab"""
    st.header("🤖 AI Recommendations")
    
    if not use_llm:
        st.info("💡 Enable AI Recommendations in the sidebar to get intelligent suggestions")
        return
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Source tab")
        return
    
    if not st.session_state.profiling_complete:
        st.warning("⚠️ Please run data profiling first")
        return
    
    # Recommendations controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Generate AI-powered recommendations based on your data quality analysis")
    
    with col2:
        if st.button("🤖 Get Recommendations", type="primary"):
            generate_recommendations(api_key, model)
    
    # Show recommendations if available
    if st.session_state.recommendations:
        display_recommendations(st.session_state.recommendations)

def generate_recommendations(api_key: str, model: str):
    """Generate AI recommendations"""
    try:
        with st.spinner("🤖 Generating AI recommendations..."):
            if not api_key:
                # Use mock recommendations if no API key
                mock_recommendations = {
                    "recommendations": [
                        {
                            "type": "data_quality",
                            "priority": "high",
                            "title": "Address Missing Values",
                            "description": "Several columns have significant missing values that could impact analysis quality.",
                            "suggested_actions": [
                                "Consider imputation strategies for numerical columns",
                                "Investigate the root cause of missing data",
                                "Document data collection processes"
                            ]
                        },
                        {
                            "type": "data_validation",
                            "priority": "medium", 
                            "title": "Standardize Data Types",
                            "description": "Some columns may benefit from consistent data type formatting.",
                            "suggested_actions": [
                                "Convert string numbers to numeric types",
                                "Standardize date formats",
                                "Review categorical variable encoding"
                            ]
                        }
                    ],
                    "summary": "Your dataset shows good overall quality with some areas for improvement in completeness and consistency."
                }
                st.session_state.recommendations = mock_recommendations
            else:
                # Use actual LLM API
                config = LLMConfig(
                    provider="openai",
                    model=model,
                    api_key=api_key
                )
                
                analyzer = DataQualityLLMAnalyzer(config)
                recommendations = analyzer.analyze_data_quality(
                    st.session_state.data,
                    st.session_state.quality_report
                )
                st.session_state.recommendations = recommendations
        
        st.success("✅ AI recommendations generated successfully!")
        
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {str(e)}")

def display_recommendations(recommendations: Dict[str, Any]):
    """Display AI recommendations"""
    st.subheader("💡 AI-Powered Recommendations")
    
    # Summary
    if 'summary' in recommendations:
        st.info(f"📋 **Summary:** {recommendations['summary']}")
    
    # Individual recommendations
    if 'recommendations' in recommendations:
        for i, rec in enumerate(recommendations['recommendations'], 1):
            priority = rec.get('priority', 'medium')
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "🟡")
            
            with st.expander(f"{priority_color} {rec.get('title', f'Recommendation {i}')}"):
                st.write(f"**Priority:** {priority.title()}")
                st.write(f"**Type:** {rec.get('type', 'General').replace('_', ' ').title()}")
                
                if 'description' in rec:
                    st.write(f"**Description:** {rec['description']}")
                
                if 'suggested_actions' in rec and rec['suggested_actions']:
                    st.write("**Suggested Actions:**")
                    for action in rec['suggested_actions']:
                        st.write(f"• {action}")

if __name__ == "__main__":
    main()
