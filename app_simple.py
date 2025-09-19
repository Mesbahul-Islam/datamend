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
import sys
import os
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
        st.write(f"Dataset: {len(df):,} rows × {len(df.columns)} columns")
    
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
            
            # Run profiling
            report = engine.profile_dataset(df, "Dataset")
            st.session_state.quality_report = report
            st.session_state.profiling_complete = True
            
        st.success("✅ Data profiling completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error during profiling: {str(e)}")

def display_profiling_results(report: DataQualityReport):
    """Display profiling results"""
    
    # Overall summary
    st.subheader("📈 Quality Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        quality_score = report.overall_quality_score
        if quality_score >= 80:
            st.metric("Quality Score", f"{quality_score:.1f}/100", delta="✅ Good")
        elif quality_score >= 60:
            st.metric("Quality Score", f"{quality_score:.1f}/100", delta="⚠️ Fair")
        else:
            st.metric("Quality Score", f"{quality_score:.1f}/100", delta="❌ Poor")
    
    with col2:
        total_issues = len(report.critical_issues)
        st.metric("Total Issues", total_issues)
    
    with col3:
        st.metric("Critical Issues", len(report.critical_issues))
    
    with col4:
        st.metric("Warning Issues", 0)  # Add warning issues if available in report
    
    # Column-wise results
    st.subheader("📋 Column Analysis")
    
    if report.column_profiles:
        columns_data = []
        for col_name, col_profile in report.column_profiles.items():
            columns_data.append({
                'Column': col_name,
                'Data Type': col_profile.data_type,
                'Completeness %': f"{((report.total_rows - col_profile.null_count) / report.total_rows * 100):.1f}%",
                'Unique Values': col_profile.unique_count,
                'Null Count': col_profile.null_count,
                'Outliers': col_profile.outliers_count,
                'Quality Score': f"{((report.total_rows - col_profile.null_count) / report.total_rows * 100):.1f}"
            })
        
        columns_df = pd.DataFrame(columns_data)
        st.dataframe(columns_df, use_container_width=True)
        
        # Visualization
        st.subheader("📊 Quality Visualization")
        
        # Quality scores by column
        fig_quality = px.bar(
            columns_df, 
            x='Column', 
            y='Quality Score',
            title="Quality Score by Column",
            color='Quality Score',
            color_continuous_scale='RdYlGn'
        )
        fig_quality.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_quality, use_container_width=True)
        
        # Null values visualization
        if any(col_profile.null_count > 0 for col_profile in report.column_profiles.values()):
            null_data = [(col, col_profile.null_count) for col, col_profile in report.column_profiles.items()]
            null_data = [item for item in null_data if item[1] > 0]
            
            if null_data:
                null_df = pd.DataFrame(null_data, columns=['Column', 'Null Count'])
                fig_nulls = px.bar(
                    null_df,
                    x='Column',
                    y='Null Count',
                    title="Missing Values by Column",
                    color='Null Count',
                    color_continuous_scale='Reds'
                )
                fig_nulls.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_nulls, use_container_width=True)

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
            run_anomaly_detection(df, detection_method, anomaly_threshold)

def run_anomaly_detection(df: pd.DataFrame, method: str, threshold: float):
    """Run anomaly detection"""
    try:
        with st.spinner("🔍 Detecting anomalies..."):
            detector = StatisticalAnomalyDetector(z_threshold=threshold)
            
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
            
            display_anomaly_results(results, df)
            
    except Exception as e:
        st.error(f"❌ Error during anomaly detection: {str(e)}")

def display_anomaly_results(results: Dict[str, Dict[str, Any]], df: pd.DataFrame):
    """Display anomaly detection results"""
    st.subheader("🎯 Anomaly Detection Results")
    
    if not results:
        st.info("No numeric columns found for anomaly detection")
        return
    
    # Calculate total anomalies across all columns and methods
    total_anomalies = 0
    anomaly_by_column = {}
    
    for column, column_results in results.items():
        column_anomalies = set()
        for method, anomaly_result in column_results.items():
            if hasattr(anomaly_result, 'anomaly_indices'):
                column_anomalies.update(anomaly_result.anomaly_indices)
        anomaly_by_column[column] = len(column_anomalies)
        total_anomalies += len(column_anomalies)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Anomalies", total_anomalies)
    with col2:
        anomaly_rate = (total_anomalies / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    with col3:
        affected_columns = len([col for col, count in anomaly_by_column.items() if count > 0])
        st.metric("Affected Columns", affected_columns)
    
    # Anomalies by column
    if total_anomalies > 0:
        anomaly_data = []
        for col, count in anomaly_by_column.items():
            if count > 0:
                anomaly_data.append({
                    'Column': col,
                    'Anomaly Count': count,
                    'Anomaly Rate %': f"{(count / len(df)) * 100:.2f}%"
                })
        
        if anomaly_data:
            anomaly_df = pd.DataFrame(anomaly_data)
            st.dataframe(anomaly_df, width='stretch')
            
            # Show detailed results by method for each column
            st.subheader("📊 Detailed Results by Method")
            
            for column, column_results in results.items():
                if anomaly_by_column[column] > 0:
                    with st.expander(f"🔍 {column} - Anomaly Details"):
                        for method, anomaly_result in column_results.items():
                            if hasattr(anomaly_result, 'total_anomalies') and anomaly_result.total_anomalies > 0:
                                st.write(f"**{method.upper()} Method:**")
                                st.write(f"- Anomalies found: {anomaly_result.total_anomalies}")
                                st.write(f"- Threshold: {anomaly_result.threshold:.3f}")
                                st.write(f"- Percentage: {anomaly_result.anomaly_percentage:.2f}%")
                                
                                # Show sample anomalous values
                                if anomaly_result.anomaly_values:
                                    sample_values = anomaly_result.anomaly_values[:5]  # Show first 5
                                    st.write(f"- Sample values: {sample_values}")
                                st.write("---")
    else:
        st.success("✅ No anomalies detected in the dataset!")


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
