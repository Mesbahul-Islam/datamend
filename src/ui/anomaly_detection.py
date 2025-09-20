"""
Anomaly Detection Module
Handles anomaly detection exclusively through ydata-profiling
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
from src.ui.data_profiling import display_ai_recommendations_section
from src.ui.data_profiling import display_ai_recommendations_section


def anomaly_detection_tab(anomaly_threshold: float):
    """Statistical outlier analysis results - powered by ydata-profiling"""
    st.header("📊 Statistical Outlier Analysis")
    
    # Clear explanation of what this tab does
    st.info("🎯 **Purpose:** View and analyze statistical outliers detected in your data using ydata-profiling's IQR method. "
            "This helps identify unusual data points that may indicate errors or rare events.")
    
    # Link to data profiling
    st.warning("⚠️ **Prerequisites:** You must first run profiling in the **Data Profiling** tab, then use the "
               "**'🎯 Outlier Detection'** sub-tab to generate results before viewing them here.")
    
    if st.session_state.data is None:
        st.warning("Please load data first using the sidebar")
        return
    
    # Show current dataset info
    if st.session_state.current_dataset:
        st.info(f"Analyzing Dataset: {st.session_state.current_dataset}")
    
    # Check if ydata-profiling outlier detection was run
    if st.session_state.get('ydata_anomaly_detection', False):
        st.success("🟢 Statistical Outlier Results Available!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 View Outlier Results", type="primary", width='stretch'):
                if st.session_state.get('ydata_anomaly_results'):
                    display_ydata_anomaly_results(st.session_state.ydata_anomaly_results, st.session_state.data)
                else:
                    st.warning("No outlier results found. Please run outlier detection from the Data Profiling tab.")
        
        with col2:
            if st.button("📈 View Comprehensive Analysis", type="secondary", width='stretch'):
                if st.session_state.get('ydata_comprehensive_analysis'):
                    display_comprehensive_analysis_summary(st.session_state.ydata_comprehensive_analysis)
                else:
                    st.info("Run 'Outlier Summary' from the Data Profiling → Outlier Detection tab for comprehensive analysis.")
        
        st.markdown("---")


def display_ydata_anomaly_results(outliers_info, df: pd.DataFrame):
    """Display ydata-profiling statistical outlier results with visualizations"""
    st.subheader("📊 Statistical Outlier Analysis Results")
    
    if not outliers_info:
        st.success("🟢 Excellent! No statistical outliers detected by ydata-profiling.")
        return
    
    # Summary metrics
    total_outliers = sum(info['count'] for info in outliers_info.values())
    outlier_rate = (total_outliers / len(df)) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Outliers", f"{total_outliers:,}")
    with col2:
        st.metric("Affected Columns", len(outliers_info))
    with col3:
        st.metric("Outlier Rate", f"{outlier_rate:.2f}%")
    with col4:
        severity = "🔴 High" if outlier_rate > 5 else "🟡 Medium" if outlier_rate > 1 else "🟢 Low"
        st.metric("Severity Level", severity)
    
    # Create visualizations for each column with anomalies
    for column, info in outliers_info.items():
        st.subheader(f"📈 Anomaly Visualization: {column}")
        
        try:
            fig = create_ydata_anomaly_visualization(df, column, info)
            if fig:
                st.pyplot(fig, width='stretch')
                plt.close(fig)  # Free memory
        except Exception as e:
            st.warning(f"Could not create visualization for {column}: {str(e)}")
        
        # Show detailed stats
        with st.expander(f"📊 Detailed Statistics for {column}"):
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.write("**Anomaly Details:**")
                st.write(f"• Count: {info['count']} anomalies")
                st.write(f"• Rate: {(info['count']/len(df)*100):.2f}%")
                st.write(f"• Method: {info['method']}")
                
            with detail_col2:
                stats = info['statistics']
                st.write("**Statistical Context:**")
                st.write(f"• Mean: {stats['mean']:.3f}")
                st.write(f"• Std Dev: {stats['std']:.3f}")
                st.write(f"• IQR: {stats['IQR']:.3f}")
    
    # AI Recommendations section after outlier analysis
    st.markdown("---")
    display_ai_recommendations_section("outliers")


def create_ydata_anomaly_visualization(df: pd.DataFrame, column: str, anomaly_info: dict):
    """Create visualization for anomalies detected by ydata-profiling"""
    plt.style.use('default')
    
    # Get data for the column
    column_data = df[column].dropna()
    if len(column_data) == 0:
        return None
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'ydata-profiling Anomaly Analysis: {column}', fontsize=16, fontweight='bold')
    
    # Plot 1: Scatter plot with anomalies highlighted
    normal_mask = ~df[column].index.isin(anomaly_info['indices'])
    normal_data = df.loc[normal_mask, column].dropna()
    anomaly_data = df.loc[anomaly_info['indices'], column]
    
    ax1.scatter(normal_data.index, normal_data.values, 
                alpha=0.6, c='blue', label='Normal Data', s=30)
    if len(anomaly_data) > 0:
        ax1.scatter(anomaly_data.index, anomaly_data.values, 
                    alpha=0.9, c='red', label='Anomalies (ydata-profiling)', s=80, marker='X')
    
    # Add boundary lines
    ax1.axhline(y=anomaly_info['lower_bound'], color='red', linestyle='--', alpha=0.7, label='Lower Bound')
    ax1.axhline(y=anomaly_info['upper_bound'], color='red', linestyle='--', alpha=0.7, label='Upper Bound')
    
    ax1.set_title('Data Points with Anomalies Highlighted (IQR Method)', fontweight='bold')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel(f'{column} Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics box
    stats_text = f'Anomalies: {len(anomaly_info["indices"])}/{len(df)} ({(len(anomaly_info["indices"])/len(df)*100):.1f}%)\n'
    stats_text += f'Normal Range: [{anomaly_info["lower_bound"]:.3f}, {anomaly_info["upper_bound"]:.3f}]'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
             verticalalignment='top', fontweight='bold')
    
    # Plot 2: Box plot showing distribution and outliers
    box_data = [column_data.values]
    bp = ax2.boxplot(box_data, labels=[column], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    
    # Highlight anomalies on box plot
    if len(anomaly_info['values']) > 0:
        ax2.scatter([1] * len(anomaly_info['values']), anomaly_info['values'], 
                    color='red', s=60, marker='X', label='ydata-profiling Anomalies', zorder=5)
    
    ax2.set_title('Box Plot with ydata-profiling Anomalies', fontweight='bold')
    ax2.set_ylabel(f'{column} Value')
    ax2.grid(True, alpha=0.3)
    if len(anomaly_info['values']) > 0:
        ax2.legend()
    
    # Add method information
    method_text = f'Detection Method: {anomaly_info["method"]}\nIQR Method: Q1 - 1.5×IQR ≤ Normal ≤ Q3 + 1.5×IQR'
    ax2.text(0.02, 0.02, method_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
             verticalalignment='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def display_comprehensive_analysis_summary(analysis_data):
    """Display summary of comprehensive anomaly analysis"""
    st.subheader("📈 Comprehensive Anomaly Analysis Summary")
    
    outliers_info = analysis_data['outliers_info']
    total_anomalies = analysis_data['total_anomalies']
    anomaly_rate = analysis_data['anomaly_rate']
    assessment = analysis_data['assessment']
    priority_data = analysis_data['priority_data']
    
    # Executive summary
    st.info(f"**Overall Assessment:** {assessment}")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Anomalies", f"{total_anomalies:,}")
    with col2:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    with col3:
        reliability = max(0, 100 - (anomaly_rate * 3))
        st.metric("Data Reliability", f"{reliability:.0f}/100")
    
    # Priority summary table
    st.subheader("📋 Column Priority Summary")
    priority_df = pd.DataFrame(priority_data)
    st.dataframe(priority_df, width='stretch', hide_index=True)
    
    # Action recommendations
    st.subheader("💡 Key Recommendations")
    
    critical_count = len([p for p in priority_data if "Critical" in p["Priority"]])
    high_count = len([p for p in priority_data if "High" in p["Priority"]])
    
    if critical_count > 0:
        st.error(f"🚨 **Critical:** {critical_count} columns need immediate attention")
    
    if high_count > 0:
        st.warning(f"⚠️ **High Priority:** {high_count} columns need review")
    
    if critical_count == 0 and high_count == 0:
        st.success("✅ **Good News:** No critical or high-priority anomalies detected")
    
    st.info("💡 **Tip:** All anomalies detected using ydata-profiling's robust IQR method")
    
    # AI Recommendations section after comprehensive analysis
    st.markdown("---")
    display_ai_recommendations_section("outliers")
