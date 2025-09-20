"""
Anomaly Detection Module
Handles anomaly detection exclusively through ydata-profiling
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any


def anomaly_detection_tab(anomaly_threshold: float):
    """Anomaly detection tab - powered by ydata-profiling"""
    st.header("Anomaly Detection")
    st.info("Powered by ydata-profiling - All anomaly detection is performed using ydata-profiling's robust statistical methods.")
    
    if st.session_state.data is None:
        st.warning("Please load data first using the sidebar")
        return
    
    # Show current dataset info
    if st.session_state.current_dataset:
        st.info(f"🎯 **Analyzing Dataset**: {st.session_state.current_dataset}")
    
    # Check if ydata-profiling anomaly detection was run
    if st.session_state.get('ydata_anomaly_detection', False):
        st.success("✅ **ydata-profiling Anomaly Results Available!** ")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 View Anomaly Results", type="primary", use_container_width=True):
                if st.session_state.get('ydata_anomaly_results'):
                    display_ydata_anomaly_results(st.session_state.ydata_anomaly_results, st.session_state.data)
                else:
                    st.warning("No anomaly results found. Please run anomaly detection from the Data Profiling tab.")
        
        with col2:
            if st.button("📈 View Comprehensive Analysis", type="secondary", use_container_width=True):
                if st.session_state.get('ydata_comprehensive_analysis'):
                    display_comprehensive_analysis_summary(st.session_state.ydata_comprehensive_analysis)
                else:
                    st.info("Run 'Anomaly Summary' from the Data Profiling tab for comprehensive analysis.")
        
        st.markdown("---")
    
    # Guide user to data profiling for anomaly detection
    st.subheader("🚀 Getting Started with Anomaly Detection")
    
    if not st.session_state.get('ydata_profile'):
        st.warning("⚠️ **Step 1:** Run data profiling first to enable anomaly detection")
        st.info("Go to the **Data Profiling** tab and click '🔄 Run Profiling' to generate a comprehensive data profile.")
    else:
        st.success("✅ **Step 1 Complete:** Data profiling is available")
        st.info("**Step 2:** Go to the **Data Profiling** tab and use the anomaly detection buttons:")
        st.write("• **🔍 Extract Anomalies** - Get outliers detected by ydata-profiling")
        st.write("• **📊 Anomaly Summary** - Get comprehensive anomaly analysis")
    
    # Show what ydata-profiling offers for anomaly detection
    st.subheader("🔬 ydata-profiling Anomaly Detection Features")
    
    features = [
        "📊 **IQR (Interquartile Range) Method** - Industry-standard outlier detection",
        "📈 **Statistical Boundaries** - Automatic calculation of normal data ranges", 
        "🎯 **Per-Column Analysis** - Individual analysis for each numeric column",
        "📋 **Comprehensive Statistics** - Mean, std dev, quartiles, and more",
        "⚡ **Fast Processing** - Optimized algorithms for large datasets",
        "🔍 **Detailed Reporting** - Complete anomaly context and explanations"
    ]
    
    for feature in features:
        st.write(feature)
    
    # Technical information
    with st.expander("🔧 Technical Details"):
        st.write("**ydata-profiling Anomaly Detection Method:**")
        st.write("• Uses the **IQR (Interquartile Range)** method")
        st.write("• Calculates Q1 (25th percentile) and Q3 (75th percentile)")
        st.write("• Defines outliers as values outside: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]")
        st.write("• Same method used by popular statistical software and libraries")
        st.write("• Robust against various data distributions")
        st.write("• Automatically handles missing values and data type detection")


def display_ydata_anomaly_results(outliers_info, df: pd.DataFrame):
    """Display ydata-profiling anomaly results with visualizations"""
    st.subheader("📊 ydata-profiling Anomaly Analysis Results")
    
    if not outliers_info:
        st.success("🎉 Excellent! No anomalies detected by ydata-profiling.")
        return
    
    # Summary metrics
    total_anomalies = sum(info['count'] for info in outliers_info.values())
    anomaly_rate = (total_anomalies / len(df)) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Anomalies", f"{total_anomalies:,}")
    with col2:
        st.metric("Affected Columns", len(outliers_info))
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    with col4:
        severity = "🔴 High" if anomaly_rate > 5 else "🟡 Medium" if anomaly_rate > 1 else "🟢 Low"
        st.metric("Severity Level", severity)
    
    # Create visualizations for each column with anomalies
    for column, info in outliers_info.items():
        st.subheader(f"📈 Anomaly Visualization: {column}")
        
        try:
            fig = create_ydata_anomaly_visualization(df, column, info)
            if fig:
                st.pyplot(fig, use_container_width=True)
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
    st.dataframe(priority_df, use_container_width=True, hide_index=True)
    
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
