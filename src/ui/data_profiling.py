"""
Data Profiling Module
Handles ydata-profiling integration and report display functionality
"""

import streamlit as st
import pandas as pd
import time
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any
from ydata_profiling import ProfileReport


@dataclass
class QualityIssue:
    """Data quality issue for downstream analytics"""
    title: str
    severity: str  # 'critical', 'high', 'medium'
    description: str
    affected_columns: List[str]
    recommendation: str


def data_profiling_tab(anomaly_threshold: float):
    """Data profiling tab"""
    st.header("Data Profiling")
    
    if st.session_state.data is None:
        st.warning("Please load data first using the sidebar")
        return
    
    # Show current dataset info
    if st.session_state.current_dataset:
        st.info(f"Analyzing Dataset: {st.session_state.current_dataset}")
    
    df = st.session_state.data
    
    # Profiling controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Cache expensive dataset size calculation
        @st.cache_data
        def get_dataset_info(dataset_hash):
            """Get dataset size information with caching"""
            dataset_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            return dataset_size_mb
        
        # Create a simple hash for the current dataset
        dataset_hash = f"{len(df)}_{len(df.columns)}_{id(df)}"
        dataset_size_mb = get_dataset_info(dataset_hash)

        # Determine processing strategy message
        if len(df) > 100000:
            processing_strategy = "Large dataset - using parallel processing"
        else:
            processing_strategy = "Standard dataset - using optimized sequential processing"
        
        st.write(f"**Dataset:** {len(df):,} rows × {len(df.columns)} columns ({dataset_size_mb:.1f} MB)")
    
    with col2:
        if st.button("Run Profiling", type="primary", key="run_profiling_button"):
            run_data_profiling(df, anomaly_threshold)
    
    # Show results if available
    if st.session_state.get('ydata_profile'):
        display_ydata_profiling_results(st.session_state.ydata_profile, anomaly_threshold)


def run_data_profiling(df: pd.DataFrame, anomaly_threshold: float):
    """Run data profiling on the dataset using ydata-profiling"""
    try:
        with st.spinner("Running comprehensive data profiling with ydata-profiling..."):
            # Run profiling with timing
            start_time = time.time()
            
            # Create profile report with ydata-profiling
            profile = ProfileReport(
                df, 
                title=f"Data Profile Report - {st.session_state.current_dataset or 'Dataset'}",
                explorative=True,
                minimal=True
            )
            
            end_time = time.time()
            profiling_time = end_time - start_time
                
            st.session_state.ydata_profile = profile
            st.session_state.profiling_time = profiling_time
            st.session_state.profiling_complete = True
            
        
    except Exception as e:
        st.error(f"Error during profiling: {str(e)}")
        st.error(f"Details: {type(e).__name__}: {str(e)}")


def display_ydata_profiling_results(profile, anomaly_threshold: float):
    """Display ydata-profiling results with interactive HTML report and summary"""
    
    # Executive Summary from ydata-profiling
    st.subheader("Data Profiling Report")

    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Quick Summary", "Detailed Report", "Anomaly Detection", "Analytics Quality"])
    
    with tab1:
        display_ydata_summary(profile)
    
    with tab2:
        display_enhanced_report(profile)
        # AI Insights for Detailed Report
        st.markdown("---")
        display_ai_recommendations_section("profiling_detailed")
    
    with tab3:
        display_anomaly_detection_section(profile, anomaly_threshold)
        # AI Insights for Anomaly Detection
        st.markdown("---")
        display_ai_recommendations_section("profiling_outliers")
    
    with tab4:
        display_quality_analysis_section(st.session_state.data)
        # AI Analytics Suitability Decision
        st.markdown("---")
        display_analytics_suitability_ai_section()
    
    # Export options section
    st.markdown("---")
    display_export_options(profile)


def display_ydata_summary(profile):
    """Display a quick summary of the ydata-profiling results"""
    st.subheader("Quick Data Summary")
    
    # Get basic statistics from the profile
    description = profile.description_set
    table_stats = description.table
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_vars = table_stats.get('n_var', 0)
        st.metric("Variables", n_vars)
    
    with col2:
        n_obs = table_stats.get('n', 0)
        st.metric("Observations", f"{n_obs:,}")
    
    with col3:
        missing_cells = table_stats.get('n_cells_missing', 0)
        # Calculate total cells properly: rows * columns
        n_obs = table_stats.get('n', 0)
        n_vars = table_stats.get('n_var', 0)
        total_cells = n_obs * n_vars if n_obs > 0 and n_vars > 0 else 1
        missing_percent = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        # Alternative: use the percentage directly from ydata-profiling
        # missing_percent = table_stats.get('p_cells_missing', 0) * 100
        st.metric("Missing Cells", f"{missing_percent:.1f}%")
    
    with col4:
        duplicate_rows = table_stats.get('n_duplicates', 0)
        duplicate_percent = (duplicate_rows / n_obs) * 100 if n_obs > 0 else 0
        st.metric("Duplicate Rows", f"{duplicate_percent:.1f}%")
    
    # Variable types
    st.subheader("Variable Types")
    
    types_summary = table_stats.get('types', {})
    if types_summary:
        types_df = pd.DataFrame([
            {"Type": type_name.replace('_', ' ').title(), "Count": count}
            for type_name, count in types_summary.items()
        ])
        st.dataframe(types_df, width='stretch', hide_index=True)
    
    #Warnings and alerts
    st.subheader("Data Quality Alerts")
    
    alerts = []
    
    if missing_percent > 10:
        alerts.append(f"High missing data: {missing_percent:.1f}% of cells are missing")
    
    if duplicate_percent > 5:
        alerts.append(f" Duplicate rows detected: {duplicate_percent:.1f}% of rows are duplicates")
    
    n_constant = table_stats.get('n_constant', 0)
    if n_constant > 0:
        alerts.append(f" Constant variables: {n_constant} variables have only one unique value")
    
    if not alerts:
        st.success("No major data quality issues detected!")
    else:
        for alert in alerts:
            st.warning(alert)


def display_full_html_report(profile):
    """Display the full HTML report from ydata-profiling"""
    st.subheader("Full Profiling Report")
    
    try:
        # Generate the HTML report
        html_report = profile.to_html()
        
        # Display in Streamlit using components
        st.components.v1.html(html_report, height=800, scrolling=True)
        
    except Exception as e:
        st.error(f"Error generating HTML report: {str(e)}")
        st.info("Try using the 'Show Quick Summary' option instead")


def download_html_report(profile):
    """Provide download link for the HTML report"""
    st.subheader("Download Report")
    
    try:
        # Generate the HTML report
        html_report = profile.to_html()
        
        # Create download button
        dataset_name = st.session_state.current_dataset or "dataset"
        filename = f"data_profile_report_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        st.download_button(
            label=" Download HTML Report",
            data=html_report,
            file_name=filename,
            mime="text/html",
            help="Download the complete profiling report as an HTML file"
        )
        
        st.success("Report ready for download! Click the button above to save the HTML file.")
        
    except Exception as e:
        st.error(f"Error preparing download: {str(e)}")


def display_enhanced_report(profile):
    """Display an enhanced report with better formatting"""

    try:
        # Get description for detailed analysis
        description = profile.description_set
        table_stats = description.table
        variables = description.variables
        
        # Dataset Overview Section
        st.markdown("### Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Variables", table_stats.get('n_var', 0))
        with col2:
            st.metric("Total Observations", f"{table_stats.get('n', 0):,}")
        with col3:
            missing_cells = table_stats.get('n_cells_missing', 0)
            # Calculate total cells properly: rows * columns
            n_obs = table_stats.get('n', 0)
            n_vars = table_stats.get('n_var', 0)
            total_cells = n_obs * n_vars if n_obs > 0 and n_vars > 0 else 1
            missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        with col4:
            duplicates = table_stats.get('n_duplicates', 0)
            duplicate_pct = (duplicates / table_stats.get('n', 1)) * 100 if table_stats.get('n', 0) > 0 else 0
            st.metric("Duplicate Rows", f"{duplicate_pct:.1f}%")
        
        # Variable Types Analysis
        st.markdown("### Variable Types Distribution")
        types_summary = table_stats.get('types', {})
        if types_summary:
            types_df = pd.DataFrame([
                {"Variable Type": type_name.replace('_', ' ').title(), "Count": count, "Percentage": f"{(count/sum(types_summary.values()))*100:.1f}%"}
                for type_name, count in types_summary.items()
            ])
            st.dataframe(types_df, width='stretch', hide_index=True)
        
        # Variable Details Analysis
        st.markdown("### Variable Analysis")
        
        if variables:
            # Create detailed variable analysis
            variable_details = []
            
            for var_name, var_info in variables.items():
                var_type = var_info.get('type', 'Unknown')
                n_missing = var_info.get('n_missing', 0)
                n_total = var_info.get('n', 0)
                missing_pct = (n_missing / n_total * 100) if n_total > 0 else 0
                n_unique = var_info.get('n_unique', 0)
                
                # Determine status
                if missing_pct > 50:
                    status = "Critical"
                elif missing_pct > 20:
                    status = "Warning"
                elif missing_pct > 5:
                    status = "Attention"
                else:
                    status = "Good"
                
                # Get key statistics based on type
                key_stats = ""
                if var_type == "Numeric":
                    mean_val = var_info.get('mean', 0)
                    std_val = var_info.get('std', 0)
                    key_stats = f"Mean: {mean_val:.2f}, Std: {std_val:.2f}" if mean_val and std_val else "Basic stats available"
                elif var_type == "Categorical":
                    n_categories = var_info.get('n_distinct', 0)
                    key_stats = f"{n_categories} categories"
                elif var_type == "Text":
                    max_length = var_info.get('max_length', 0)
                    key_stats = f"Max length: {max_length}" if max_length else "Text analysis available"
                
                variable_details.append({
                    "Variable": var_name,
                    "Type": var_type,
                    "Status": status,
                    "Missing %": f"{missing_pct:.1f}%",
                    "Unique Values": f"{n_unique:,}",
                    "Key Statistics": key_stats
                })
            
            # Display variable details table
            variables_df = pd.DataFrame(variable_details)
            st.dataframe(variables_df, width='stretch', hide_index=True)
        
        # Data Quality Issues
        st.markdown("### Data Quality Assessment")
        
        quality_issues = []
        
        # Check for high missing data
        high_missing_vars = [name for name, info in variables.items() 
                           if (info.get('n_missing', 0) / info.get('n', 1)) > 0.1]
        if high_missing_vars:
            quality_issues.append(f"High missing data in {len(high_missing_vars)} variables: {', '.join(high_missing_vars[:3])}{'...' if len(high_missing_vars) > 3 else ''}")
        
        # Check for duplicate rows
        if duplicate_pct > 5:
            quality_issues.append(f" Duplicate rows detected: {duplicate_pct:.1f}% of the dataset")
        
        # Check for constant variables
        constant_vars = [name for name, info in variables.items() 
                        if info.get('n_distinct', 0) <= 1]
        if constant_vars:
            quality_issues.append(f" Constant variables found: {', '.join(constant_vars)}")
        
        if quality_issues:
            for issue in quality_issues:
                st.warning(issue)
        else:
            st.success(" No major data quality issues detected!")
        
    except Exception as e:
        st.error(f"Error displaying enhanced report: {str(e)}")
        st.info("Please try the Quick Summary view or download the report for offline viewing.")


def display_anomaly_detection_section(profile, anomaly_threshold: float):
    """Display anomaly detection section using ydata-profiling"""
    st.markdown("### Statistical Anomaly Detection")
    st.info(f"Uses configurable IQR (Interquartile Range) method with threshold {anomaly_threshold:.1f} to detect statistical outliers in numeric columns.")
    
    # Technical explanation
    with st.expander("How it works"):
        st.write("**Configurable IQR Method:**")
        st.write("• Calculates Q1 (25th percentile) and Q3 (75th percentile)")
        st.write(f"• Defines outliers as values outside: [Q1 - {anomaly_threshold:.1f}×IQR, Q3 + {anomaly_threshold:.1f}×IQR]")
        st.write("• Industry-standard method with adjustable sensitivity")
        st.write("• Lower thresholds (e.g., 1.0) detect more outliers, higher thresholds (e.g., 3.0) are more conservative")
        st.write("• Robust against various data distributions")
    
    if st.button("Extract Anomalies", type="primary", use_container_width=True, help="Extract anomalies detected by ydata-profiling"):
        display_profiling_outliers(profile, st.session_state.data, anomaly_threshold)
    
    if st.button("Anomaly Summary", type="secondary", use_container_width=True, help="Show comprehensive anomaly analysis from profiling"):
        # Clear the page and show full analysis
        st.empty()
        display_full_anomaly_analysis_page(profile, st.session_state.data, anomaly_threshold)


def display_export_options(profile):
    """Provide various export and sharing options"""
    st.subheader("Export & Share Options")
    
    try:
        # Generate the HTML report
        html_report = profile.to_html()
        
        # Create download button
        dataset_name = st.session_state.current_dataset or "dataset"
        filename = f"data_profile_report_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.download_button(
                label="Download Complete HTML Report",
                data=html_report,
                file_name=filename,
                mime="text/html",
                help="Download the complete profiling report as an HTML file",
                width='stretch'
            )
        
        with col2:
            file_size_mb = len(html_report.encode('utf-8')) / (1024 * 1024)
            st.metric("Report Size", f"{file_size_mb:.1f} MB")
        
        st.success("Report ready for download! The HTML file contains:")
        
        features = [
            "Interactive data exploration",
            "Comprehensive statistical analysis", 
            "Distribution plots and histograms",
            "Correlation matrices",
            "Data quality warnings",
            "Missing data patterns",
            "Anomaly detection results",
            "Mobile-responsive design"
        ]
        
        col1, col2 = st.columns(2)
        for i, feature in enumerate(features):
            if i % 2 == 0:
                col1.write(f"• {feature}")
            else:
                col2.write(f"• {feature}")
        
        st.info("**Tip**: Open the downloaded HTML file in your browser for the best viewing experience with full interactivity.")
        
    except Exception as e:
        st.error(f"Error preparing export options: {str(e)}")


def extract_profiling_outliers(profile, df: pd.DataFrame, threshold: float = 1.5):
    """Extract outlier information from ydata-profiling report"""
    try:
        outliers_info = {}
        description = profile.description_set
        variables = description.variables
        
        for var_name, var_info in variables.items():
            if var_info.get('type') in ['Numeric', 'Real'] and var_name in df.columns:
                # Extract outlier information using ydata-profiling's approach
                outliers = []
                
                # Get percentiles and statistics from profiling
                column_data = df[var_name].dropna()
                if len(column_data) == 0:
                    continue
                
                # Use configurable IQR method for anomaly detection
                Q1 = column_data.quantile(0.25)
                Q3 = column_data.quantile(0.75)
                IQR = Q3 - Q1
                
                # Configurable IQR anomaly detection using threshold parameter
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (df[var_name] < lower_bound) | (df[var_name] > upper_bound)
                outlier_indices = df[outlier_mask].index.tolist()
                outlier_values = df.loc[outlier_indices, var_name].tolist()
                
                if len(outlier_indices) > 0:
                    # Extract additional statistics from profiling
                    mean_val = var_info.get('mean', column_data.mean())
                    std_val = var_info.get('std', column_data.std())
                    min_val = var_info.get('min', column_data.min())
                    max_val = var_info.get('max', column_data.max())
                    
                    outliers_info[var_name] = {
                        'count': len(outlier_indices),
                        'indices': outlier_indices,
                        'values': outlier_values,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'method': f'Configurable IQR (threshold={threshold})',
                        'statistics': {
                            'mean': mean_val,
                            'std': std_val,
                            'min': min_val,
                            'max': max_val,
                            'Q1': Q1,
                            'Q3': Q3,
                            'IQR': IQR
                        }
                    }
        
        return outliers_info
    except Exception as e:
        st.error(f"Error extracting outliers from ydata-profiling: {str(e)}")
        return {}


def display_profiling_outliers(profile, df: pd.DataFrame, threshold: float = 1.5):
    """Display outliers detected by ydata-profiling"""
    st.subheader(f"Anomaly Detection (IQR Threshold: {threshold:.1f})")
    
    with st.spinner("Extracting outlier information from ydata-profiling report..."):
        outliers_info = extract_profiling_outliers(profile, df, threshold)
    
    if not outliers_info:
        st.success("No outliers detected by ydata-profiling analysis!")
        st.info("This suggests your numeric data follows expected statistical distributions with no significant anomalies.")
        return
    
    # Summary metrics
    total_outliers = sum(info['count'] for info in outliers_info.values())
    affected_columns = len(outliers_info)
    outlier_rate = (total_outliers / len(df)) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Anomalies", total_outliers)
    with col2:
        st.metric("Affected Columns", f"{affected_columns}/{len([col for col in df.columns if df[col].dtype in ['int64', 'float64']])}")
    with col3:
        st.metric("Anomaly Rate", f"{outlier_rate:.2f}%")
    with col4:
        severity = "High" if outlier_rate > 5 else "Medium" if outlier_rate > 1 else "Low"
        st.metric("Severity", severity)
    
    # Detailed analysis
    st.subheader(" Column-wise Anomaly Analysis")
    
    for column, info in outliers_info.items():
        anomaly_pct = (info['count']/len(df)*100)
        priority_icon = "" if anomaly_pct > 10 else "" if anomaly_pct > 5 else "" if anomaly_pct > 1 else ""
        
        with st.expander(f"{priority_icon} **{column}** - {info['count']} anomalies ({anomaly_pct:.2f}%)", expanded=False):
            # Use full width for better visibility
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("** Statistical Boundaries (IQR Method):**")
                st.write(f"•Lower bound: {info['lower_bound']:.3f}")
                st.write(f"• Upper bound: {info['upper_bound']:.3f}")
                st.write(f"• Detection method: {info['method']}")
                
                st.write("** Column Statistics:**")
                stats = info['statistics']
                st.write(f"• Mean: {stats['mean']:.3f}")
                st.write(f"• Std Dev: {stats['std']:.3f}")
                st.write(f"• Range: {stats['min']:.3f} to {stats['max']:.3f}")
            
            with col2:
                st.write("** Sample Anomalous Values:**")
                sample_values = info['values'][:8]  # Show first 8
                for i, value in enumerate(sample_values, 1):
                    deviation = "above" if value > info['upper_bound'] else "below"
                    st.write(f"{i}. {value:.3f} ({deviation} normal range)")
                if len(info['values']) > 8:
                    st.write(f"... and {len(info['values']) - 8} more anomalies")
            
            # Full width sections for better visibility
            st.markdown("---")
            
            # Show outlier locations in full width
            if st.checkbox(f"Show anomaly row indices for {column}", key=f"show_indices_{column}"):
                st.write("** Row indices with anomalies:**")
                indices_text = ", ".join(map(str, info['indices'][:50]))  # Show more indices
                if len(info['indices']) > 50:
                    indices_text += f" ... and {len(info['indices']) - 50} more"
                st.code(indices_text)  # Use code block for better formatting
            
            # Statistical analysis in full width
            if st.checkbox(f"Show detailed statistics for {column}", key=f"show_stats_{column}"):
                st.write("** Detailed Statistics:**")
                stats = info['statistics']
                
                # Create 3 columns for better distribution
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                with detail_col1:
                    st.write(f"• **Q1 (25th percentile):** {stats['Q1']:.3f}")
                    st.write(f"• **Q3 (75th percentile):** {stats['Q3']:.3f}")
                with detail_col2:
                    st.write(f"• **IQR (Q3-Q1):** {stats['IQR']:.3f}")
                    st.write(f"• **Data points:** {len(df[column].dropna())}")
                with detail_col3:
                    st.write(f"• **Minimum value:** {stats['min']:.3f}")
                    st.write(f"• **Maximum value:** {stats['max']:.3f}")
    
    # Store results for anomaly detection tab
    st.session_state.ydata_anomaly_results = outliers_info
    st.session_state.ydata_anomaly_detection = True


def display_full_anomaly_analysis_page(profile, df: pd.DataFrame, threshold: float = 1.5):
    """Display a full-page anomaly analysis combining both detailed and comprehensive views"""
    # Clear existing content and create a full-page layout
    st.markdown(f"# Complete Anomaly Detection Analysis (IQR Threshold: {threshold:.1f})")
    st.markdown("### Using configurable IQR statistical methods")
    
    with st.spinner("Extracting comprehensive outlier information from ydata-profiling report..."):
        outliers_info = extract_profiling_outliers(profile, df, threshold)
    
    if not outliers_info:
        st.success("No outliers detected by ydata-profiling analysis!")
        st.info("This suggests your numeric data follows expected statistical distributions with no significant anomalies.")
        
        # Show what was analyzed
        numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        st.write(f"**Analysis completed on {len(numeric_cols)} numeric columns:** {', '.join(numeric_cols[:5])}")
        if len(numeric_cols) > 5:
            st.write(f"... and {len(numeric_cols) - 5} more columns")
        return
    
    # Executive Summary Section
    st.markdown("---")
    st.markdown("##  Executive Summary")
    
    total_anomalies = sum(info['count'] for info in outliers_info.values())
    total_numeric_cols = len([col for col in df.columns if df[col].dtype in ['int64', 'float64']])
    anomaly_rate = (total_anomalies / len(df)) * 100
    
    # Assessment
    if anomaly_rate < 1:
        assessment = "**Excellent** - Very few anomalies detected"
        message = "Your data quality is excellent with minimal outliers."
    elif anomaly_rate < 5:
        assessment = "**Good** - Some anomalies present" 
        message = "Your data has some outliers that may need investigation."
    elif anomaly_rate < 10:
        assessment = "**Attention Needed** - Moderate anomalies"
        message = "Several anomalies detected that require attention."
    else:
        assessment = "**Critical** -High anomaly rate"
        message = "High number of anomalies detected - data quality review recommended."
    
    st.info(f"**Data Quality Assessment:** {assessment}")
    st.write(f"**Summary:** {message}")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Anomalies", f"{total_anomalies:,}")
    with col2:
        st.metric("Affected Columns", f"{len(outliers_info)}/{total_numeric_cols}")
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    with col4:
        reliability_score = max(0, 100 - (anomaly_rate * 3))
        st.metric("Data Reliability", f"{reliability_score:.0f}/100")
    
    # Priority Analysis Section
    st.markdown("---")
    st.markdown("##  Priority Anomaly Analysis")
    
    priority_data = []
    for column, info in outliers_info.items():
        col_anomaly_rate = (info['count'] / len(df)) * 100
        
        if col_anomaly_rate > 10:
            priority = "Critical"
        elif col_anomaly_rate > 5:
            priority = "High"
        elif col_anomaly_rate > 1:
            priority = "Medium"
        else:
            priority = "Low"
        
        priority_data.append({
            "Column": column,
            "Anomalies": info['count'],
            "Rate (%)": f"{col_anomaly_rate:.2f}%",
            "Priority": priority,
            "Action": get_action_recommendation(col_anomaly_rate)
        })
    
    # Sort by anomaly rate (highest first)
    priority_data.sort(key=lambda x: float(x["Rate (%)"].rstrip('%')), reverse=True)
    
    priority_df = pd.DataFrame(priority_data)
    st.dataframe(priority_df, width='stretch', hide_index=True)
    
    # Actionable Recommendations Section
    st.markdown("---")
    st.markdown("##  Recommended Actions")
    
    recommendations = []
    
    critical_cols = [item for item in priority_data if "Critical" in item["Priority"]]
    if critical_cols:
        recommendations.append("**Immediate Action Required**:Critical anomaly levels detected - investigate data collection processes")
    
    high_cols = [item for item in priority_data if "High" in item["Priority"]]
    if high_cols:
        recommendations.append("**High Priority**: Review anomalies to determine if they represent valid edge cases or errors")
    
    if anomaly_rate > 5:
        recommendations.append("**Process Review**:High overall anomaly rate suggests reviewing data validation procedures")
    
    if len(outliers_info) > len([col for col in df.columns if df[col].dtype in ['int64', 'float64']]) * 0.5:
        recommendations.append("**Data Quality Audit**: More than half of numeric columns have anomalies")
    
    if recommendations:
        for rec in recommendations:
            st.warning(rec)
    else:
        st.success("**No immediate action required** - Anomaly levels are within acceptable ranges")
    
    # Detailed Column Analysis Section
    st.markdown("---")
    st.markdown("##  Detailed Column-wise Analysis")
    
    for column, info in outliers_info.items():
        anomaly_pct = (info['count']/len(df)*100)
        priority_icon = "" if anomaly_pct > 10 else "" if anomaly_pct > 5 else "" if anomaly_pct > 1 else ""
        
        with st.expander(f"{priority_icon} **{column}** - {info['count']} anomalies ({anomaly_pct:.2f}%)", expanded=anomaly_pct > 5):
            # Summary metrics in full width
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Total Anomalies", f"{info['count']}")
            with metric_col2:
                st.metric("Anomaly Rate", f"{anomaly_pct:.2f}%")
            with metric_col3:
                st.metric("IQR Value", f"{info['statistics']['IQR']:.3f}")
            with metric_col4:
                priority_text = "Critical" if anomaly_pct > 10 else "High" if anomaly_pct > 5 else "Medium"
                st.metric("Priority", priority_text)
            
            st.markdown("---")
            
            # Detailed analysis in organized columns
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.write("** Statistical Context:**")
                stats = info['statistics']
                st.write(f"• **Mean:** {stats['mean']:.3f}")
                st.write(f"• **Standard Deviation:** {stats['std']:.3f}")
                st.write(f"• **Q1 (25th percentile):** {stats['Q1']:.3f}")
                st.write(f"• **Q3 (75th percentile):** {stats['Q3']:.3f}")
                st.write(f"• **Data Range:** {stats['min']:.3f} to {stats['max']:.3f}")
                
                st.write("** Anomaly Boundaries:**")
                st.write(f"• **Normal Range:** {info['lower_bound']:.3f} to {info['upper_bound']:.3f}")
                st.write(f"• **Detection Method:** {info['method']}")
                st.write(f"• **Total Data Points:** {len(df[column].dropna())}")
            
            with detail_col2:
                st.write("** Sample Anomalous Values:**")
                sample_values = info['values'][:8]  # Show first 8
                for i, value in enumerate(sample_values, 1):
                    deviation = "above" if value > info['upper_bound'] else "below"
                    st.write(f"{i}. **{value:.3f}** ({deviation} normal range)")
                if len(info['values']) > 8:
                    st.write(f"... and {len(info['values']) - 8} more anomalies")
                
                # Show outlier locations
                st.write("** Sample Row Indices with Anomalies:**")
                indices_text = ", ".join(map(str, info['indices'][:20]))  # Show first 20 indices
                if len(info['indices']) > 20:
                    indices_text += f" ... and {len(info['indices']) - 20} more"
                st.code(indices_text)
            
            # Full width action recommendation
            st.markdown("---")
            st.write("** Recommended Action:**")
            action = get_action_recommendation(anomaly_pct)
            if anomaly_pct > 10:
                st.error(f" {action}")
            elif anomaly_pct > 5:
                st.warning(f" {action}")
            else:
                st.info(f" {action}")
    
    # Store comprehensive results
    st.session_state.ydata_comprehensive_analysis = {
        'outliers_info': outliers_info,
        'total_anomalies': total_anomalies,
        'anomaly_rate': anomaly_rate,
        'assessment': assessment,
        'priority_data': priority_data
    }
    st.session_state.ydata_anomaly_results = outliers_info
    st.session_state.ydata_anomaly_detection = True


def display_comprehensive_anomaly_analysis(profile, df: pd.DataFrame, threshold: float = 1.5):
    """Display comprehensive anomaly analysis from ydata-profiling"""
    st.subheader(" Comprehensive Anomaly Analysis")
    
    with st.spinner("Analyzing anomalies from ydata-profiling report..."):
        outliers_info = extract_profiling_outliers(profile, df, threshold)
        description = profile.description_set
        variables = description.variables
    
    if not outliers_info:
        st.success("Excellent! No anomalies detected in your dataset.")
        st.info("Your data appears to follow expected statistical patterns with no significant outliers.")
        
        # Show what was analyzed
        numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        st.write(f"**Analysis completed on {len(numeric_cols)} numeric columns:** {', '.join(numeric_cols[:5])}")
        if len(numeric_cols) > 5:
            st.write(f"... and {len(numeric_cols) - 5} more columns")
        return
    
    # Executive Summary
    total_anomalies = sum(info['count'] for info in outliers_info.values())
    total_numeric_cols = len([col for col in df.columns if df[col].dtype in ['int64', 'float64']])
    anomaly_rate = (total_anomalies / len(df)) * 100
    
    # Assessment
    if anomaly_rate < 1:
        assessment = "**Excellent** - Very few anomalies detected"
        message = "Your data quality is excellent with minimal outliers."
    elif anomaly_rate < 5:
        assessment = "**Good** - Some anomalies present" 
        message = "Your data has some outliers that may need investigation."
    elif anomaly_rate < 10:
        assessment = "**Attention Needed** - Moderate anomalies"
        message = "Several anomalies detected that require attention."
    else:
        assessment = "**Critical** -High anomaly rate"
        message = "High number of anomalies detected - data quality review recommended."
    
    st.info(f"**Data Quality Assessment:** {assessment}")
    st.write(f"**Summary:** {message}")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Anomalies", f"{total_anomalies:,}")
    with col2:
        st.metric("Affected Columns", f"{len(outliers_info)}/{total_numeric_cols}")
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    with col4:
        reliability_score = max(0, 100 - (anomaly_rate * 3))
        st.metric("Data Reliability", f"{reliability_score:.0f}/100")
    
    # Priority Analysis
    st.subheader(" Priority Anomaly Columns")
    
    priority_data = []
    for column, info in outliers_info.items():
        col_anomaly_rate = (info['count'] / len(df)) * 100
        
        if col_anomaly_rate > 10:
            priority = "Critical"
        elif col_anomaly_rate > 5:
            priority = "High"
        elif col_anomaly_rate > 1:
            priority = "Medium"
        else:
            priority = "Low"
        
        priority_data.append({
            "Column": column,
            "Anomalies": info['count'],
            "Rate (%)": f"{col_anomaly_rate:.2f}%",
            "Priority": priority,
            "Action": get_action_recommendation(col_anomaly_rate)
        })
    
    # Sort by anomaly rate (highest first)
    priority_data.sort(key=lambda x: float(x["Rate (%)"].rstrip('%')), reverse=True)
    
    priority_df = pd.DataFrame(priority_data)
    st.dataframe(priority_df, width='stretch', hide_index=True)
    
    # Actionable Recommendations
    st.subheader(" Recommended Actions")
    
    recommendations = []
    
    critical_cols = [item for item in priority_data if "Critical" in item["Priority"]]
    if critical_cols:
        recommendations.append("**Immediate Action Required**:Critical anomaly levels detected - investigate data collection processes")
    
    high_cols = [item for item in priority_data if "High" in item["Priority"]]
    if high_cols:
        recommendations.append("**High Priority**: Review anomalies to determine if they represent valid edge cases or errors")
    
    if anomaly_rate > 5:
        recommendations.append("**Process Review**:High overall anomaly rate suggests reviewing data validation procedures")
    
    if len(outliers_info) > len([col for col in df.columns if df[col].dtype in ['int64', 'float64']]) * 0.5:
        recommendations.append("**Data Quality Audit**: More than half of numeric columns have anomalies")
    
    if recommendations:
        for rec in recommendations:
            st.warning(rec)
    else:
        st.success("**No immediate action required** - Anomaly levels are within acceptable ranges")
    
    # Column Details
    st.subheader(" Detailed Column Analysis")
    
    for i, (column, info) in enumerate(outliers_info.items()):
        if i < 5:  # Show details for top 5 problematic columns
            col_anomaly_rate = (info['count'] / len(df)) * 100
            priority_icon = "" if col_anomaly_rate > 10 else "" if col_anomaly_rate > 5 else ""
            
            with st.expander(f"{priority_icon} **{column}** - Detailed Analysis", expanded=False):
                # Use full width for better visibility
                st.write(f"** Anomaly Overview for {column}**")
                
                # Summary metrics in full width
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Total Anomalies", f"{info['count']}")
                with metric_col2:
                    st.metric("Anomaly Rate", f"{col_anomaly_rate:.2f}%")
                with metric_col3:
                    st.metric("IQR Value", f"{info['statistics']['IQR']:.3f}")
                with metric_col4:
                    priority_text = "Critical" if col_anomaly_rate > 10 else "High" if col_anomaly_rate > 5 else "Medium"
                    st.metric("Priority", priority_text)
                
                st.markdown("---")
                
                # Detailed analysis in organized columns
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.write("** Statistical Context:**")
                    stats = info['statistics']
                    st.write(f"• **Mean:** {stats['mean']:.3f}")
                    st.write(f"• **Standard Deviation:** {stats['std']:.3f}")
                    st.write(f"• **Q1 (25th percentile):** {stats['Q1']:.3f}")
                    st.write(f"• **Q3 (75th percentile):** {stats['Q3']:.3f}")
                    st.write(f"• **Data Range:** {stats['min']:.3f} to {stats['max']:.3f}")
                
                with detail_col2:
                    st.write("** Anomaly Boundaries:**")
                    st.write(f"• **Normal Range:** {info['lower_bound']:.3f} to {info['upper_bound']:.3f}")
                    st.write(f"• **Detection Method:** {info['method']}")
                    st.write(f"• **Total Data Points:** {len(df[column].dropna())}")
                    
                    st.write("** Sample Anomalous Values:**")
                    sample_values = info['values'][:5]  # Show more samples
                    for j, value in enumerate(sample_values, 1):
                        deviation = "above" if value > info['upper_bound'] else "below"
                        st.write(f"  {j}. **{value:.3f}** ({deviation} normal range)")
                    if len(info['values']) > 5:
                        st.write(f"  ... and {len(info['values']) - 5} more anomalies")
                
                # Full width action recommendation
                st.markdown("---")
                st.write("** Recommended Action:**")
                action = get_action_recommendation(col_anomaly_rate)
                if col_anomaly_rate > 10:
                    st.error(f" {action}")
                elif col_anomaly_rate > 5:
                    st.warning(f" {action}")
                else:
                    st.info(f" {action}")
    
    if len(outliers_info) > 5:
        st.info(f" Showing top 5 problematic columns. Total columns with anomalies: {len(outliers_info)}")
    
    # Store comprehensive results
    st.session_state.ydata_comprehensive_analysis = {
        'outliers_info': outliers_info,
        'total_anomalies': total_anomalies,
        'anomaly_rate': anomaly_rate,
        'assessment': assessment,
        'priority_data': priority_data
    }


def get_action_recommendation(anomaly_rate: float) -> str:
    """Get action recommendation based on anomaly rate"""
    if anomaly_rate > 10:
        return "Immediate investigation required"
    elif anomaly_rate > 5:
        return "Review and validate anomalies"
    elif anomaly_rate > 1:
        return "Monitor and document patterns"
    else:
        return "Normal monitoring"


def display_quality_analysis_section(df: pd.DataFrame):
    """Display quality analysis section with button to check for downstream analytics issues"""
    st.markdown("### Data Quality for Analytics")
    
    # Clear explanation of the difference
    st.info(" **Focus:** Identifies data quality issues that could break or bias downstream analytics "
            "(machine learning, SQL queries, statistical analysis) - different from statistical anomaly detection.")
    
    # Comparison with anomaly detection
    with st.expander(" What's the difference from anomaly detection?"):
        st.write("**Statistical Anomaly Detection** (Previous tab):")
        st.write("• Finds unusual values in individual data points")
        st.write("• Uses statistical methods (IQR, standard deviation)")
        st.write("• Helps identify data entry errors or rare events")
        st.write("")
        st.write("**Analytics Quality Analysis** (This tab):")
        st.write("• Finds structural issues that break analytics workflows")
        st.write("• Checks for empty columns, mixed data types, high missing data")
        st.write("• Prevents analytics failures before they happen")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("**Do you want to check for potential anomalies affecting downstream analytics?**")
        st.caption("This might take a while for large datasets as it performs comprehensive statistical analysis.")
    
    with col2:
        if st.button("Run Analytics Quality Check", type="secondary", key="run_quality_check"):
            run_quality_analysis(df)
    
    # Show results if available
    if st.session_state.get('quality_analysis_results'):
        display_quality_analysis_results(st.session_state.quality_analysis_results)


def run_quality_analysis(df: pd.DataFrame):
    """Run quality analysis to find issues affecting downstream analytics"""
    try:
        with st.spinner("Analyzing data quality for downstream analytics..."):
            issues = []
            
            # 1. Empty columns (breaks SQL)
            empty_cols = df.columns[df.isnull().all()].tolist()
            if empty_cols:
                issues.append(QualityIssue(
                    title="Empty Columns Detected",
                    severity="critical",
                    description=f"{len(empty_cols)} completely empty columns will break SQL queries and analytics",
                    affected_columns=empty_cols,
                    recommendation="Remove empty columns before running analytics"
                ))
            
            # 2.High missing data (>80% missing)
            high_missing = []
            for col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df)
                if missing_pct > 0.8:
                    high_missing.append(col)
            
            if high_missing:
                issues.append(QualityIssue(
                    title="Extreme Missing Data",
                    severity="critical", 
                    description=f"{len(high_missing)} columns with >80% missing data will cause analytics to fail",
                    affected_columns=high_missing,
                    recommendation="Remove or impute missing data before analytics"
                ))
            
            # 3. Duplicate records (affects statistical analysis)
            duplicates = df.duplicated().sum()
            if duplicates > len(df) * 0.1:  # >10% duplicates
                issues.append(QualityIssue(
                    title="High Duplicate Rate",
                    severity="high",
                    description=f"{duplicates} duplicate records ({duplicates/len(df)*100:.1f}%) will skew analytics results",
                    affected_columns=list(df.columns),
                    recommendation="Remove duplicate records to ensure accurate analytics"
                ))
            
            # 4. Constant columns (no predictive value)
            constant_cols = []
            for col in df.columns:
                if df[col].nunique() == 1:
                    constant_cols.append(col)
            
            if constant_cols:
                issues.append(QualityIssue(
                    title="Constant Columns",
                    severity="medium",
                    description=f"{len(constant_cols)} columns with only one value provide no analytical value",
                    affected_columns=constant_cols,
                    recommendation="Remove constant columns to improve model performance"
                ))
            
            # 5.High cardinality categorical columns (memory issues)
            high_cardinality_cols = []
            for col in df.select_dtypes(include=['object']).columns:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.95 and df[col].nunique() > 100:
                    high_cardinality_cols.append(col)
            
            if high_cardinality_cols:
                issues.append(QualityIssue(
                    title="High Cardinality Categorical Columns",
                    severity="medium",
                    description=f"{len(high_cardinality_cols)} categorical columns with very high cardinality may cause memory issues",
                    affected_columns=high_cardinality_cols,
                    recommendation="Consider grouping rare categories or using embedding techniques"
                ))
            
            # 6. Outliers in numeric columns (affects ML models)
            outlier_cols = []
            for col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_ratio = len(outliers) / len(df)
                
                if outlier_ratio > 0.05:  # More than 5% outliers
                    outlier_cols.append(col)
            
            if outlier_cols:
                issues.append(QualityIssue(
                    title="High Outlier Rate",
                    severity="medium",
                    description=f"{len(outlier_cols)} numeric columns have >5% outliers which may affect ML models",
                    affected_columns=outlier_cols,
                    recommendation="Consider outlier treatment (removal, capping, or transformation)"
                ))
            
            # 7. Mixed data types in object columns
            mixed_type_cols = []
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    # Check if column can be converted to numeric
                    pd.to_numeric(df[col], errors='raise')
                except (ValueError, TypeError):
                    # Check for mixed types by looking at sample values
                    sample_values = df[col].dropna().head(100)
                    numeric_count = 0
                    for val in sample_values:
                        try:
                            float(val)
                            numeric_count += 1
                        except (ValueError, TypeError):
                            pass
                    
                    if 0.1 < numeric_count / len(sample_values) < 0.9:  # Mixed types
                        mixed_type_cols.append(col)
            
            if mixed_type_cols:
                issues.append(QualityIssue(
                    title="Mixed Data Types",
                    severity="high",
                    description=f"{len(mixed_type_cols)} columns contain mixed data types which will cause analytics errors",
                    affected_columns=mixed_type_cols,
                    recommendation="Clean and standardize data types before analytics"
                ))
            
            # Calculate overall quality score (more strict scoring)
            critical_count = len([i for i in issues if i.severity == "critical"])
            high_count = len([i for i in issues if i.severity == "high"])
            medium_count = len([i for i in issues if i.severity == "medium"])
            
            # More strict scoring: Critical issues have major impact, high issues are significant
            score = max(0, 100 - (critical_count * 50) - (high_count * 30) - (medium_count * 15))
            
            # Additional penalties for multiple issues
            total_issues = len(issues)
            if total_issues > 5:
                score = max(0, score - ((total_issues - 5) * 5))  # Extra penalty for many issues
            
            # Cap score at 85 if any critical issues exist (never "excellent" with critical issues)
            if critical_count > 0:
                score = min(score, 85)
            
            # Cap score at 90 if any high issues exist
            if high_count > 0:
                score = min(score, 90)
            
            results = {
                "quality_score": score,
                "issues": issues,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            st.session_state.quality_analysis_results = results
            
    except Exception as e:
        st.error(f"Error during quality analysis: {str(e)}")


def display_quality_analysis_results(results: Dict[str, Any]):
    """Display the results of quality analysis"""
    st.subheader(" Analytics Quality Report")
    
    quality_score = results.get("quality_score", 0)
    issues = results.get("issues", [])
    
    # Quality score display with more realistic thresholds
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if quality_score >= 90:
            st.metric("Analytics Quality Score", f"{quality_score}/100", delta="Excellent", delta_color="normal")
        elif quality_score >= 75:
            st.metric("Analytics Quality Score", f"{quality_score}/100", delta="Good", delta_color="normal")
        elif quality_score >= 60:
            st.metric("Analytics Quality Score", f"{quality_score}/100", delta="Fair", delta_color="off")
        elif quality_score >= 40:
            st.metric("Analytics Quality Score", f"{quality_score}/100", delta="Poor", delta_color="inverse")
        else:
            st.metric("Analytics Quality Score", f"{quality_score}/100", delta="Critical", delta_color="inverse")
    
    with col2:
        critical_count = len([i for i in issues if i.severity == "critical"])
        st.metric("Critical Issues", critical_count, delta_color="inverse" if critical_count > 0 else "normal")
    
    with col3:
        total_issues = len(issues)
        st.metric("Total Issues", total_issues)
    
    # More realistic overall assessment
    if quality_score >= 90:
        st.success("📊 Excellent data quality - Ready for advanced analytics with minimal preprocessing.")
    elif quality_score >= 75:
        st.info("✅ Good data quality - Suitable for most analytics with minor cleanup.")
    elif quality_score >= 60:
        st.warning("⚠️ Fair data quality - Requires moderate cleanup before reliable analytics.")
    elif quality_score >= 40:
        st.warning("🔧 Poor data quality - Significant preprocessing needed before analytics.")
    else:
        st.error("❌ Critical data quality issues - Major remediation required before any analytics.")
    
    # Always recommend AI analysis for deeper insights
    st.info("💡 **Recommendation**: Use the 'AI Analytics Suitability Assessment' below for deeper insights and specific recommendations based on these quality metrics.")
    
    # Issues breakdown
    if issues:
        st.subheader(" Issues RequiringAttention")
        
        # Group issues by severity
        critical_issues = [i for i in issues if i.severity == "critical"]
        high_issues = [i for i in issues if i.severity == "high"]
        medium_issues = [i for i in issues if i.severity == "medium"]
        
        # Display critical issues first
        if critical_issues:
            st.markdown("**Critical Issues (Must Fix)**")
            for issue in critical_issues:
                with st.expander(f" {issue.title}"):
                    st.write(f"**Impact:** {issue.description}")
                    st.write(f"**Affected Columns:** `{', '.join(issue.affected_columns)}`")
                    st.write(f"**Recommendation:** {issue.recommendation}")
        
        if high_issues:
            st.markdown("**High Priority Issues**")
            for issue in high_issues:
                with st.expander(f" {issue.title}"):
                    st.write(f"**Impact:** {issue.description}")
                    st.write(f"**Affected Columns:** `{', '.join(issue.affected_columns)}`")
                    st.write(f"**Recommendation:** {issue.recommendation}")
        
        if medium_issues:
            st.markdown("**Medium Priority Issues**")
            for issue in medium_issues:
                with st.expander(f" {issue.title}"):
                    st.write(f"**Impact:** {issue.description}")
                    st.write(f"**Affected Columns:** `{', '.join(issue.affected_columns)}`")
                    st.write(f"**Recommendation:** {issue.recommendation}")
    else:
        st.success("No quality issues detected! Your data is ready for analytics.")
    
    # Export quality report
    if st.button(" Export Quality Report", key="export_quality_report"):
        export_quality_report(results)


def export_quality_report(results: Dict[str, Any]):
    """Export quality analysis report"""
    try:
        issues = results.get("issues", [])
        
        # Create a DataFrame for export
        export_data = []
        for issue in issues:
            export_data.append({
                "Issue": issue.title,
                "Severity": issue.severity,
                "Description": issue.description,
                "Affected Columns": ", ".join(issue.affected_columns),
                "Recommendation": issue.recommendation
            })
        
        if export_data:
            df_export = pd.DataFrame(export_data)
            csv_data = df_export.to_csv(index=False)
            
            st.download_button(
                label="Download Quality Report CSV",
                data=csv_data,
                file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No issues found to export.")
    
    except Exception as e:
        st.error(f"Error exporting report: {str(e)}")


def display_ai_recommendations_section(context: str):
    """Display AI recommendations section with context-specific suggestions"""
    llm_config = st.session_state.get('llm_auto_config', {})
    
    # Only show if LLM configuration exists and has an API key
    if not llm_config or not llm_config.get('api_key'):
        st.warning("🤖 AI Recommendations available! Enable AI in the sidebar and add your API key to see intelligent insights here.")
        return
    
    if context == "profiling":
        st.subheader("🤖 AI Insights for Data Profiling")
        description = "Get AI-powered insights about your data quality, patterns, and potential issues based on the profiling results."
    elif context == "profiling_detailed":
        st.subheader("🤖 AI Insights for Detailed Report")
        description = "Get comprehensive AI analysis of your detailed data profiling report including patterns, relationships, and data quality insights."
    elif context == "profiling_outliers":
        st.subheader("🤖 AI Analysis of Anomaly Detection")
        description = "Get AI explanations about your anomaly detection results and recommendations for handling statistical anomalies."
    elif context == "outliers":
        st.subheader("🤖 AI Explanation of Outlier Analysis")
        description = "Get AI explanations about what the statistical outlier results mean for your data and what actions to take."
    elif context == "lineage":
        st.subheader("🤖 AI Analysis of Data Lineage")
        description = "Get AI insights about your data lineage, transformation patterns, and recommendations for data governance."
    else:
        st.subheader("🤖 AI Recommendations")
        description = "Get AI-powered recommendations based on your analysis."
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write(description)
        st.caption("💡 Powered by Google Gemini")
    
    with col2:
        button_key = f"get_ai_{context}_insights"
        if st.button("Get AI Insights", type="secondary", key=button_key):
            # Clear any existing recommendations for this context before generating new ones
            recommendations_key = f'ai_recommendations_{context}'
            if recommendations_key in st.session_state:
                del st.session_state[recommendations_key]
            generate_contextual_recommendations(context, llm_config)
    
    with col3:
        # Add a clear button to remove insights for this context
        clear_key = f"clear_ai_{context}_insights"
        recommendations_key = f'ai_recommendations_{context}'
        if st.session_state.get(recommendations_key):
            if st.button("Clear Insights", type="secondary", key=clear_key):
                if recommendations_key in st.session_state:
                    del st.session_state[recommendations_key]
                st.rerun()
    
    # Show recommendations if available for this context
    recommendations_key = f'ai_recommendations_{context}'
    if st.session_state.get(recommendations_key):
        display_contextual_recommendations(st.session_state[recommendations_key], context)


def generate_contextual_recommendations(context: str, llm_config: dict):
    """Generate context-specific AI recommendations"""
    try:
        from src.llm.analyzer import DataQualityLLMAnalyzer, LLMConfig
        
        with st.spinner(f"Generating AI insights for {context}..."):
            config = LLMConfig(
                provider=llm_config.get('provider', 'openai'),
                model=llm_config.get('model', 'gpt-3.5-turbo'),
                api_key=llm_config.get('api_key', '')
            )
            analyzer = DataQualityLLMAnalyzer(config)
            
            # Build context-specific prompts
            if context == "profiling":
                prompt = build_profiling_prompt()
            elif context == "profiling_detailed":
                prompt = build_detailed_profiling_prompt()
            elif context == "profiling_outliers":
                prompt = build_profiling_outliers_prompt()
            elif context == "outliers":
                prompt = build_outliers_prompt()
            elif context == "lineage":
                prompt = build_lineage_prompt()
            else:
                prompt = "Analyze the data and provide recommendations."
            
            # Call the LLM API and store raw response
            raw_response = analyzer._call_llm_api(prompt)
            
            # Store the raw response directly in session state with context-specific key
            recommendations_key = f'ai_recommendations_{context}'
            st.session_state[recommendations_key] = raw_response
            
            # Track which context was last generated to help with debugging
            st.session_state['last_ai_context'] = context
        
        st.success(f"🟢 AI insights for {context} generated successfully!")
        # Force a rerun to show the new insights immediately
        st.rerun()
        
    except Exception as e:
        st.error(f"🔴 Error generating AI insights: {str(e)}")
        if "api_key" in str(e).lower():
            if llm_config.get('provider') == "google gemini":
                st.info("💡 Tip: Get your Google AI API key from https://aistudio.google.com/app/apikey")
            else:
                st.info("💡 Tip: Check your OpenAI API key and ensure it has sufficient credits")


def build_profiling_prompt() -> str:
    """Build a prompt for data profiling insights"""
    if not st.session_state.get('ydata_profile'):
        return "No profiling data available."
    
    profile = st.session_state.ydata_profile
    description = profile.description_set
    table_stats = description.table
    
    # Extract key statistics
    n_vars = table_stats.get('n_var', 0)
    n_obs = table_stats.get('n', 0)
    missing_cells = table_stats.get('n_cells_missing', 0)
    total_cells = n_obs * n_vars if n_obs > 0 and n_vars > 0 else 1
    missing_percent = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    duplicate_rows = table_stats.get('n_duplicates', 0)
    duplicate_percent = (duplicate_rows / n_obs) * 100 if n_obs > 0 else 0
    
    prompt = f"""
    Analyze this data profiling summary and provide actionable insights:
    
    Dataset Overview:
    - Variables: {n_vars}
    - Observations: {n_obs:,}
    - Missing data: {missing_percent:.1f}% of cells
    - Duplicate rows: {duplicate_percent:.1f}%
    
    Variable types summary: {table_stats.get('types', {})}
    
    Please provide:
    1. Assessment of overall data quality
    2. Key concerns and risks for analytics
    3. Specific recommendations for data cleaning
    4. Prioritized action items
    
    Focus on practical, actionable advice for a data analyst.
    """
    
    return prompt


def build_outliers_prompt() -> str:
    """Build a prompt for outlier analysis insights"""
    if not st.session_state.get('ydata_anomaly_results'):
        return "No outlier analysis data available."
    
    outlier_data = st.session_state.ydata_anomaly_results
    total_outliers = sum(len(outliers) for outliers in outlier_data.values())
    
    # Count outliers per column
    column_outlier_counts = {col: len(outliers) for col, outliers in outlier_data.items()}
    
    prompt = f"""
    Analyze these statistical anomaly detection results and explain their significance:
    
    Outlier Summary:
    - Total outliers detected: {total_outliers}
    - Columns with outliers: {len(column_outlier_counts)}
    - Outliers per column: {column_outlier_counts}
    
    Please provide:
    1. Interpretation of what these outliers might indicate
    2. Whether the outlier rate seems normal or concerning
    3. Recommendations for handling these outliers
    4. Impact on downstream analytics and modeling
    5. Next steps for investigation
    
    Explain in terms a business analyst would understand.
    """
    
    return prompt


def build_lineage_prompt() -> str:
    """Build a prompt for data lineage insights"""
    if not st.session_state.get('lineage_data') or st.session_state.lineage_data.empty:
        return "No lineage data available."
    
    lineage_data = st.session_state.lineage_data
    
    # Extract key information
    query_types = lineage_data.get('QUERY_TYPE', pd.Series()).value_counts().to_dict() if 'QUERY_TYPE' in lineage_data.columns else {}
    unique_users = lineage_data.get('USER_NAME', pd.Series()).nunique() if 'USER_NAME' in lineage_data.columns else 0
    total_queries = len(lineage_data)
    
    prompt = f"""
    Analyze this data lineage information and provide governance insights:
    
    Lineage Summary:
    - Total queries analyzed: {total_queries}
    - Query types: {query_types}
    - Unique users: {unique_users}
    
    Please provide:
    1. Assessment of data transformation patterns
    2. Data governance considerations
    3. Risk assessment for data quality
    4. Recommendations for monitoring and controls
    5. Best practices for this data pipeline
    
    Focus on actionable data governance recommendations.
    """
    
    return prompt


def display_contextual_recommendations(raw_response: str, context: str):
    """Display the raw AI response in a clean, readable format"""
    if context == "profiling":
        icon = "📊"
        title = "Data Profiling Insights"
    elif context == "outliers":
        icon = "🎯"
        title = "Outlier Analysis Explanation"
    elif context == "lineage":
        icon = "🔗"
        title = "Data Lineage Analysis"
    else:
        icon = "💡"
        title = "AI Recommendations"
    
    st.subheader(f"{icon} {title}")
    
    # Display the response in a clean format
    if raw_response:
        cleaned_response = raw_response.strip()
        
        # Remove markdown code block markers if present
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        if cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]   # Remove ```
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]  # Remove trailing ```
        
        cleaned_response = cleaned_response.strip()
        
        # Try to parse as JSON first
        try:
            import json
            parsed = json.loads(cleaned_response)
            
            # Check if it's the structured data quality assessment format
            if "data_quality_assessment" in parsed:
                display_structured_assessment(parsed["data_quality_assessment"])
            elif "data_profiling_insights" in parsed:
                # Handle detailed data profiling insights format
                display_data_profiling_insights(parsed["data_profiling_insights"])
            elif "analysisSummary" in parsed and "treatmentStrategies" in parsed and "recommendations" in parsed:
                # Handle anomaly detection analysis format
                display_anomaly_analysis_results(parsed)
            elif "data_cleaning_recommendations" in parsed:
                # Handle case where the response has the recommendations at the top level
                display_structured_assessment(parsed)
            else:
                # Display as formatted JSON
                st.json(parsed)
                
        except json.JSONDecodeError as e:
            # If not JSON, display as markdown text
            st.markdown(cleaned_response)
    else:
        st.warning("No AI insights were generated. Please try again or check your API configuration.")


def display_structured_assessment(assessment: dict):
    """Display structured data quality assessment in a user-friendly format"""
    
    # Handle both nested and flat JSON structures
    # Check if this is the assessment object or the outer wrapper
    if "data_quality_assessment" in assessment:
        assessment_data = assessment["data_quality_assessment"]
        recommendations = assessment.get("data_cleaning_recommendations", [])
        action_items = assessment.get("prioritized_action_items", [])
    else:
        assessment_data = assessment
        recommendations = assessment.get("data_cleaning_recommendations", [])
        action_items = assessment.get("prioritized_action_items", [])
    
    # Overall Assessment
    overall_key = "overall_assessment" if "overall_assessment" in assessment_data else "overall_quality"
    if overall_key in assessment_data:
        st.info(f"**📋 Overall Assessment:** {assessment_data[overall_key]}")
    
    # Key Concerns and Risks
    if "key_concerns_and_risks" in assessment_data and assessment_data["key_concerns_and_risks"]:
        st.subheader("⚠️ Key Concerns and Risks")
        
        for i, concern in enumerate(assessment_data["key_concerns_and_risks"], 1):
            with st.expander(f"🔴 {concern.get('concern', f'Concern {i}')}"):
                st.write(f"**Risk:** {concern.get('risk', 'No risk description provided')}")
                st.write(f"**Mitigation:** {concern.get('mitigation', 'No mitigation strategy provided')}")
    
    # Data Cleaning Recommendations
    if recommendations:
        st.subheader("🧹 Data Cleaning Recommendations")
        
        for rec in recommendations:
            priority = rec.get('priority', 'Medium')
            priority_color = {
                'High': '🔴',
                'Medium': '🟡', 
                'Low': '🟢'
            }.get(priority, '🟡')
            
            variable = rec.get('variable', 'Not specified')
            issue = rec.get('issue', rec.get('action', 'Action'))
            
            with st.expander(f"{priority_color} {variable}: {issue} - {priority} Priority"):
                if 'recommendation' in rec:
                    st.write(f"**Recommendation:** {rec['recommendation']}")
                elif 'details' in rec:
                    st.write(f"**Details:** {rec['details']}")
    
    # Prioritized Action Items
    if action_items:
        st.subheader("📋 Prioritized Action Plan")
        
        for item in action_items:
            priority = item.get('priority', 0)
            action = item.get('action', 'No action specified')
            rationale = item.get('rationale', 'No rationale provided')
            
            st.write(f"**{priority}.** {action}")
            st.caption(f"💡 Rationale: {rationale}")
            st.markdown("---")


def display_analytics_suitability_ai_section():
    """Display AI section for analytics suitability assessment"""
    llm_config = st.session_state.get('llm_auto_config', {})
    
    # Only show if LLM configuration exists and has an API key
    if not llm_config or not llm_config.get('api_key'):
        st.warning("🤖 AI Analytics Suitability Assessment available! Enable AI in the sidebar and add your API key to get intelligent analytics readiness insights.")
        return
    
    st.subheader("🤖 AI Analytics Suitability Assessment")
    st.write("Get AI-powered assessment of whether your data is suitable for analytics and machine learning based on quality analysis results.")
    st.caption("💡 Powered by AI - Analyzes quality metrics to make data readiness decisions")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.info("This assessment requires quality analysis results. Run the Analytics Quality Check above first.")
    
    with col2:
        if st.button("Assess Analytics Readiness", type="primary", key="analytics_suitability_ai"):
            # Check if quality analysis has been run
            if not st.session_state.get('quality_analysis_results'):
                st.error("Please run the Analytics Quality Check first to get quality metrics for AI assessment.")
                return
            
            # Clear any existing assessment
            if 'ai_analytics_suitability' in st.session_state:
                del st.session_state['ai_analytics_suitability']
            
            generate_analytics_suitability_assessment(llm_config)
    
    with col3:
        # Add a clear button
        if st.session_state.get('ai_analytics_suitability'):
            if st.button("Clear Assessment", type="secondary", key="clear_analytics_suitability"):
                if 'ai_analytics_suitability' in st.session_state:
                    del st.session_state['ai_analytics_suitability']
                st.rerun()
    
    # Show suitability assessment if available
    if st.session_state.get('ai_analytics_suitability'):
        display_analytics_suitability_results(st.session_state['ai_analytics_suitability'])


def generate_analytics_suitability_assessment(llm_config: dict):
    """Generate AI assessment of data suitability for analytics"""
    try:
        from src.llm.analyzer import DataQualityLLMAnalyzer, LLMConfig
        
        with st.spinner("🔍 AI analyzing data quality metrics for analytics suitability..."):
            config = LLMConfig(
                provider=llm_config.get('provider', 'openai'),
                model=llm_config.get('model', 'gpt-3.5-turbo'),
                api_key=llm_config.get('api_key', '')
            )
            analyzer = DataQualityLLMAnalyzer(config)
            
            # Build analytics suitability prompt with quality analysis results
            prompt = build_analytics_suitability_prompt()
            
            # Call the LLM API
            raw_response = analyzer._call_llm_api(prompt)
            
            # Store the response
            st.session_state['ai_analytics_suitability'] = raw_response
            st.session_state['last_ai_context'] = 'analytics_suitability'
        
        st.success("🟢 Analytics suitability assessment completed!")
        st.rerun()
        
    except Exception as e:
        st.error(f"🔴 Error generating analytics suitability assessment: {str(e)}")
        if "api_key" in str(e).lower():
            if llm_config.get('provider') == "google gemini":
                st.info("💡 Tip: Get your Google AI API key from https://aistudio.google.com/app/apikey")
            else:
                st.info("💡 Tip: Check your OpenAI API key and ensure it has sufficient credits")


def build_analytics_suitability_prompt() -> str:
    """Build a prompt for analytics suitability assessment based on quality analysis"""
    quality_results = st.session_state.get('quality_analysis_results', {})
    
    if not quality_results:
        return "No quality analysis results available for assessment."
    
    quality_score = quality_results.get('quality_score', 0)
    issues = quality_results.get('issues', [])
    
    # Categorize issues by severity
    critical_issues = [i for i in issues if i.severity == "critical"]
    high_issues = [i for i in issues if i.severity == "high"]
    medium_issues = [i for i in issues if i.severity == "medium"]
    
    # Build detailed issue summary
    issues_summary = []
    for issue in issues:
        issues_summary.append(f"- {issue.title} ({issue.severity}): {issue.description}")
    
    issues_text = "\n".join(issues_summary) if issues_summary else "No significant issues detected"
    
    prompt = f"""
    As a data analytics expert, assess whether this dataset is suitable for analytics and machine learning based on the following quality analysis:

    QUALITY METRICS:
    - Overall Quality Score: {quality_score}/100
    - Total Issues Found: {len(issues)}
    - Critical Issues: {len(critical_issues)}
    - High Priority Issues: {len(high_issues)}
    - Medium Priority Issues: {len(medium_issues)}

    DETAILED ISSUES:
    {issues_text}

    Please provide a JSON response with the following structure:
    {{
        "analytics_suitability": {{
            "overall_verdict": "SUITABLE" or "NOT_SUITABLE" or "CONDITIONALLY_SUITABLE",
            "confidence_score": 85,
            "readiness_level": "Production Ready" or "Needs Minor Cleanup" or "Requires Major Cleanup" or "Not Ready",
            "key_blockers": ["List of main issues preventing analytics use"],
            "recommended_actions": ["Priority actions to improve analytics readiness"],
            "analytics_impact": {{
                "machine_learning": "Good/Fair/Poor - explanation",
                "statistical_analysis": "Good/Fair/Poor - explanation", 
                "sql_queries": "Good/Fair/Poor - explanation",
                "visualization": "Good/Fair/Poor - explanation"
            }},
            "timeline_estimate": "Immediate/1-3 days/1-2 weeks/1+ months",
            "executive_summary": "Brief summary of findings and recommendation"
        }}
    }}

    Base your assessment on:
    1. Critical issues that would break analytics workflows
    2. Data completeness and consistency
    3. Impact on different types of analytics
    4. Effort required to make data analytics-ready

    Be specific and actionable in your recommendations.
    """
    
    return prompt


def display_analytics_suitability_results(raw_response: str):
    """Display analytics suitability assessment results"""
    st.subheader("📊 Analytics Readiness Assessment")
    
    if not raw_response:
        st.warning("No assessment results available.")
        return
    
    try:
        import json
        
        # Clean and parse the response
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        parsed = json.loads(cleaned_response)
        
        if "analytics_suitability" in parsed:
            assessment = parsed["analytics_suitability"]
            
            # Overall Verdict
            verdict = assessment.get("overall_verdict", "UNKNOWN")
            confidence = assessment.get("confidence_score", 0)
            readiness = assessment.get("readiness_level", "Unknown")
            
            # Display verdict with appropriate styling
            if verdict == "SUITABLE":
                st.success(f"✅ **ANALYTICS READY** - {readiness}")
                st.metric("AI Confidence", f"{confidence}%", delta="High Confidence" if confidence > 80 else "Medium Confidence")
            elif verdict == "CONDITIONALLY_SUITABLE":
                st.warning(f"⚠️ **CONDITIONALLY SUITABLE** - {readiness}")
                st.metric("AI Confidence", f"{confidence}%", delta="Review Required")
            else:
                st.error(f"❌ **NOT SUITABLE** - {readiness}")
                st.metric("AI Confidence", f"{confidence}%", delta="Major Issues Found")
            
            # Executive Summary
            if assessment.get("executive_summary"):
                st.info(f"**📋 Executive Summary:** {assessment['executive_summary']}")
            
            # Timeline Estimate
            timeline = assessment.get("timeline_estimate", "Unknown")
            st.write(f"**⏱️ Estimated Time to Analytics-Ready:** {timeline}")
            
            # Analytics Impact Assessment
            if "analytics_impact" in assessment:
                st.subheader("📈 Impact on Different Analytics Types")
                impact = assessment["analytics_impact"]
                
                col1, col2 = st.columns(2)
                with col1:
                    if "machine_learning" in impact:
                        ml_status = impact["machine_learning"]
                        if "Good" in ml_status:
                            st.success(f"🤖 **Machine Learning:** {ml_status}")
                        elif "Fair" in ml_status:
                            st.warning(f"🤖 **Machine Learning:** {ml_status}")
                        else:
                            st.error(f"🤖 **Machine Learning:** {ml_status}")
                    
                    if "statistical_analysis" in impact:
                        stats_status = impact["statistical_analysis"]
                        if "Good" in stats_status:
                            st.success(f"📊 **Statistical Analysis:** {stats_status}")
                        elif "Fair" in stats_status:
                            st.warning(f"📊 **Statistical Analysis:** {stats_status}")
                        else:
                            st.error(f"📊 **Statistical Analysis:** {stats_status}")
                
                with col2:
                    if "sql_queries" in impact:
                        sql_status = impact["sql_queries"]
                        if "Good" in sql_status:
                            st.success(f"💾 **SQL Queries:** {sql_status}")
                        elif "Fair" in sql_status:
                            st.warning(f"💾 **SQL Queries:** {sql_status}")
                        else:
                            st.error(f"💾 **SQL Queries:** {sql_status}")
                    
                    if "visualization" in impact:
                        viz_status = impact["visualization"]
                        if "Good" in viz_status:
                            st.success(f"📈 **Visualization:** {viz_status}")
                        elif "Fair" in viz_status:
                            st.warning(f"📈 **Visualization:** {viz_status}")
                        else:
                            st.error(f"📈 **Visualization:** {viz_status}")
            
            # Key Blockers
            if assessment.get("key_blockers"):
                st.subheader("🚫 Key Blockers for Analytics")
                for i, blocker in enumerate(assessment["key_blockers"], 1):
                    st.error(f"{i}. {blocker}")
            
            # Recommended Actions
            if assessment.get("recommended_actions"):
                st.subheader("🔧 Recommended Actions")
                for i, action in enumerate(assessment["recommended_actions"], 1):
                    st.write(f"**{i}.** {action}")
            
        else:
            # Fallback to raw display
            st.json(parsed)
            
    except json.JSONDecodeError:
        # Display as markdown if not JSON
        st.markdown(raw_response)


def build_detailed_profiling_prompt() -> str:
    """Build a prompt for detailed data profiling insights"""
    if not st.session_state.get('ydata_profile'):
        return "No detailed profiling data available."
    
    profile = st.session_state.ydata_profile
    description = profile.description_set
    table_stats = description.table
    variables = description.variables
    
    # Extract comprehensive statistics
    n_vars = table_stats.get('n_var', 0)
    n_obs = table_stats.get('n', 0)
    missing_cells = table_stats.get('n_cells_missing', 0)
    total_cells = n_obs * n_vars if n_obs > 0 and n_vars > 0 else 1
    missing_percent = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    duplicate_rows = table_stats.get('n_duplicates', 0)
    duplicate_percent = (duplicate_rows / n_obs) * 100 if n_obs > 0 else 0
    
    # Variable type analysis
    types_summary = table_stats.get('types', {})
    
    # Sample variable details
    variable_samples = []
    for var_name, var_info in list(variables.items())[:5]:  # First 5 variables
        var_type = var_info.get('type', 'Unknown')
        missing_count = var_info.get('n_missing', 0)
        unique_count = var_info.get('n_distinct', 0)
        variable_samples.append(f"- {var_name}: {var_type}, {missing_count} missing, {unique_count} unique values")
    
    prompt = f"""
    Provide comprehensive insights for this detailed data profiling report:
    
    DATASET OVERVIEW:
    - Total Variables: {n_vars}
    - Total Observations: {n_obs:,}
    - Missing Data: {missing_percent:.1f}% of all cells
    - Duplicate Rows: {duplicate_percent:.1f}%
    - Variable Types Distribution: {types_summary}
    
    SAMPLE VARIABLE DETAILS:
    {chr(10).join(variable_samples)}
    
    Please provide detailed insights covering:
    1. **Data Structure Assessment**: Evaluate the overall structure and organization
    2. **Data Quality Deep Dive**: Detailed analysis of quality issues and their implications
    3. **Variable Relationship Insights**: Potential relationships and dependencies between variables
    4. **Advanced Cleaning Strategies**: Sophisticated approaches for data preparation
    5. **Analytics Preparation Roadmap**: Step-by-step plan for making data analytics-ready
    6. **Risk Assessment**: Potential risks and challenges for downstream analytics
    
    Focus on actionable, detailed recommendations for a data scientist or analyst.
    """
    
    return prompt


def build_profiling_outliers_prompt() -> str:
    """Build a prompt for profiling-based outlier analysis insights"""
    if not st.session_state.get('ydata_anomaly_results'):
        return "No anomaly detection results available from profiling."
    
    outlier_data = st.session_state.ydata_anomaly_results
    total_outliers = sum(info['count'] for info in outlier_data.values())
    dataset_size = len(st.session_state.data) if st.session_state.data is not None else 1
    outlier_rate = (total_outliers / dataset_size) * 100
    
    # Extract detailed outlier information
    outlier_summary = []
    for column, info in outlier_data.items():
        col_outlier_rate = (info['count'] / dataset_size) * 100
        outlier_summary.append(f"- {column}: {info['count']} outliers ({col_outlier_rate:.2f}%), method: {info.get('method', 'IQR')}")
    
    prompt = f"""
    Analyze these statistical anomaly detection results from comprehensive data profiling:
    
    ANOMALY DETECTION SUMMARY:
    - Total Outliers Detected: {total_outliers}
    - Overall Outlier Rate: {outlier_rate:.2f}%
    - Columns with Outliers: {len(outlier_data)}
    - Dataset Size: {dataset_size:,} rows
    
    DETAILED OUTLIER BREAKDOWN:
    {chr(10).join(outlier_summary)}
    
    Please provide comprehensive analysis covering:
    1. **Statistical Significance**: Whether these outlier rates are concerning or normal
    2. **Pattern Analysis**: What patterns or trends the outliers might indicate
    3. **Business Impact**: How these outliers could affect business insights and decisions
    4. **Treatment Strategies**: Specific approaches for handling each type of outlier
    5. **Model Impact Assessment**: How these outliers would affect different types of analytics
    6. **Investigation Priorities**: Which outliers should be investigated first and why
    7. **Validation Recommendations**: How to validate whether outliers are errors or legitimate values
    
    Provide practical, implementable recommendations for a data analysis team.
    """
    
    return prompt


def display_anomaly_analysis_results(parsed_response: dict):
    """Display structured anomaly analysis results in a user-friendly format"""
    
    analysis_summary = parsed_response.get("analysisSummary", {})
    treatment_strategies = parsed_response.get("treatmentStrategies", {})
    recommendations = parsed_response.get("recommendations", [])
    
    # Analysis Summary Section
    if analysis_summary:
        st.subheader("📊 Analysis Summary")
        
        # Statistical Significance
        if "statisticalSignificance" in analysis_summary:
            st.info(f"**📈 Statistical Significance:** {analysis_summary['statisticalSignificance']}")
        
        # Pattern Analysis
        if "patternAnalysis" in analysis_summary:
            with st.expander("🔍 Pattern Analysis", expanded=True):
                st.write(analysis_summary['patternAnalysis'])
        
        # Business Impact
        if "businessImpact" in analysis_summary:
            with st.expander("💼 Business Impact Assessment", expanded=True):
                st.write(analysis_summary['businessImpact'])
        
        # Model Impact Assessment
        if "modelImpactAssessment" in analysis_summary:
            with st.expander("🤖 Model Impact Assessment", expanded=False):
                st.write(analysis_summary['modelImpactAssessment'])
        
        # Investigation Priorities
        if "investigationPriorities" in analysis_summary:
            st.subheader("🎯 Investigation Priorities")
            st.warning(analysis_summary['investigationPriorities'])
        
        # Validation Recommendations
        if "validationRecommendations" in analysis_summary:
            st.subheader("✅ Validation Recommendations")
            st.info(analysis_summary['validationRecommendations'])
    
    # Treatment Strategies Section
    if treatment_strategies:
        st.subheader("🔧 Treatment Strategies")
        
        configurable_iqr = treatment_strategies.get("configurableIQR", {})
        if configurable_iqr:
            if "description" in configurable_iqr:
                st.info(f"**Method Context:** {configurable_iqr['description']}")
            
            strategies = configurable_iqr.get("strategies", [])
            if strategies:
                st.write("**Available Treatment Options:**")
                
                for i, strategy in enumerate(strategies):
                    if isinstance(strategy, dict):
                        strategy_name = strategy.get("strategyName", f"Strategy {i+1}")
                        condition = strategy.get("condition", "No condition specified")
                        action = strategy.get("action", "No action specified")
                        example = strategy.get("example", "")
                        
                        with st.expander(f"🛠️ {strategy_name}", expanded=False):
                            st.write(f"**When to Use:** {condition}")
                            st.write(f"**Action:** {action}")
                            if example:
                                st.code(f"Example: {example}")
    
    # Recommendations Section
    if recommendations:
        st.subheader("📋 Action Plan & Recommendations")
        
        # Group recommendations by priority
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for rec in recommendations:
            if isinstance(rec, dict):
                priority = rec.get("priority", "Medium").lower()
                if priority == "high":
                    high_priority.append(rec)
                elif priority == "medium":
                    medium_priority.append(rec)
                else:
                    low_priority.append(rec)
        
        # Display by priority
        if high_priority:
            st.markdown("#### 🔴 High Priority Actions")
            for i, rec in enumerate(high_priority, 1):
                action = rec.get("action", "No action specified")
                details = rec.get("details", "No details provided")
                
                st.error(f"**{i}. {action}**")
                st.write(f"📝 {details}")
                st.markdown("---")
        
        if medium_priority:
            st.markdown("#### 🟡 Medium Priority Actions")
            for i, rec in enumerate(medium_priority, 1):
                action = rec.get("action", "No action specified")
                details = rec.get("details", "No details provided")
                
                st.warning(f"**{i}. {action}**")
                st.write(f"📝 {details}")
                st.markdown("---")
        
        if low_priority:
            st.markdown("#### 🟢 Low Priority Actions")
            for i, rec in enumerate(low_priority, 1):
                action = rec.get("action", "No action specified")
                details = rec.get("details", "No details provided")
                
                st.info(f"**{i}. {action}**")
                st.write(f"📝 {details}")
                if i < len(low_priority):  # Don't add separator after last item
                    st.markdown("---")
    
    # Summary Call-to-Action
    st.markdown("---")
    st.success("💡 **Next Steps:** Follow the high-priority recommendations first, then work through medium and low priority items based on your project timeline and resources.")


def display_data_profiling_insights(insights: dict):
    """Display structured data profiling insights in a user-friendly format"""
    
    # Dataset Overview Section
    dataset_overview = insights.get("dataset_overview", {})
    if dataset_overview:
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_vars = dataset_overview.get("total_variables", 0)
            st.metric("Total Variables", total_vars)
        with col2:
            total_obs = dataset_overview.get("total_observations", 0)
            st.metric("Total Observations", f"{total_obs:,}")
        with col3:
            missing_pct = dataset_overview.get("missing_data_percentage", 0)
            st.metric("Missing Data", f"{missing_pct}%")
        with col4:
            dup_pct = dataset_overview.get("duplicate_rows_percentage", 0)
            st.metric("Duplicates", f"{dup_pct}%")
        
        # Variable type distribution
        var_types = dataset_overview.get("variable_type_distribution", {})
        if var_types:
            st.write("**Variable Type Distribution:**")
            type_cols = st.columns(len(var_types))
            for i, (var_type, count) in enumerate(var_types.items()):
                with type_cols[i]:
                    st.metric(f"{var_type} Variables", count)
        
        # Initial assessment
        if "initial_assessment" in dataset_overview:
            st.info(f"**Initial Assessment:** {dataset_overview['initial_assessment']}")
    
    # Data Structure Assessment
    data_structure = insights.get("data_structure_assessment", {})
    if data_structure and "recommendations" in data_structure:
        st.subheader("🏗️ Data Structure Assessment")
        display_recommendations_by_priority(data_structure["recommendations"], "Structure")
    
    # Data Quality Deep Dive
    data_quality = insights.get("data_quality_deep_dive", {})
    if data_quality and "recommendations" in data_quality:
        st.subheader("🔍 Data Quality Deep Dive")
        display_recommendations_by_priority(data_quality["recommendations"], "Quality")
    
    # Variable Relationship Insights
    var_relationships = insights.get("variable_relationship_insights", {})
    if var_relationships and "recommendations" in var_relationships:
        st.subheader("🔗 Variable Relationship Insights")
        display_recommendations_by_priority(var_relationships["recommendations"], "Relationships")
    
    # Advanced Cleaning Strategies
    cleaning_strategies = insights.get("advanced_cleaning_strategies", {})
    if cleaning_strategies and "recommendations" in cleaning_strategies:
        st.subheader("🧹 Advanced Cleaning Strategies")
        display_recommendations_by_priority(cleaning_strategies["recommendations"], "Cleaning")
    
    # Analytics Preparation Roadmap
    roadmap = insights.get("analytics_preparation_roadmap", {})
    if roadmap and "steps" in roadmap:
        st.subheader("🗺️ Analytics Preparation Roadmap")
        
        steps = roadmap["steps"]
        for step in steps:
            step_num = step.get("step_number", "")
            description = step.get("description", "")
            deliverable = step.get("deliverable", "")
            
            with st.expander(f"📋 Step {step_num}: {description.split(':')[0]}", expanded=False):
                st.write(f"**Task:** {description}")
                st.write(f"**Deliverable:** {deliverable}")
    
    # Risk Assessment
    risk_assessment = insights.get("risk_assessment", {})
    if risk_assessment and "potential_risks" in risk_assessment:
        st.subheader("⚠️ Risk Assessment")
        
        risks = risk_assessment["potential_risks"]
        for i, risk in enumerate(risks, 1):
            risk_name = risk.get("risk", f"Risk {i}")
            description = risk.get("description", "")
            mitigation = risk.get("mitigation", "")
            
            with st.expander(f"🚨 {risk_name}", expanded=False):
                st.write(f"**Description:** {description}")
                if mitigation:
                    st.write(f"**Mitigation:** {mitigation}")


def display_recommendations_by_priority(recommendations: list, section_name: str):
    """Display recommendations grouped by priority level"""
    
    # Group recommendations by priority
    high_priority = []
    medium_priority = []
    low_priority = []
    
    for rec in recommendations:
        priority = rec.get("priority", "Medium").lower()
        if priority == "high":
            high_priority.append(rec)
        elif priority == "medium":
            medium_priority.append(rec)
        else:
            low_priority.append(rec)
    
    # Display by priority
    if high_priority:
        st.markdown("#### 🔴 High Priority")
        for i, rec in enumerate(high_priority, 1):
            area = rec.get("area", "General")
            action = rec.get("action", "No action specified")
            rationale = rec.get("rationale", "No rationale provided")
            
            with st.expander(f"🎯 {area}", expanded=True):
                st.error(f"**Action:** {action}")
                st.write(f"**Rationale:** {rationale}")
    
    if medium_priority:
        st.markdown("#### 🟡 Medium Priority")
        for i, rec in enumerate(medium_priority, 1):
            area = rec.get("area", "General")
            action = rec.get("action", "No action specified")
            rationale = rec.get("rationale", "No rationale provided")
            
            with st.expander(f"⚡ {area}", expanded=False):
                st.warning(f"**Action:** {action}")
                st.write(f"**Rationale:** {rationale}")
    
    if low_priority:
        st.markdown("#### 🟢 Low Priority")
        for i, rec in enumerate(low_priority, 1):
            area = rec.get("area", "General")
            action = rec.get("action", "No action specified")
            rationale = rec.get("rationale", "No rationale provided")
            
            with st.expander(f"📌 {area}", expanded=False):
                st.info(f"**Action:** {action}")
                st.write(f"**Rationale:** {rationale}")
