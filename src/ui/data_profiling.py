"""
Data Profiling Module
Handles ydata-profiling integration and report display functionality
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from ydata_profiling import ProfileReport


def data_profiling_tab(chunk_size: int, max_workers: int, anomaly_threshold: float):
    """Data profiling tab"""
    st.header("📊 Data Profiling")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Source tab")
        return
    
    # Show current dataset info
    if st.session_state.current_dataset:
        st.info(f"📊 **Analyzing Dataset**: {st.session_state.current_dataset}")
    
    df = st.session_state.data
    
    # Profiling controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Calculate dataset size information
        dataset_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        # Determine processing strategy message
        if len(df) > 100000:
            processing_strategy = "🚀 Large dataset - using parallel processing"
        else:
            processing_strategy = "⚡ Standard dataset - using optimized sequential processing"
        
        st.write(f"**Dataset:** {len(df):,} rows × {len(df.columns)} columns ({dataset_size_mb:.1f} MB)")
    
    with col2:
        if st.button("🔄 Run Profiling", type="primary", key="run_profiling_button"):
            run_data_profiling(df, chunk_size, max_workers, anomaly_threshold)
    
    # Show results if available
    if st.session_state.get('ydata_profile'):
        display_ydata_profiling_results(st.session_state.ydata_profile)


def run_data_profiling(df: pd.DataFrame, chunk_size: int, max_workers: int, anomaly_threshold: float):
    """Run data profiling on the dataset using ydata-profiling"""
    try:
        with st.spinner("🔍 Running comprehensive data profiling with ydata-profiling..."):
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
        st.error(f"❌ Error during profiling: {str(e)}")
        st.error(f"Details: {type(e).__name__}: {str(e)}")


def display_ydata_profiling_results(profile):
    """Display ydata-profiling results with interactive HTML report and summary"""
    
    # Executive Summary from ydata-profiling
    st.subheader("📈 Data Profiling Report")

    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Quick Summary", "📄 Detailed Report", "💾 Export Options"])
    
    with tab1:
        display_ydata_summary(profile)
    
    with tab2:
        display_enhanced_report(profile)
    
    with tab3:
        display_export_options(profile)


def display_ydata_summary(profile):
    """Display a quick summary of the ydata-profiling results"""
    st.subheader("📋 Quick Data Summary")
    
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
    st.subheader("📊 Variable Types")
    
    types_summary = table_stats.get('types', {})
    if types_summary:
        types_df = pd.DataFrame([
            {"Type": type_name.replace('_', ' ').title(), "Count": count}
            for type_name, count in types_summary.items()
        ])
        st.dataframe(types_df, use_container_width=True, hide_index=True)
    
    # Warnings and alerts
    st.subheader("⚠️ Data Quality Alerts")
    
    alerts = []
    
    if missing_percent > 10:
        alerts.append(f"🔸 High missing data: {missing_percent:.1f}% of cells are missing")
    
    if duplicate_percent > 5:
        alerts.append(f"🔸 Duplicate rows detected: {duplicate_percent:.1f}% of rows are duplicates")
    
    n_constant = table_stats.get('n_constant', 0)
    if n_constant > 0:
        alerts.append(f"🔸 Constant variables: {n_constant} variables have only one unique value")
    
    if not alerts:
        st.success("✅ No major data quality issues detected!")
    else:
        for alert in alerts:
            st.warning(alert)


def display_full_html_report(profile):
    """Display the full HTML report from ydata-profiling"""
    st.subheader("📊 Full Profiling Report")
    
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
    st.subheader("💾 Download Report")
    
    try:
        # Generate the HTML report
        html_report = profile.to_html()
        
        # Create download button
        dataset_name = st.session_state.current_dataset or "dataset"
        filename = f"data_profile_report_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        st.download_button(
            label="📥 Download HTML Report",
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
        st.markdown("### 📈 Dataset Overview")
        
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
        st.markdown("### 🔢 Variable Types Distribution")
        types_summary = table_stats.get('types', {})
        if types_summary:
            types_df = pd.DataFrame([
                {"Variable Type": type_name.replace('_', ' ').title(), "Count": count, "Percentage": f"{(count/sum(types_summary.values()))*100:.1f}%"}
                for type_name, count in types_summary.items()
            ])
            st.dataframe(types_df, use_container_width=True, hide_index=True)
        
        # Variable Details Analysis
        st.markdown("### 📋 Variable Analysis")
        
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
                    status = "🔴 Critical"
                elif missing_pct > 20:
                    status = "🟠 Warning"
                elif missing_pct > 5:
                    status = "🟡 Attention"
                else:
                    status = "🟢 Good"
                
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
            st.dataframe(variables_df, use_container_width=True, hide_index=True)
        
        # Data Quality Issues
        st.markdown("### ⚠️ Data Quality Assessment")
        
        quality_issues = []
        
        # Check for high missing data
        high_missing_vars = [name for name, info in variables.items() 
                           if (info.get('n_missing', 0) / info.get('n', 1)) > 0.1]
        if high_missing_vars:
            quality_issues.append(f"🔸 High missing data in {len(high_missing_vars)} variables: {', '.join(high_missing_vars[:3])}{'...' if len(high_missing_vars) > 3 else ''}")
        
        # Check for duplicate rows
        if duplicate_pct > 5:
            quality_issues.append(f"🔸 Duplicate rows detected: {duplicate_pct:.1f}% of the dataset")
        
        # Check for constant variables
        constant_vars = [name for name, info in variables.items() 
                        if info.get('n_distinct', 0) <= 1]
        if constant_vars:
            quality_issues.append(f"🔸 Constant variables found: {', '.join(constant_vars)}")
        
        if quality_issues:
            for issue in quality_issues:
                st.warning(issue)
        else:
            st.success("✅ No major data quality issues detected!")
        
        # Full HTML Report Option
        st.markdown("### 📄 Complete Interactive Report")
        st.info("For the most comprehensive analysis with interactive visualizations, you can generate the full HTML report.")
        
        col1, col2 = st.columns([1, 1])
        
        # Buttons for generating and downloading report
        generate_report = False
        with col1:
            if st.button("🔍 Generate Interactive Report", type="primary", use_container_width=True):
                generate_report = True
        
        with col2:
            if st.button("💾 Download for Best View", type="secondary", use_container_width=True):
                html_report = profile.to_html()
                dataset_name = st.session_state.current_dataset or "dataset"
                filename = f"data_profile_report_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                st.download_button(
                    label="📥 Download HTML Report",
                    data=html_report,
                    file_name=filename,
                    mime="text/html",
                    help="Download for optimal viewing experience",
                    use_container_width=True
                )
        
        # Display HTML report spanning full width (two columns)
        if generate_report:
            with st.spinner("Generating comprehensive HTML report..."):
                html_report = profile.to_html()
                # Display the HTML report spanning the full width
                st.components.v1.html(html_report, width=None, height=1200, scrolling=True)
        
        # Anomaly Detection Integration
        st.markdown("### 🎯 Anomaly Detection via ydata-profiling")
        st.info("💡 Extract and analyze outliers/anomalies directly from the ydata-profiling report.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔍 Extract Anomalies", type="primary", use_container_width=True, help="Extract outliers detected by ydata-profiling"):
                display_profiling_outliers(profile, st.session_state.data)
        
        with col2:
            if st.button("📊 Anomaly Summary", type="secondary", use_container_width=True, help="Show comprehensive anomaly analysis from profiling"):
                display_comprehensive_anomaly_analysis(profile, st.session_state.data)
        
    except Exception as e:
        st.error(f"Error displaying enhanced report: {str(e)}")
        st.info("Please try the Quick Summary view or download the report for offline viewing.")


def display_export_options(profile):
    """Provide various export and sharing options"""
    st.subheader("💾 Export & Share Options")
    
    try:
        # Generate the HTML report
        html_report = profile.to_html()
        
        # Create download button
        dataset_name = st.session_state.current_dataset or "dataset"
        filename = f"data_profile_report_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.download_button(
                label="📥 Download Complete HTML Report",
                data=html_report,
                file_name=filename,
                mime="text/html",
                help="Download the complete profiling report as an HTML file",
                use_container_width=True
            )
        
        with col2:
            file_size_mb = len(html_report.encode('utf-8')) / (1024 * 1024)
            st.metric("Report Size", f"{file_size_mb:.1f} MB")
        
        st.success("📄 Report ready for download! The HTML file contains:")
        
        features = [
            "🔍 Interactive data exploration",
            "📊 Comprehensive statistical analysis", 
            "📈 Distribution plots and histograms",
            "🔗 Correlation matrices",
            "⚠️ Data quality warnings",
            "📋 Missing data patterns",
            "🎯 Outlier detection results",
            "📱 Mobile-responsive design"
        ]
        
        col1, col2 = st.columns(2)
        for i, feature in enumerate(features):
            if i % 2 == 0:
                col1.write(feature)
            else:
                col2.write(feature)
        
        st.info("💡 **Tip**: Open the downloaded HTML file in your browser for the best viewing experience with full interactivity.")
        
    except Exception as e:
        st.error(f"Error preparing export options: {str(e)}")


def extract_profiling_outliers(profile, df: pd.DataFrame):
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
                
                # Use ydata-profiling's IQR method for outlier detection
                Q1 = column_data.quantile(0.25)
                Q3 = column_data.quantile(0.75)
                IQR = Q3 - Q1
                
                # Standard IQR outlier detection (same as ydata-profiling)
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
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
                        'method': 'ydata-profiling IQR',
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


def display_profiling_outliers(profile, df: pd.DataFrame):
    """Display outliers detected by ydata-profiling"""
    st.subheader("📊 Anomaly Detection via ydata-profiling")
    
    with st.spinner("Extracting outlier information from ydata-profiling report..."):
        outliers_info = extract_profiling_outliers(profile, df)
    
    if not outliers_info:
        st.success("🎉 No outliers detected by ydata-profiling analysis!")
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
        severity = "🔴 High" if outlier_rate > 5 else "🟡 Medium" if outlier_rate > 1 else "🟢 Low"
        st.metric("Severity", severity)
    
    # Detailed analysis
    st.subheader("📋 Column-wise Anomaly Analysis")
    
    for column, info in outliers_info.items():
        anomaly_pct = (info['count']/len(df)*100)
        priority_icon = "🚨" if anomaly_pct > 10 else "⚠️" if anomaly_pct > 5 else "🟡" if anomaly_pct > 1 else "🔵"
        
        with st.expander(f"{priority_icon} **{column}** - {info['count']} anomalies ({anomaly_pct:.2f}%)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📊 Statistical Boundaries (IQR Method):**")
                st.write(f"• Lower bound: {info['lower_bound']:.3f}")
                st.write(f"• Upper bound: {info['upper_bound']:.3f}")
                st.write(f"• Detection method: {info['method']}")
                
                st.write("**📈 Column Statistics:**")
                stats = info['statistics']
                st.write(f"• Mean: {stats['mean']:.3f}")
                st.write(f"• Std Dev: {stats['std']:.3f}")
                st.write(f"• Range: {stats['min']:.3f} to {stats['max']:.3f}")
            
            with col2:
                st.write("**🎯 Sample Anomalous Values:**")
                sample_values = info['values'][:8]  # Show first 8
                for i, value in enumerate(sample_values, 1):
                    deviation = "above" if value > info['upper_bound'] else "below"
                    st.write(f"{i}. {value:.3f} ({deviation} normal range)")
                if len(info['values']) > 8:
                    st.write(f"... and {len(info['values']) - 8} more anomalies")
            
            # Show outlier locations
            if st.checkbox(f"Show anomaly row indices for {column}", key=f"show_indices_{column}"):
                indices_text = ", ".join(map(str, info['indices'][:20]))
                if len(info['indices']) > 20:
                    indices_text += f" ... and {len(info['indices']) - 20} more"
                st.write(f"**Row indices with anomalies:** {indices_text}")
            
            # Statistical analysis
            if st.checkbox(f"Show detailed statistics for {column}", key=f"show_stats_{column}"):
                stats = info['statistics']
                st.write("**📊 Detailed Statistics:**")
                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    st.write(f"• Q1 (25th percentile): {stats['Q1']:.3f}")
                    st.write(f"• Q3 (75th percentile): {stats['Q3']:.3f}")
                    st.write(f"• IQR (Q3-Q1): {stats['IQR']:.3f}")
                with detail_col2:
                    st.write(f"• Minimum value: {stats['min']:.3f}")
                    st.write(f"• Maximum value: {stats['max']:.3f}")
                    st.write(f"• Data points: {len(df[column].dropna())}")
    
    # Store results for anomaly detection tab
    st.session_state.ydata_anomaly_results = outliers_info
    st.session_state.ydata_anomaly_detection = True


def display_comprehensive_anomaly_analysis(profile, df: pd.DataFrame):
    """Display comprehensive anomaly analysis from ydata-profiling"""
    st.subheader("📈 Comprehensive Anomaly Analysis")
    
    with st.spinner("Analyzing anomalies from ydata-profiling report..."):
        outliers_info = extract_profiling_outliers(profile, df)
        description = profile.description_set
        variables = description.variables
    
    if not outliers_info:
        st.success("🎉 Excellent! No anomalies detected in your dataset.")
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
        assessment = "🟢 **Excellent** - Very few anomalies detected"
        message = "Your data quality is excellent with minimal outliers."
    elif anomaly_rate < 5:
        assessment = "🟡 **Good** - Some anomalies present" 
        message = "Your data has some outliers that may need investigation."
    elif anomaly_rate < 10:
        assessment = "🟠 **Attention Needed** - Moderate anomalies"
        message = "Several anomalies detected that require attention."
    else:
        assessment = "🔴 **Critical** - High anomaly rate"
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
    st.subheader("🎯 Priority Anomaly Columns")
    
    priority_data = []
    for column, info in outliers_info.items():
        col_anomaly_rate = (info['count'] / len(df)) * 100
        
        if col_anomaly_rate > 10:
            priority = "🚨 Critical"
        elif col_anomaly_rate > 5:
            priority = "⚠️ High"
        elif col_anomaly_rate > 1:
            priority = "🟡 Medium"
        else:
            priority = "🔵 Low"
        
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
    st.dataframe(priority_df, use_container_width=True, hide_index=True)
    
    # Actionable Recommendations
    st.subheader("💡 Recommended Actions")
    
    recommendations = []
    
    critical_cols = [item for item in priority_data if "Critical" in item["Priority"]]
    if critical_cols:
        recommendations.append("🚨 **Immediate Action Required**: Critical anomaly levels detected - investigate data collection processes")
    
    high_cols = [item for item in priority_data if "High" in item["Priority"]]
    if high_cols:
        recommendations.append("⚠️ **High Priority**: Review anomalies to determine if they represent valid edge cases or errors")
    
    if anomaly_rate > 5:
        recommendations.append("📊 **Process Review**: High overall anomaly rate suggests reviewing data validation procedures")
    
    if len(outliers_info) > len([col for col in df.columns if df[col].dtype in ['int64', 'float64']]) * 0.5:
        recommendations.append("🔍 **Data Quality Audit**: More than half of numeric columns have anomalies")
    
    if recommendations:
        for rec in recommendations:
            st.warning(rec)
    else:
        st.success("✅ **No immediate action required** - Anomaly levels are within acceptable ranges")
    
    # Column Details
    st.subheader("📊 Detailed Column Analysis")
    
    for i, (column, info) in enumerate(outliers_info.items()):
        if i < 3:  # Show details for top 3 problematic columns
            col_anomaly_rate = (info['count'] / len(df)) * 100
            priority_icon = "🚨" if col_anomaly_rate > 10 else "⚠️" if col_anomaly_rate > 5 else "🟡"
            
            with st.expander(f"{priority_icon} **{column}** - Detailed Analysis"):
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.write("**Anomaly Summary:**")
                    st.write(f"• Total anomalies: {info['count']}")
                    st.write(f"• Anomaly rate: {col_anomaly_rate:.2f}%")
                    st.write(f"• Detection: IQR method from ydata-profiling")
                    
                    stats = info['statistics']
                    st.write("**Statistical Context:**")
                    st.write(f"• Mean: {stats['mean']:.3f}")
                    st.write(f"• Std Dev: {stats['std']:.3f}")
                
                with detail_col2:
                    st.write("**Anomaly Boundaries:**")
                    st.write(f"• Normal range: {info['lower_bound']:.3f} to {info['upper_bound']:.3f}")
                    st.write(f"• IQR: {info['statistics']['IQR']:.3f}")
                    
                    st.write("**Sample Anomalous Values:**")
                    sample_values = info['values'][:3]
                    for j, value in enumerate(sample_values, 1):
                        deviation = "above" if value > info['upper_bound'] else "below"
                        st.write(f"{j}. {value:.3f} ({deviation})")
    
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
