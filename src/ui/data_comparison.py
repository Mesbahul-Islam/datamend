"""
Data Comparison and Root Cause Analysis Module
Handles dataset comparison, inconsistency detection, and root cause analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ChangeType(Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class InconsistencyType(Enum):
    ROW_COUNT_DIFF = "row_count_difference"
    COLUMN_DIFF = "column_difference"
    DATA_TYPE_DIFF = "data_type_difference"
    VALUE_DIFF = "value_difference"
    SCHEMA_DIFF = "schema_difference"
    DUPLICATE_DIFF = "duplicate_difference"
    NULL_VALUE_DIFF = "null_value_difference"
    STATISTICAL_DIFF = "statistical_difference"


@dataclass
class Inconsistency:
    type: InconsistencyType
    description: str
    severity: str  # "high", "medium", "low"
    affected_columns: List[str]
    source1_value: Any
    source2_value: Any
    potential_causes: List[str]
    recommended_actions: List[str]


@dataclass
class ComparisonResult:
    summary: Dict[str, Any]
    inconsistencies: List[Inconsistency]
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]


def data_comparison_tab():
    """Data comparison and root cause analysis tab"""
    st.header("Data Comparison & Root Cause Analysis")
    st.info("Compare datasets from different sources to identify inconsistencies and trace root causes")
    
    if not st.session_state.get('datasets') or len(st.session_state.datasets) < 2:
        st.warning("You need at least 2 datasets to perform comparison. Please load data using the sidebar.")
        return
    
    # Dataset selection for comparison
    st.subheader("Select Datasets for Comparison")
    
    col1, col2 = st.columns(2)
    
    dataset_names = list(st.session_state.datasets.keys())
    
    with col1:
        dataset1_name = st.selectbox("Primary Dataset (Reference):", dataset_names, key="comp_dataset1")
        dataset1 = get_dataset_dataframe(dataset1_name)
        if dataset1 is not None:
            st.caption(f"Source: {get_dataset_source_type(dataset1_name)}")
            st.caption(f"Shape: {dataset1.shape[0]:,} rows × {dataset1.shape[1]} columns")
    
    with col2:
        available_datasets = [name for name in dataset_names if name != dataset1_name]
        if available_datasets:
            dataset2_name = st.selectbox("Comparison Dataset:", available_datasets, key="comp_dataset2")
            dataset2 = get_dataset_dataframe(dataset2_name)
            if dataset2 is not None:
                st.caption(f"Source: {get_dataset_source_type(dataset2_name)}")
                st.caption(f"Shape: {dataset2.shape[0]:,} rows × {dataset2.shape[1]} columns")
        else:
            st.warning("Please select a different dataset for comparison")
            return
    
    if dataset1 is None or dataset2 is None:
        st.error("Error loading selected datasets")
        return
    
    # Comparison configuration
    st.subheader("Comparison Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Key columns for matching rows
        common_columns = list(set(dataset1.columns) & set(dataset2.columns))
        if common_columns:
            key_columns = st.multiselect(
                "Key columns for row matching:",
                common_columns,
                help="Select columns to use for matching rows between datasets"
            )
        else:
            st.warning("No common columns found between datasets")
            key_columns = []
    
    with col2:
        comparison_type = st.selectbox(
            "Comparison Type:",
            ["Full Comparison", "Schema Only", "Statistical Only", "Sample Comparison"],
            help="Choose the depth of comparison analysis"
        )
    
    with col3:
        if comparison_type == "Sample Comparison":
            sample_size = st.number_input("Sample size:", min_value=100, max_value=10000, value=1000)
        else:
            sample_size = None
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            tolerance_numeric = st.number_input("Numeric tolerance:", min_value=0.0, max_value=1.0, value=0.001, format="%.6f")
            ignore_case = st.checkbox("Ignore case in text comparison", value=True)
        
        with col2:
            check_duplicates = st.checkbox("Check for duplicates", value=True)
            analyze_nulls = st.checkbox("Analyze null patterns", value=True)
    
    # Run comparison
    if st.button("Run Comparison Analysis", type="primary"):
        if comparison_type in ["Full Comparison", "Sample Comparison"] and not key_columns:
            st.warning("Please select key columns for row-level comparison, or choose 'Schema Only' or 'Statistical Only'")
        else:
            run_comparison_analysis(
                dataset1, dataset2, dataset1_name, dataset2_name,
                key_columns, comparison_type, sample_size,
                tolerance_numeric, ignore_case, check_duplicates, analyze_nulls
            )


def get_dataset_dataframe(dataset_name: str) -> Optional[pd.DataFrame]:
    """Get DataFrame from dataset name, handling both formats"""
    if dataset_name not in st.session_state.datasets:
        return None
    
    dataset_info = st.session_state.datasets[dataset_name]
    
    if isinstance(dataset_info, dict) and 'dataframe' in dataset_info:
        return dataset_info['dataframe']
    else:
        return dataset_info


def get_dataset_source_type(dataset_name: str) -> str:
    """Determine the source type of a dataset"""
    dataset_info = st.session_state.datasets[dataset_name]
    
    if isinstance(dataset_info, dict) and 'source_type' in dataset_info:
        return dataset_info['source_type'].upper()
    elif dataset_name.startswith('snowflake_'):
        return 'SNOWFLAKE'
    elif dataset_name.endswith('.csv'):
        return 'CSV'
    elif dataset_name.endswith(('.xlsx', '.xls')):
        return 'EXCEL'
    else:
        return 'UNKNOWN'


def run_comparison_analysis(df1: pd.DataFrame, df2: pd.DataFrame, 
                          name1: str, name2: str,
                          key_columns: List[str], comparison_type: str, 
                          sample_size: Optional[int],
                          tolerance_numeric: float, ignore_case: bool,
                          check_duplicates: bool, analyze_nulls: bool):
    """Run comprehensive comparison analysis between two datasets"""
    
    with st.spinner("Running comparison analysis..."):
        # Initialize comparison result
        inconsistencies = []
        detailed_analysis = {}
        
        # Sample datasets if requested
        if comparison_type == "Sample Comparison" and sample_size:
            df1_sample = df1.sample(min(sample_size, len(df1))).reset_index(drop=True)
            df2_sample = df2.sample(min(sample_size, len(df2))).reset_index(drop=True)
        else:
            df1_sample = df1
            df2_sample = df2
        
        # 1. Schema Comparison
        schema_inconsistencies = compare_schemas(df1_sample, df2_sample, name1, name2)
        inconsistencies.extend(schema_inconsistencies)
        
        # 2. Basic Statistics Comparison
        stats_inconsistencies = compare_basic_statistics(df1_sample, df2_sample, name1, name2)
        inconsistencies.extend(stats_inconsistencies)
        
        # 3. Duplicate Analysis
        if check_duplicates:
            dup_inconsistencies = compare_duplicates(df1_sample, df2_sample, name1, name2)
            inconsistencies.extend(dup_inconsistencies)
        
        # 4. Null Value Analysis
        if analyze_nulls:
            null_inconsistencies = compare_null_patterns(df1_sample, df2_sample, name1, name2)
            inconsistencies.extend(null_inconsistencies)
        
        # 5. Row-level comparison (if key columns provided)
        if comparison_type in ["Full Comparison", "Sample Comparison"] and key_columns:
            row_inconsistencies = compare_row_level_data(
                df1_sample, df2_sample, name1, name2, key_columns, 
                tolerance_numeric, ignore_case
            )
            inconsistencies.extend(row_inconsistencies)
        
        # 6. Statistical Distribution Comparison
        dist_inconsistencies = compare_statistical_distributions(df1_sample, df2_sample, name1, name2)
        inconsistencies.extend(dist_inconsistencies)
        
        # Generate comparison summary
        summary = generate_comparison_summary(df1_sample, df2_sample, name1, name2, inconsistencies)
        
        # Generate recommendations and root cause analysis
        recommendations = generate_recommendations(inconsistencies, name1, name2)
        
        # Store results in session state
        st.session_state.comparison_result = ComparisonResult(
            summary=summary,
            inconsistencies=inconsistencies,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations
        )
        
        # Display results
        display_comparison_results(inconsistencies, summary, recommendations, name1, name2)


def compare_schemas(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str) -> List[Inconsistency]:
    """Compare schemas between two datasets"""
    inconsistencies = []
    
    # Column presence comparison
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    missing_in_df2 = cols1 - cols2
    missing_in_df1 = cols2 - cols1
    common_cols = cols1 & cols2
    
    if missing_in_df2:
        inconsistencies.append(Inconsistency(
            type=InconsistencyType.COLUMN_DIFF,
            description=f"Columns present in {name1} but missing in {name2}",
            severity="high",
            affected_columns=list(missing_in_df2),
            source1_value=list(missing_in_df2),
            source2_value="missing",
            potential_causes=[
                "Schema evolution - new columns added to source",
                "ETL pipeline missing column mappings",
                "Data source configuration differences",
                "Version mismatch between datasets"
            ],
            recommended_actions=[
                "Verify ETL pipeline column mappings",
                "Check data source schema versions",
                "Update data ingestion process",
                "Implement schema validation checks"
            ]
        ))
    
    if missing_in_df1:
        inconsistencies.append(Inconsistency(
            type=InconsistencyType.COLUMN_DIFF,
            description=f"Columns present in {name2} but missing in {name1}",
            severity="high",
            affected_columns=list(missing_in_df1),
            source1_value="missing",
            source2_value=list(missing_in_df1),
            potential_causes=[
                "Schema evolution - new columns added to target",
                "Data transformation adding derived columns",
                "Source system updates not reflected",
                "Manual data modifications"
            ],
            recommended_actions=[
                "Synchronize schema definitions",
                "Update source data extraction",
                "Review data transformation logic",
                "Implement automated schema sync"
            ]
        ))
    
    # Data type comparison for common columns
    for col in common_cols:
        dtype1 = str(df1[col].dtype)
        dtype2 = str(df2[col].dtype)
        
        if dtype1 != dtype2:
            inconsistencies.append(Inconsistency(
                type=InconsistencyType.DATA_TYPE_DIFF,
                description=f"Data type mismatch for column '{col}'",
                severity="medium",
                affected_columns=[col],
                source1_value=dtype1,
                source2_value=dtype2,
                potential_causes=[
                    "Data type conversion during ETL",
                    "Source system data type changes",
                    "Implicit type casting differences",
                    "Data loading configuration differences"
                ],
                recommended_actions=[
                    "Standardize data type definitions",
                    "Add explicit type casting in ETL",
                    "Review data loading parameters",
                    "Implement data type validation"
                ]
            ))
    
    return inconsistencies


def compare_basic_statistics(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str) -> List[Inconsistency]:
    """Compare basic statistics between datasets"""
    inconsistencies = []
    
    # Row count comparison
    row_diff = abs(len(df1) - len(df2))
    row_diff_pct = (row_diff / max(len(df1), len(df2))) * 100 if max(len(df1), len(df2)) > 0 else 0
    
    if row_diff_pct > 5:  # More than 5% difference
        severity = "high" if row_diff_pct > 20 else "medium"
        inconsistencies.append(Inconsistency(
            type=InconsistencyType.ROW_COUNT_DIFF,
            description=f"Significant row count difference: {row_diff:,} rows ({row_diff_pct:.1f}%)",
            severity=severity,
            affected_columns=[],
            source1_value=len(df1),
            source2_value=len(df2),
            potential_causes=[
                "Data filtering differences",
                "Incremental data loading",
                "Data retention policy differences",
                "ETL processing timing issues",
                "Data quality filtering variations"
            ],
            recommended_actions=[
                "Verify data loading timeframes",
                "Check filtering criteria consistency",
                "Review ETL processing logs",
                "Implement row count monitoring",
                "Standardize data retention policies"
            ]
        ))
    
    return inconsistencies


def compare_duplicates(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str) -> List[Inconsistency]:
    """Compare duplicate patterns between datasets"""
    inconsistencies = []
    
    # Overall duplicate comparison
    dups1 = df1.duplicated().sum()
    dups2 = df2.duplicated().sum()
    
    if dups1 != dups2:
        inconsistencies.append(Inconsistency(
            type=InconsistencyType.DUPLICATE_DIFF,
            description=f"Different duplicate counts",
            severity="medium",
            affected_columns=[],
            source1_value=dups1,
            source2_value=dups2,
            potential_causes=[
                "Data deduplication process differences",
                "Primary key constraints variations",
                "Data loading sequence issues",
                "Business rule changes for duplicates"
            ],
            recommended_actions=[
                "Standardize deduplication logic",
                "Implement consistent primary keys",
                "Review data loading procedures",
                "Add duplicate monitoring alerts"
            ]
        ))
    
    return inconsistencies


def compare_null_patterns(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str) -> List[Inconsistency]:
    """Compare null value patterns between datasets"""
    inconsistencies = []
    
    common_cols = list(set(df1.columns) & set(df2.columns))
    
    for col in common_cols:
        null1_pct = (df1[col].isnull().sum() / len(df1)) * 100
        null2_pct = (df2[col].isnull().sum() / len(df2)) * 100
        
        diff = abs(null1_pct - null2_pct)
        
        if diff > 10:  # More than 10% difference in null rates
            inconsistencies.append(Inconsistency(
                type=InconsistencyType.NULL_VALUE_DIFF,
                description=f"Significant null value difference in column '{col}': {diff:.1f}% difference",
                severity="medium" if diff > 20 else "low",
                affected_columns=[col],
                source1_value=f"{null1_pct:.1f}%",
                source2_value=f"{null2_pct:.1f}%",
                potential_causes=[
                    "Data validation rule changes",
                    "Source system data quality issues",
                    "ETL transformation logic differences",
                    "Data collection process changes"
                ],
                recommended_actions=[
                    "Review data validation rules",
                    "Check source data quality",
                    "Standardize null handling logic",
                    "Implement null value monitoring"
                ]
            ))
    
    return inconsistencies


def compare_row_level_data(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str,
                          key_columns: List[str], tolerance_numeric: float, ignore_case: bool) -> List[Inconsistency]:
    """Compare data at row level using key columns"""
    inconsistencies = []
    
    try:
        # Create key for matching
        df1_keys = df1[key_columns].copy()
        df2_keys = df2[key_columns].copy()
        
        # Handle case sensitivity for text columns
        if ignore_case:
            for col in key_columns:
                if df1_keys[col].dtype == 'object':
                    df1_keys[col] = df1_keys[col].astype(str).str.lower()
                    df2_keys[col] = df2_keys[col].astype(str).str.lower()
        
        # Create composite keys
        df1['_key'] = df1_keys.apply(lambda x: '|'.join(x.astype(str)), axis=1)
        df2['_key'] = df2_keys.apply(lambda x: '|'.join(x.astype(str)), axis=1)
        
        # Find common and different keys
        keys1 = set(df1['_key'])
        keys2 = set(df2['_key'])
        
        only_in_df1 = keys1 - keys2
        only_in_df2 = keys2 - keys1
        common_keys = keys1 & keys2
        
        # Rows only in dataset 1
        if only_in_df1:
            inconsistencies.append(Inconsistency(
                type=InconsistencyType.VALUE_DIFF,
                description=f"{len(only_in_df1)} rows exist only in {name1}",
                severity="medium",
                affected_columns=key_columns,
                source1_value=len(only_in_df1),
                source2_value=0,
                potential_causes=[
                    "Data synchronization lag",
                    "Filtering criteria differences",
                    "Data deletion in target system",
                    "Incremental loading issues"
                ],
                recommended_actions=[
                    "Check data synchronization timing",
                    "Review filtering logic",
                    "Verify data retention policies",
                    "Implement change data capture"
                ]
            ))
        
        # Rows only in dataset 2
        if only_in_df2:
            inconsistencies.append(Inconsistency(
                type=InconsistencyType.VALUE_DIFF,
                description=f"{len(only_in_df2)} rows exist only in {name2}",
                severity="medium",
                affected_columns=key_columns,
                source1_value=0,
                source2_value=len(only_in_df2),
                potential_causes=[
                    "New data added to target",
                    "Data recovery in target system",
                    "Manual data insertion",
                    "ETL processing differences"
                ],
                recommended_actions=[
                    "Verify new data sources",
                    "Check manual data modifications",
                    "Review ETL processing logic",
                    "Implement data lineage tracking"
                ]
            ))
        
        # Value differences in common rows (sample check)
        if common_keys:
            sample_keys = list(common_keys)[:min(1000, len(common_keys))]  # Sample for performance
            
            value_differences = 0
            different_columns = set()
            
            for key in sample_keys:
                row1 = df1[df1['_key'] == key].iloc[0]
                row2 = df2[df2['_key'] == key].iloc[0]
                
                common_cols = list(set(df1.columns) & set(df2.columns) - {'_key'})
                
                for col in common_cols:
                    val1 = row1[col]
                    val2 = row2[col]
                    
                    # Skip if both are null
                    if pd.isnull(val1) and pd.isnull(val2):
                        continue
                    
                    # Check for differences
                    if pd.isnull(val1) != pd.isnull(val2):
                        value_differences += 1
                        different_columns.add(col)
                    elif not pd.isnull(val1) and not pd.isnull(val2):
                        if pd.api.types.is_numeric_dtype(df1[col]):
                            if abs(float(val1) - float(val2)) > tolerance_numeric:
                                value_differences += 1
                                different_columns.add(col)
                        else:
                            str1 = str(val1).lower() if ignore_case else str(val1)
                            str2 = str(val2).lower() if ignore_case else str(val2)
                            if str1 != str2:
                                value_differences += 1
                                different_columns.add(col)
            
            if value_differences > 0:
                inconsistencies.append(Inconsistency(
                    type=InconsistencyType.VALUE_DIFF,
                    description=f"Value differences found in {value_differences} cells across {len(different_columns)} columns",
                    severity="high" if value_differences > len(sample_keys) * 0.1 else "medium",
                    affected_columns=list(different_columns),
                    source1_value=f"Sampled {len(sample_keys)} rows",
                    source2_value=f"{value_differences} differences found",
                    potential_causes=[
                        "Data transformation differences",
                        "Source system updates",
                        "Data correction processes",
                        "Calculation logic variations",
                        "Data entry errors"
                    ],
                    recommended_actions=[
                        "Review data transformation logic",
                        "Check source system change logs",
                        "Validate calculation formulas",
                        "Implement data quality checks",
                        "Add change tracking mechanisms"
                    ]
                ))
        
        # Clean up temporary columns
        df1.drop('_key', axis=1, inplace=True)
        df2.drop('_key', axis=1, inplace=True)
        
    except Exception as e:
        st.error(f"Error in row-level comparison: {str(e)}")
    
    return inconsistencies


def compare_statistical_distributions(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str) -> List[Inconsistency]:
    """Compare statistical distributions of numeric columns"""
    inconsistencies = []
    
    numeric_cols = list(set(df1.select_dtypes(include=[np.number]).columns) & 
                         set(df2.select_dtypes(include=[np.number]).columns))
    
    for col in numeric_cols:
        try:
            # Basic statistics comparison
            stats1 = df1[col].describe()
            stats2 = df2[col].describe()
            
            # Check for significant differences in key statistics
            metrics_to_check = ['mean', 'std', 'min', 'max', '50%']
            significant_diffs = []
            
            for metric in metrics_to_check:
                if metric in stats1 and metric in stats2:
                    val1, val2 = stats1[metric], stats2[metric]
                    if val1 != 0:
                        pct_diff = abs(val1 - val2) / abs(val1) * 100
                        if pct_diff > 20:  # More than 20% difference
                            significant_diffs.append(f"{metric}: {pct_diff:.1f}%")
            
            if significant_diffs:
                inconsistencies.append(Inconsistency(
                    type=InconsistencyType.STATISTICAL_DIFF,
                    description=f"Statistical distribution differences in column '{col}': {', '.join(significant_diffs)}",
                    severity="medium",
                    affected_columns=[col],
                    source1_value=stats1.to_dict(),
                    source2_value=stats2.to_dict(),
                    potential_causes=[
                        "Data range differences",
                        "Outlier handling variations",
                        "Data scaling/normalization differences",
                        "Business rule changes",
                        "Data quality improvement/degradation"
                    ],
                    recommended_actions=[
                        "Review data range specifications",
                        "Check outlier detection logic",
                        "Validate business rules",
                        "Compare data quality metrics",
                        "Implement statistical monitoring"
                    ]
                ))
        except Exception:
            continue
    
    return inconsistencies


def generate_comparison_summary(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str, 
                               inconsistencies: List[Inconsistency]) -> Dict[str, Any]:
    """Generate a summary of the comparison results"""
    summary = {
        'datasets': {
            name1: {'rows': len(df1), 'columns': len(df1.columns)},
            name2: {'rows': len(df2), 'columns': len(df2.columns)}
        },
        'inconsistency_counts': {
            'total': len(inconsistencies),
            'high_severity': len([i for i in inconsistencies if i.severity == 'high']),
            'medium_severity': len([i for i in inconsistencies if i.severity == 'medium']),
            'low_severity': len([i for i in inconsistencies if i.severity == 'low'])
        },
        'inconsistency_types': {},
        'comparison_timestamp': datetime.now().isoformat()
    }
    
    # Count inconsistencies by type
    for inconsistency in inconsistencies:
        inc_type = inconsistency.type.value
        if inc_type not in summary['inconsistency_types']:
            summary['inconsistency_types'][inc_type] = 0
        summary['inconsistency_types'][inc_type] += 1
    
    return summary


def generate_recommendations(inconsistencies: List[Inconsistency], name1: str, name2: str) -> List[str]:
    """Generate prioritized recommendations based on inconsistencies found"""
    recommendations = []
    
    # High-level recommendations based on inconsistency patterns
    high_severity_count = len([i for i in inconsistencies if i.severity == 'high'])
    schema_issues = len([i for i in inconsistencies if i.type == InconsistencyType.COLUMN_DIFF])
    data_issues = len([i for i in inconsistencies if i.type == InconsistencyType.VALUE_DIFF])
    
    if high_severity_count > 0:
        recommendations.append(f"🚨 PRIORITY: Address {high_severity_count} high-severity inconsistencies immediately")
    
    if schema_issues > 0:
        recommendations.append(f"📋 Schema Alignment: Resolve {schema_issues} schema differences between datasets")
    
    if data_issues > 0:
        recommendations.append(f"🔍 Data Quality: Investigate {data_issues} data value inconsistencies")
    
    # Specific recommendations from inconsistencies
    all_recommendations = []
    for inconsistency in inconsistencies:
        all_recommendations.extend(inconsistency.recommended_actions)
    
    # Get unique recommendations and add top ones
    unique_recommendations = list(set(all_recommendations))
    recommendations.extend(unique_recommendations[:10])  # Top 10 unique recommendations
    
    # Add general best practices
    recommendations.extend([
        "📊 Implement automated data quality monitoring",
        "🔄 Set up regular data comparison schedules",
        "📝 Document data lineage and transformation logic",
        "⚠️ Create alerts for significant data changes",
        "🎯 Establish data quality SLAs and metrics"
    ])
    
    return recommendations


def display_comparison_results(inconsistencies: List[Inconsistency], summary: Dict[str, Any], 
                              recommendations: List[str], name1: str, name2: str):
    """Display comprehensive comparison results"""
    
    st.success("Comparison analysis completed!")
    
    # Summary metrics
    st.subheader("Comparison Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Inconsistencies", summary['inconsistency_counts']['total'])
    with col2:
        st.metric("High Severity", summary['inconsistency_counts']['high_severity'])
    with col3:
        st.metric("Medium Severity", summary['inconsistency_counts']['medium_severity'])
    with col4:
        st.metric("Low Severity", summary['inconsistency_counts']['low_severity'])
    
    # Dataset information
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**{name1}**\n{summary['datasets'][name1]['rows']:,} rows × {summary['datasets'][name1]['columns']} columns")
    with col2:
        st.info(f"**{name2}**\n{summary['datasets'][name2]['rows']:,} rows × {summary['datasets'][name2]['columns']} columns")
    
    # Inconsistency breakdown chart
    if inconsistencies:
        st.subheader("Inconsistency Analysis")
        
        # Create visualization
        fig = create_inconsistency_charts(inconsistencies, summary)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed inconsistency list
        st.subheader("Detailed Inconsistencies")
        
        # Filter by severity
        severity_filter = st.selectbox("Filter by severity:", ["All", "High", "Medium", "Low"])
        
        filtered_inconsistencies = inconsistencies
        if severity_filter != "All":
            filtered_inconsistencies = [i for i in inconsistencies if i.severity.lower() == severity_filter.lower()]
        
        for i, inconsistency in enumerate(filtered_inconsistencies):
            with st.expander(f"{get_severity_emoji(inconsistency.severity)} {inconsistency.description}"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Details:**")
                    st.write(f"• **Type:** {inconsistency.type.value.replace('_', ' ').title()}")
                    st.write(f"• **Severity:** {inconsistency.severity.upper()}")
                    if inconsistency.affected_columns:
                        st.write(f"• **Affected Columns:** {', '.join(inconsistency.affected_columns)}")
                    st.write(f"• **{name1} Value:** {inconsistency.source1_value}")
                    st.write(f"• **{name2} Value:** {inconsistency.source2_value}")
                
                with col2:
                    st.write("**Root Cause Analysis:**")
                    for cause in inconsistency.potential_causes:
                        st.write(f"• {cause}")
                    
                    st.write("**Recommended Actions:**")
                    for action in inconsistency.recommended_actions:
                        st.write(f"• {action}")
    
    # Recommendations
    st.subheader("Recommendations & Action Plan")
    for i, recommendation in enumerate(recommendations, 1):
        st.write(f"{i}. {recommendation}")
    
    # Export option
    st.subheader("Export Results")
    if st.button("Generate Detailed Report"):
        generate_comparison_report(inconsistencies, summary, recommendations, name1, name2)


def create_inconsistency_charts(inconsistencies: List[Inconsistency], summary: Dict[str, Any]):
    """Create visualization charts for inconsistencies"""
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Inconsistencies by Severity', 'Inconsistencies by Type'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )
    
    # Severity breakdown
    severity_data = [
        summary['inconsistency_counts']['high_severity'],
        summary['inconsistency_counts']['medium_severity'],
        summary['inconsistency_counts']['low_severity']
    ]
    severity_labels = ['High', 'Medium', 'Low']
    severity_colors = ['#ff4b4b', '#ffa500', '#90EE90']
    
    fig.add_trace(
        go.Pie(
            labels=severity_labels,
            values=severity_data,
            marker_colors=severity_colors,
            hole=0.3
        ),
        row=1, col=1
    )
    
    # Type breakdown
    type_counts = summary['inconsistency_types']
    type_labels = [t.replace('_', ' ').title() for t in type_counts.keys()]
    type_values = list(type_counts.values())
    
    fig.add_trace(
        go.Bar(
            x=type_labels,
            y=type_values,
            marker_color='lightblue'
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    
    return fig


def get_severity_emoji(severity: str) -> str:
    """Get emoji for severity level"""
    emojis = {
        'high': '🔴',
        'medium': '🟡', 
        'low': '🟢'
    }
    return emojis.get(severity.lower(), '⚪')


def generate_comparison_report(inconsistencies: List[Inconsistency], summary: Dict[str, Any], 
                              recommendations: List[str], name1: str, name2: str):
    """Generate a downloadable comparison report"""
    
    report_content = f"""
# Data Comparison Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

Comparison between:
- **Primary Dataset:** {name1} ({summary['datasets'][name1]['rows']:,} rows × {summary['datasets'][name1]['columns']} columns)
- **Comparison Dataset:** {name2} ({summary['datasets'][name2]['rows']:,} rows × {summary['datasets'][name2]['columns']} columns)

### Key Findings
- **Total Inconsistencies:** {summary['inconsistency_counts']['total']}
- **High Severity Issues:** {summary['inconsistency_counts']['high_severity']}
- **Medium Severity Issues:** {summary['inconsistency_counts']['medium_severity']}
- **Low Severity Issues:** {summary['inconsistency_counts']['low_severity']}

## Detailed Analysis

"""

    for i, inconsistency in enumerate(inconsistencies, 1):
        report_content += f"""
### {i}. {inconsistency.description}

**Severity:** {inconsistency.severity.upper()}
**Type:** {inconsistency.type.value.replace('_', ' ').title()}
**Affected Columns:** {', '.join(inconsistency.affected_columns) if inconsistency.affected_columns else 'N/A'}

**Values:**
- {name1}: {inconsistency.source1_value}
- {name2}: {inconsistency.source2_value}

**Potential Root Causes:**
{chr(10).join(f'- {cause}' for cause in inconsistency.potential_causes)}

**Recommended Actions:**
{chr(10).join(f'- {action}' for action in inconsistency.recommended_actions)}

---
"""

    report_content += f"""
## Recommendations

{chr(10).join(f'{i}. {rec}' for i, rec in enumerate(recommendations, 1))}

## Next Steps

1. Prioritize high-severity inconsistencies for immediate resolution
2. Implement recommended monitoring and validation processes  
3. Schedule regular data comparison reviews
4. Document resolution steps for future reference

---
*Report generated by DataMend - Data Quality Engine*
"""
    
    st.download_button(
        label="Download Report",
        data=report_content,
        file_name=f"data_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )
    
    st.success("Report generated successfully!")
