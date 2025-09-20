"""
Simple Metadata Extraction Module
Extracts basic metadata from datasets without complex calculations
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class ColumnMetadata:
    """Basic metadata for a single column"""
    name: str
    data_type: str
    non_null_count: int
    null_count: int
    null_percentage: float


@dataclass
class DatasetMetadata:
    """Basic metadata for a dataset"""
    # Basic info
    name: str
    source_type: str
    row_count: int
    column_count: int
    creation_time: datetime
    
    # Schema info
    column_names: List[str]
    column_types: Dict[str, str]
    
    # Data quality metrics
    total_null_count: int
    
    # Column-level metadata
    columns: Dict[str, ColumnMetadata]
    
    # Source-specific metadata
    source_metadata: Dict[str, Any]


class MetadataExtractor:
    """Extract basic metadata from datasets quickly"""
    
    def __init__(self):
        pass
    
    def extract_metadata(self, df: pd.DataFrame, name: str, source_type: str, 
                        source_metadata: Optional[Dict[str, Any]] = None) -> DatasetMetadata:
        """Extract basic metadata from a DataFrame quickly"""
        
        # Basic dataset info - no expensive operations
        row_count = len(df)
        column_count = len(df.columns)
        creation_time = datetime.now()
        
        # Schema information - simple operations only
        column_names = list(df.columns)
        column_types = {col: str(df[col].dtype) for col in df.columns}
        
        # Basic data quality - just null counts
        total_null_count = int(df.isnull().sum().sum())
        
        # Extract basic column-level metadata - no complex calculations
        columns_metadata = {}
        for col in df.columns:
            columns_metadata[col] = self._extract_basic_column_metadata(df[col], col)
        
        return DatasetMetadata(
            name=name,
            source_type=source_type,
            row_count=row_count,
            column_count=column_count,
            creation_time=creation_time,
            column_names=column_names,
            column_types=column_types,
            total_null_count=total_null_count,
            columns=columns_metadata,
            source_metadata=source_metadata or {}
        )
    
    def _extract_basic_column_metadata(self, series: pd.Series, col_name: str) -> ColumnMetadata:
        """Extract basic metadata for a single column - no expensive operations"""
        
        # Only basic, fast operations
        total_count = len(series)
        non_null_count = int(series.count())
        null_count = total_count - non_null_count
        null_percentage = (null_count / total_count * 100) if total_count > 0 else 0
        
        return ColumnMetadata(
            name=col_name,
            data_type=str(series.dtype),
            non_null_count=non_null_count,
            null_count=null_count,
            null_percentage=round(null_percentage, 2)
        )
    
    def extract_snowflake_metadata(self, connection, table_name: str, schema: str = None, database: str = None) -> DatasetMetadata:
        """Extract metadata from Snowflake table without loading data"""
        
        # Build table reference
        table_ref = f"{database}.{schema}.{table_name}" if database and schema else table_name
        
        # Get basic table info from Snowflake metadata
        info_query = f"""
        SELECT 
            COUNT(*) as row_count,
            COUNT(DISTINCT column_name) as column_count
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE table_name = '{table_name.upper()}'
        """
        
        if schema:
            info_query += f" AND table_schema = '{schema.upper()}'"
        if database:
            info_query += f" AND table_catalog = '{database.upper()}'"
        
        # Get column metadata from Snowflake information schema
        columns_query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE table_name = '{table_name.upper()}'
        """
        
        if schema:
            columns_query += f" AND table_schema = '{schema.upper()}'"
        if database:
            columns_query += f" AND table_catalog = '{database.upper()}'"
        columns_query += " ORDER BY ordinal_position"
        
        try:
            # Execute queries
            cursor = connection.cursor()
            
            # Get basic info
            cursor.execute(info_query)
            info_result = cursor.fetchone()
            row_count = info_result[0] if info_result else 0
            
            # Get column info
            cursor.execute(columns_query)
            columns_result = cursor.fetchall()
            
            column_names = [row[0] for row in columns_result]
            column_types = {row[0]: row[1] for row in columns_result}
            
            # Create basic column metadata (no data sampling for speed)
            columns_metadata = {}
            for row in columns_result:
                col_name, data_type, is_nullable = row
                columns_metadata[col_name] = ColumnMetadata(
                    name=col_name,
                    data_type=data_type,
                    non_null_count=0,  # Would need data sampling to calculate
                    null_count=0,      # Would need data sampling to calculate
                    null_percentage=0.0
                )
            
            return DatasetMetadata(
                name=table_name,
                source_type="snowflake",
                row_count=row_count,
                column_count=len(column_names),
                creation_time=datetime.now(),
                column_names=column_names,
                column_types=column_types,
                total_null_count=0,  # Would need full table scan
                columns=columns_metadata,
                source_metadata={
                    "database": database,
                    "schema": schema,
                    "table_name": table_name
                }
            )
            
        except Exception as e:
            # Fallback to basic metadata
            return DatasetMetadata(
                name=table_name,
                source_type="snowflake",
                row_count=0,
                column_count=0,
                creation_time=datetime.now(),
                column_names=[],
                column_types={},
                total_null_count=0,
                columns={},
                source_metadata={
                    "database": database,
                    "schema": schema,
                    "table_name": table_name,
                    "error": str(e)
                }
            )
    
    def to_dict(self, metadata: DatasetMetadata) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        result = asdict(metadata)
        # Convert datetime to string for JSON serialization
        result['creation_time'] = metadata.creation_time.isoformat()
        return result
    
    def to_json(self, metadata: DatasetMetadata) -> str:
        """Convert metadata to JSON string"""
        return json.dumps(self.to_dict(metadata), indent=2)
    
    def from_dict(self, data: Dict[str, Any]) -> DatasetMetadata:
        """Create metadata from dictionary"""
        # Convert string back to datetime
        if isinstance(data['creation_time'], str):
            data['creation_time'] = datetime.fromisoformat(data['creation_time'])
        
        # Convert column metadata
        if 'columns' in data:
            columns = {}
            for col_name, col_data in data['columns'].items():
                columns[col_name] = ColumnMetadata(**col_data)
            data['columns'] = columns
        
        return DatasetMetadata(**data)
            max_value = series.max()
        except (TypeError, ValueError):
            min_value = None
            max_value = None
        
        # Statistical measures for numeric columns
        mean_value = None
        std_value = None
        if pd.api.types.is_numeric_dtype(series):
            try:
                mean_value = float(series.mean())
                std_value = float(series.std())
            except (TypeError, ValueError):
                pass
        
        # Sample values
        sample_values = self._get_sample_values(series)
        
        # Generate hashes
        value_hash = self._calculate_column_value_hash(series)
        stats_hash = self._calculate_column_stats_hash(series)
        
        return ColumnMetadata(
            name=col_name,
            data_type=str(series.dtype),
            non_null_count=non_null_count,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            duplicate_count=duplicate_count,
            min_value=min_value,
            max_value=max_value,
            mean_value=mean_value,
            std_value=std_value,
            memory_usage=memory_usage,
            sample_values=sample_values,
            value_hash=value_hash,
            stats_hash=stats_hash
        )
    
    def _get_sample_values(self, series: pd.Series) -> List[Any]:
        """Get sample values from a series"""
        try:
            # Get unique values, limited to sample size
            unique_values = series.dropna().unique()
            if len(unique_values) > self.sample_size:
                sample = np.random.choice(unique_values, self.sample_size, replace=False)
            else:
                sample = unique_values
            
            # Convert to Python native types for JSON serialization
            return [self._to_serializable(val) for val in sample[:50]]  # Limit to 50 for display
        except Exception:
            return []
    
    def _to_serializable(self, value: Any) -> Any:
        """Convert value to JSON serializable format"""
        if pd.isna(value):
            return None
        elif isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat()
        else:
            return str(value)
    
    def _calculate_schema_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of schema (column names and types)"""
        schema_str = json.dumps({
            'columns': list(df.columns),
            'types': {col: str(df[col].dtype) for col in df.columns}
        }, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def _calculate_structure_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of data structure"""
        structure_info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'index_type': str(type(df.index).__name__)
        }
        structure_str = json.dumps(structure_info, sort_keys=True)
        return hashlib.md5(structure_str.encode()).hexdigest()
    
    def _calculate_content_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of sample content"""
        try:
            # Sample data for content hash
            sample_size = min(self.sample_size, len(df))
            if sample_size == 0:
                return hashlib.md5(b"empty").hexdigest()
            
            sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
            
            # Create content signature
            content_info = []
            for col in sample_df.columns:
                col_sample = sample_df[col].dropna()
                if len(col_sample) > 0:
                    # Get statistical summary
                    if pd.api.types.is_numeric_dtype(col_sample):
                        content_info.append({
                            'column': col,
                            'type': 'numeric',
                            'mean': float(col_sample.mean()) if not col_sample.empty else 0,
                            'std': float(col_sample.std()) if not col_sample.empty else 0,
                            'unique_count': col_sample.nunique()
                        })
                    else:
                        content_info.append({
                            'column': col,
                            'type': 'categorical',
                            'unique_count': col_sample.nunique(),
                            'most_common': str(col_sample.mode().iloc[0]) if not col_sample.mode().empty else ""
                        })
            
            content_str = json.dumps(content_info, sort_keys=True)
            return hashlib.md5(content_str.encode()).hexdigest()
        except Exception:
            return hashlib.md5(b"error").hexdigest()
    
    def _calculate_stats_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of statistical summary"""
        try:
            stats_info = {
                'shape': df.shape,
                'null_counts': df.isnull().sum().to_dict(),
                'duplicate_count': int(df.duplicated().sum())
            }
            
            # Add numeric statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                numeric_stats = df[numeric_cols].describe().to_dict()
                stats_info['numeric_stats'] = numeric_stats
            
            stats_str = json.dumps(stats_info, sort_keys=True, default=str)
            return hashlib.md5(stats_str.encode()).hexdigest()
        except Exception:
            return hashlib.md5(b"stats_error").hexdigest()
    
    def _calculate_column_value_hash(self, series: pd.Series) -> str:
        """Calculate hash of column values"""
        try:
            # Use unique values for hash
            unique_vals = series.dropna().unique()
            if len(unique_vals) == 0:
                return hashlib.md5(b"empty_column").hexdigest()
            
            # Sort and hash
            if pd.api.types.is_numeric_dtype(series):
                sorted_vals = np.sort(unique_vals)
                val_str = str(sorted_vals.tolist())
            else:
                sorted_vals = sorted([str(v) for v in unique_vals])
                val_str = str(sorted_vals)
            
            return hashlib.md5(val_str.encode()).hexdigest()
        except Exception:
            return hashlib.md5(b"value_error").hexdigest()
    
    def _calculate_column_stats_hash(self, series: pd.Series) -> str:
        """Calculate hash of column statistics"""
        try:
            stats = {
                'count': series.count(),
                'null_count': series.isnull().sum(),
                'unique_count': series.nunique(),
                'dtype': str(series.dtype)
            }
            
            if pd.api.types.is_numeric_dtype(series):
                stats.update({
                    'mean': float(series.mean()) if not series.empty else 0,
                    'std': float(series.std()) if not series.empty else 0,
                    'min': float(series.min()) if not series.empty else 0,
                    'max': float(series.max()) if not series.empty else 0
                })
            
            stats_str = json.dumps(stats, sort_keys=True)
            return hashlib.md5(stats_str.encode()).hexdigest()
        except Exception:
            return hashlib.md5(b"column_stats_error").hexdigest()
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score (0-100)"""
        try:
            if df.empty:
                return 0.0
            
            # Factors for quality score
            total_cells = df.shape[0] * df.shape[1]
            null_cells = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
            
            # Calculate completeness (non-null percentage)
            completeness = ((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 0
            
            # Calculate uniqueness (non-duplicate percentage)
            uniqueness = ((df.shape[0] - duplicate_rows) / df.shape[0] * 100) if df.shape[0] > 0 else 0
            
            # Weighted average (completeness 70%, uniqueness 30%)
            quality_score = (completeness * 0.7) + (uniqueness * 0.3)
            
            return round(quality_score, 2)
        except Exception:
            return 0.0


class MetadataComparator:
    """Compare metadata between datasets to identify changes"""
    
    def compare_metadata(self, metadata1: DatasetMetadata, metadata2: DatasetMetadata) -> Dict[str, Any]:
        """Compare two dataset metadata objects"""
        
        comparison_result = {
            'summary': {},
            'changes': [],
            'recommendations': [],
            'change_score': 0,
            'comparison_timestamp': datetime.now().isoformat()
        }
        
        # Basic comparison
        self._compare_basic_info(metadata1, metadata2, comparison_result)
        
        # Schema comparison
        self._compare_schema(metadata1, metadata2, comparison_result)
        
        # Quality comparison
        self._compare_quality_metrics(metadata1, metadata2, comparison_result)
        
        # Column-level comparison
        self._compare_columns(metadata1, metadata2, comparison_result)
        
        # Hash-based change detection
        self._compare_hashes(metadata1, metadata2, comparison_result)
        
        # Generate recommendations
        self._generate_metadata_recommendations(comparison_result)
        
        # Calculate overall change score
        comparison_result['change_score'] = self._calculate_change_score(comparison_result)
        
        return comparison_result
    
    def _compare_basic_info(self, meta1: DatasetMetadata, meta2: DatasetMetadata, result: Dict):
        """Compare basic dataset information"""
        result['summary'] = {
            'dataset1': {
                'name': meta1.name,
                'source': meta1.source_type,
                'rows': meta1.row_count,
                'columns': meta1.column_count,
                'created': meta1.creation_time.isoformat() if isinstance(meta1.creation_time, datetime) else str(meta1.creation_time)
            },
            'dataset2': {
                'name': meta2.name,
                'source': meta2.source_type,
                'rows': meta2.row_count,
                'columns': meta2.column_count,
                'created': meta2.creation_time.isoformat() if isinstance(meta2.creation_time, datetime) else str(meta2.creation_time)
            },
            'differences': {
                'row_diff': meta2.row_count - meta1.row_count,
                'column_diff': meta2.column_count - meta1.column_count,
                'row_diff_pct': ((meta2.row_count - meta1.row_count) / meta1.row_count * 100) if meta1.row_count > 0 else 0,
                'memory_diff': meta2.memory_usage - meta1.memory_usage
            }
        }
        
        # Flag significant changes
        if abs(result['summary']['differences']['row_diff_pct']) > 5:
            result['changes'].append({
                'type': 'row_count_change',
                'severity': 'high' if abs(result['summary']['differences']['row_diff_pct']) > 20 else 'medium',
                'description': f"Row count changed by {result['summary']['differences']['row_diff_pct']:.1f}%",
                'details': {
                    'previous': meta1.row_count,
                    'current': meta2.row_count,
                    'change': result['summary']['differences']['row_diff']
                }
            })
    
    def _compare_schema(self, meta1: DatasetMetadata, meta2: DatasetMetadata, result: Dict):
        """Compare schema changes"""
        # Column presence changes
        cols1 = set(meta1.column_names)
        cols2 = set(meta2.column_names)
        
        added_columns = cols2 - cols1
        removed_columns = cols1 - cols2
        common_columns = cols1 & cols2
        
        if added_columns:
            result['changes'].append({
                'type': 'columns_added',
                'severity': 'medium',
                'description': f"Added {len(added_columns)} new columns",
                'details': {'columns': list(added_columns)}
            })
        
        if removed_columns:
            result['changes'].append({
                'type': 'columns_removed',
                'severity': 'high',
                'description': f"Removed {len(removed_columns)} columns",
                'details': {'columns': list(removed_columns)}
            })
        
        # Data type changes
        type_changes = []
        for col in common_columns:
            if meta1.column_types[col] != meta2.column_types[col]:
                type_changes.append({
                    'column': col,
                    'previous_type': meta1.column_types[col],
                    'current_type': meta2.column_types[col]
                })
        
        if type_changes:
            result['changes'].append({
                'type': 'data_type_changes',
                'severity': 'high',
                'description': f"Data type changes in {len(type_changes)} columns",
                'details': {'changes': type_changes}
            })
        
        # Schema hash comparison
        if meta1.schema_hash != meta2.schema_hash:
            result['changes'].append({
                'type': 'schema_structure_change',
                'severity': 'medium',
                'description': "Overall schema structure has changed",
                'details': {
                    'previous_hash': meta1.schema_hash,
                    'current_hash': meta2.schema_hash
                }
            })
    
    def _compare_quality_metrics(self, meta1: DatasetMetadata, meta2: DatasetMetadata, result: Dict):
        """Compare data quality metrics"""
        quality_diff = meta2.data_quality_score - meta1.data_quality_score
        
        if abs(quality_diff) > 5:
            severity = 'high' if abs(quality_diff) > 15 else 'medium'
            direction = 'improved' if quality_diff > 0 else 'degraded'
            
            result['changes'].append({
                'type': 'quality_score_change',
                'severity': severity,
                'description': f"Data quality {direction} by {abs(quality_diff):.1f} points",
                'details': {
                    'previous_score': meta1.data_quality_score,
                    'current_score': meta2.data_quality_score,
                    'change': quality_diff
                }
            })
        
        # Null count changes
        null_diff = meta2.total_null_count - meta1.total_null_count
        if null_diff != 0:
            result['changes'].append({
                'type': 'null_count_change',
                'severity': 'low' if abs(null_diff) < 100 else 'medium',
                'description': f"Total null values changed by {null_diff}",
                'details': {
                    'previous_nulls': meta1.total_null_count,
                    'current_nulls': meta2.total_null_count,
                    'change': null_diff
                }
            })
        
        # Duplicate count changes
        dup_diff = meta2.total_duplicate_rows - meta1.total_duplicate_rows
        if dup_diff != 0:
            result['changes'].append({
                'type': 'duplicate_count_change',
                'severity': 'low' if abs(dup_diff) < 10 else 'medium',
                'description': f"Duplicate rows changed by {dup_diff}",
                'details': {
                    'previous_duplicates': meta1.total_duplicate_rows,
                    'current_duplicates': meta2.total_duplicate_rows,
                    'change': dup_diff
                }
            })
    
    def _compare_columns(self, meta1: DatasetMetadata, meta2: DatasetMetadata, result: Dict):
        """Compare column-level metadata"""
        common_columns = set(meta1.column_names) & set(meta2.column_names)
        
        column_changes = []
        
        for col in common_columns:
            col_meta1 = meta1.columns[col]
            col_meta2 = meta2.columns[col]
            
            # Value hash comparison (fast content change detection)
            if col_meta1.value_hash != col_meta2.value_hash:
                column_changes.append({
                    'column': col,
                    'change_type': 'content_changed',
                    'description': f"Content in column '{col}' has changed",
                    'details': {
                        'unique_count_change': col_meta2.unique_count - col_meta1.unique_count,
                        'null_percentage_change': col_meta2.null_percentage - col_meta1.null_percentage
                    }
                })
            
            # Statistical changes for numeric columns
            if col_meta1.mean_value is not None and col_meta2.mean_value is not None:
                mean_change_pct = abs((col_meta2.mean_value - col_meta1.mean_value) / col_meta1.mean_value * 100) if col_meta1.mean_value != 0 else 0
                if mean_change_pct > 10:
                    column_changes.append({
                        'column': col,
                        'change_type': 'statistical_change',
                        'description': f"Mean value in '{col}' changed by {mean_change_pct:.1f}%",
                        'details': {
                            'previous_mean': col_meta1.mean_value,
                            'current_mean': col_meta2.mean_value,
                            'change_pct': mean_change_pct
                        }
                    })
            
            # Significant unique count changes
            unique_change_pct = abs((col_meta2.unique_count - col_meta1.unique_count) / col_meta1.unique_count * 100) if col_meta1.unique_count > 0 else 0
            if unique_change_pct > 20:
                column_changes.append({
                    'column': col,
                    'change_type': 'cardinality_change',
                    'description': f"Unique values in '{col}' changed by {unique_change_pct:.1f}%",
                    'details': {
                        'previous_unique': col_meta1.unique_count,
                        'current_unique': col_meta2.unique_count,
                        'change_pct': unique_change_pct
                    }
                })
        
        if column_changes:
            result['changes'].append({
                'type': 'column_level_changes',
                'severity': 'medium',
                'description': f"Changes detected in {len(column_changes)} columns",
                'details': {'column_changes': column_changes}
            })
    
    def _compare_hashes(self, meta1: DatasetMetadata, meta2: DatasetMetadata, result: Dict):
        """Compare hash-based fingerprints for change detection"""
        hash_changes = []
        
        # Structure hash
        if meta1.structure_hash != meta2.structure_hash:
            hash_changes.append({
                'hash_type': 'structure',
                'description': 'Dataset structure has changed',
                'impact': 'Schema or indexing changes detected'
            })
        
        # Content hash
        if meta1.content_hash != meta2.content_hash:
            hash_changes.append({
                'hash_type': 'content',
                'description': 'Dataset content has changed',
                'impact': 'Data values have been modified'
            })
        
        # Stats hash
        if meta1.stats_hash != meta2.stats_hash:
            hash_changes.append({
                'hash_type': 'statistics',
                'description': 'Statistical properties have changed',
                'impact': 'Data distribution or quality metrics changed'
            })
        
        if hash_changes:
            result['changes'].append({
                'type': 'fingerprint_changes',
                'severity': 'high',
                'description': f"Hash-based change detection: {len(hash_changes)} changes",
                'details': {'hash_changes': hash_changes}
            })
    
    def _generate_metadata_recommendations(self, result: Dict):
        """Generate recommendations based on detected changes"""
        recommendations = []
        
        # Analyze change patterns
        change_types = [change['type'] for change in result['changes']]
        
        if 'columns_removed' in change_types:
            recommendations.append("🚨 Critical: Column removal detected - verify ETL pipeline integrity")
            recommendations.append("📋 Review data source schema changes and update mappings")
        
        if 'data_type_changes' in change_types:
            recommendations.append("⚠️ Data type changes detected - validate downstream processes")
            recommendations.append("🔧 Update data validation rules to handle type changes")
        
        if 'quality_score_change' in change_types:
            quality_changes = [c for c in result['changes'] if c['type'] == 'quality_score_change']
            for qc in quality_changes:
                if qc['details']['change'] < 0:
                    recommendations.append("📉 Data quality degradation detected - investigate data sources")
                else:
                    recommendations.append("📈 Data quality improvement detected - document successful changes")
        
        if 'fingerprint_changes' in change_types:
            recommendations.append("🔍 Significant data changes detected - perform detailed audit")
            recommendations.append("📊 Consider implementing change tracking for critical datasets")
        
        if 'row_count_change' in change_types:
            recommendations.append("📦 Row count changes - verify data loading completeness")
            recommendations.append("⏰ Check if changes align with expected data refresh cycles")
        
        # General recommendations
        if len(result['changes']) > 5:
            recommendations.append("🔄 Multiple changes detected - consider comprehensive data validation")
        
        if not result['changes']:
            recommendations.append("✅ No significant changes detected - datasets appear consistent")
        
        result['recommendations'] = recommendations
    
    def _calculate_change_score(self, result: Dict) -> float:
        """Calculate overall change score (0-100, higher means more changes)"""
        if not result['changes']:
            return 0.0
        
        severity_weights = {'low': 1, 'medium': 3, 'high': 5}
        total_score = 0
        
        for change in result['changes']:
            weight = severity_weights.get(change['severity'], 1)
            total_score += weight
        
        # Normalize to 0-100 scale
        max_possible_score = len(result['changes']) * 5  # If all were high severity
        normalized_score = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
        
        return min(100.0, normalized_score)


def extract_dataset_metadata(df: pd.DataFrame, name: str, source_type: str, 
                           source_metadata: Optional[Dict[str, Any]] = None) -> DatasetMetadata:
    """Convenience function to extract metadata from a dataset"""
    extractor = MetadataExtractor()
    return extractor.extract_metadata(df, name, source_type, source_metadata)


def compare_dataset_metadata(metadata1: DatasetMetadata, metadata2: DatasetMetadata) -> Dict[str, Any]:
    """Convenience function to compare two metadata objects"""
    comparator = MetadataComparator()
    return comparator.compare_metadata(metadata1, metadata2)
