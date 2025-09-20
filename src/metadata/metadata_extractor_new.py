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
