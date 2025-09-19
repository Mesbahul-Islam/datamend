"""
Database Connectors for Distributed Data Platforms

This module provides connectors for various data platforms:
- Snowflake
- Hadoop (via Hive/Spark)
- Oracle
- PostgreSQL
- MySQL
- Other JDBC-compatible databases
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import warnings
from datetime import datetime
import hashlib

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for database connections"""
    platform_type: str  # snowflake, oracle, hadoop, postgres, etc.
    connection_string: str
    username: str
    password: str
    database_name: str
    schema_name: Optional[str] = None
    warehouse: Optional[str] = None  # For Snowflake
    role: Optional[str] = None  # For Snowflake
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class TableMetadata:
    """Metadata about a table across platforms"""
    platform: str
    database: str
    schema: str
    table_name: str
    column_count: int
    row_count: int
    columns: List[str]
    column_types: Dict[str, str]
    primary_keys: List[str]
    indexes: List[str]
    last_updated: datetime
    data_size_mb: float


@dataclass
class DataConsistencyResult:
    """Results of cross-platform consistency check"""
    table_name: str
    platforms_compared: List[str]
    is_consistent: bool
    inconsistencies: List[Dict[str, Any]]
    row_count_differences: Dict[str, int]
    schema_differences: Dict[str, Any]
    data_differences: Dict[str, Any]
    consistency_score: float  # 0-100
    warnings: List[str]


class BaseDatabaseConnector(ABC):
    """Abstract base class for database connectors"""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connection = None
        self.platform_type = config.platform_type
        
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close database connection"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame"""
        pass
    
    @abstractmethod
    def get_table_metadata(self, table_name: str, schema: Optional[str] = None) -> TableMetadata:
        """Get metadata about a table"""
        pass
    
    @abstractmethod
    def get_table_sample(self, table_name: str, sample_size: int = 1000, schema: Optional[str] = None) -> pd.DataFrame:
        """Get a sample of data from a table"""
        pass
    
    def get_data_fingerprint(self, table_name: str, columns: List[str], schema: Optional[str] = None) -> str:
        """Generate a fingerprint of the data for consistency checking"""
        try:
            # Get a deterministic sample
            query = self._build_fingerprint_query(table_name, columns, schema)
            df = self.execute_query(query)
            
            # Create a hash of the data structure and sample content
            fingerprint_data = {
                'columns': sorted(columns),
                'row_count': len(df),
                'data_types': {col: str(df[col].dtype) for col in df.columns},
                'sample_hash': hashlib.md5(df.to_string().encode()).hexdigest()
            }
            
            return hashlib.sha256(str(fingerprint_data).encode()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to generate fingerprint for {table_name}: {str(e)}")
            return ""
    
    @abstractmethod
    def _build_fingerprint_query(self, table_name: str, columns: List[str], schema: Optional[str] = None) -> str:
        """Build a platform-specific query for data fingerprinting"""
        pass


class SnowflakeConnector(BaseDatabaseConnector):
    """Connector for Snowflake data warehouse"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.warehouse = config.warehouse
        self.role = config.role
    
    def connect(self) -> bool:
        """Establish connection to Snowflake"""
        try:
            import snowflake.connector
            
            self.connection = snowflake.connector.connect(
                user=self.config.username,
                password=self.config.password,
                account=self.config.connection_string,
                warehouse=self.warehouse,
                database=self.config.database_name,
                schema=self.config.schema_name,
                role=self.role
            )
            logger.info(f"Connected to Snowflake: {self.config.database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            return False
    
    def disconnect(self):
        """Close Snowflake connection"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from Snowflake")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute query on Snowflake"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # Fetch results and column names
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            df = pd.DataFrame(results, columns=columns)
            cursor.close()
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return pd.DataFrame()
    
    def get_table_metadata(self, table_name: str, schema: Optional[str] = None) -> TableMetadata:
        """Get Snowflake table metadata"""
        schema = schema or self.config.schema_name
        
        # Query for table information
        info_query = f"""
        SELECT 
            COUNT(*) as row_count,
            COUNT(DISTINCT column_name) as column_count
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE table_name = '{table_name.upper()}'
        AND table_schema = '{schema.upper()}'
        """
        
        columns_query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE table_name = '{table_name.upper()}'
        AND table_schema = '{schema.upper()}'
        ORDER BY ordinal_position
        """
        
        try:
            info_df = self.execute_query(info_query)
            columns_df = self.execute_query(columns_query)
            
            # Get actual row count
            row_count_query = f"SELECT COUNT(*) as row_count FROM {schema}.{table_name}"
            row_count_df = self.execute_query(row_count_query)
            actual_row_count = row_count_df.iloc[0]['row_count'] if not row_count_df.empty else 0
            
            columns = columns_df['COLUMN_NAME'].tolist()
            column_types = dict(zip(columns_df['COLUMN_NAME'], columns_df['DATA_TYPE']))
            
            return TableMetadata(
                platform="snowflake",
                database=self.config.database_name,
                schema=schema,
                table_name=table_name,
                column_count=len(columns),
                row_count=actual_row_count,
                columns=columns,
                column_types=column_types,
                primary_keys=[],  # Would need additional query
                indexes=[],  # Would need additional query
                last_updated=datetime.now(),
                data_size_mb=0.0  # Would need additional query
            )
        except Exception as e:
            logger.error(f"Failed to get metadata for {table_name}: {str(e)}")
            return None
    
    def get_table_sample(self, table_name: str, sample_size: int = 1000, schema: Optional[str] = None) -> pd.DataFrame:
        """Get sample data from Snowflake table"""
        schema = schema or self.config.schema_name
        query = f"SELECT * FROM {schema}.{table_name} SAMPLE ({sample_size} ROWS)"
        return self.execute_query(query)
    
    def _build_fingerprint_query(self, table_name: str, columns: List[str], schema: Optional[str] = None) -> str:
        """Build Snowflake-specific fingerprint query"""
        schema = schema or self.config.schema_name
        column_list = ", ".join(columns)
        return f"""
        SELECT {column_list}
        FROM {schema}.{table_name}
        SAMPLE (1000 ROWS)
        ORDER BY 1
        """


class OracleConnector(BaseDatabaseConnector):
    """Connector for Oracle database"""
    
    def connect(self) -> bool:
        """Establish connection to Oracle"""
        try:
            import cx_Oracle
            
            self.connection = cx_Oracle.connect(
                f"{self.config.username}/{self.config.password}@{self.config.connection_string}"
            )
            logger.info(f"Connected to Oracle: {self.config.database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Oracle: {str(e)}")
            return False
    
    def disconnect(self):
        """Close Oracle connection"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from Oracle")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute query on Oracle"""
        try:
            return pd.read_sql(query, self.connection)
        except Exception as e:
            logger.error(f"Oracle query execution failed: {str(e)}")
            return pd.DataFrame()
    
    def get_table_metadata(self, table_name: str, schema: Optional[str] = None) -> TableMetadata:
        """Get Oracle table metadata"""
        schema = schema or self.config.username.upper()
        
        columns_query = f"""
        SELECT 
            column_name,
            data_type,
            nullable
        FROM all_tab_columns 
        WHERE table_name = '{table_name.upper()}'
        AND owner = '{schema}'
        ORDER BY column_id
        """
        
        try:
            columns_df = self.execute_query(columns_query)
            
            # Get row count
            row_count_query = f"SELECT COUNT(*) as row_count FROM {schema}.{table_name}"
            row_count_df = self.execute_query(row_count_query)
            actual_row_count = row_count_df.iloc[0]['row_count'] if not row_count_df.empty else 0
            
            columns = columns_df['COLUMN_NAME'].tolist()
            column_types = dict(zip(columns_df['COLUMN_NAME'], columns_df['DATA_TYPE']))
            
            return TableMetadata(
                platform="oracle",
                database=self.config.database_name,
                schema=schema,
                table_name=table_name,
                column_count=len(columns),
                row_count=actual_row_count,
                columns=columns,
                column_types=column_types,
                primary_keys=[],
                indexes=[],
                last_updated=datetime.now(),
                data_size_mb=0.0
            )
        except Exception as e:
            logger.error(f"Failed to get Oracle metadata for {table_name}: {str(e)}")
            return None
    
    def get_table_sample(self, table_name: str, sample_size: int = 1000, schema: Optional[str] = None) -> pd.DataFrame:
        """Get sample data from Oracle table"""
        schema = schema or self.config.username.upper()
        query = f"SELECT * FROM (SELECT * FROM {schema}.{table_name} ORDER BY DBMS_RANDOM.VALUE) WHERE ROWNUM <= {sample_size}"
        return self.execute_query(query)
    
    def _build_fingerprint_query(self, table_name: str, columns: List[str], schema: Optional[str] = None) -> str:
        """Build Oracle-specific fingerprint query"""
        schema = schema or self.config.username.upper()
        column_list = ", ".join(columns)
        return f"""
        SELECT {column_list}
        FROM (
            SELECT {column_list}
            FROM {schema}.{table_name}
            ORDER BY DBMS_RANDOM.VALUE
        )
        WHERE ROWNUM <= 1000
        """


class HadoopConnector(BaseDatabaseConnector):
    """Connector for Hadoop (via Hive/Spark)"""
    
    def connect(self) -> bool:
        """Establish connection to Hadoop/Hive"""
        try:
            from pyhive import hive
            
            self.connection = hive.Connection(
                host=self.config.connection_string,
                port=10000,
                username=self.config.username,
                database=self.config.database_name
            )
            logger.info(f"Connected to Hadoop/Hive: {self.config.database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Hadoop/Hive: {str(e)}")
            return False
    
    def disconnect(self):
        """Close Hadoop connection"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from Hadoop/Hive")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute query on Hadoop/Hive"""
        try:
            return pd.read_sql(query, self.connection)
        except Exception as e:
            logger.error(f"Hadoop query execution failed: {str(e)}")
            return pd.DataFrame()
    
    def get_table_metadata(self, table_name: str, schema: Optional[str] = None) -> TableMetadata:
        """Get Hadoop table metadata"""
        try:
            # Use DESCRIBE to get column information
            describe_query = f"DESCRIBE {table_name}"
            columns_df = self.execute_query(describe_query)
            
            # Get row count
            row_count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            row_count_df = self.execute_query(row_count_query)
            actual_row_count = row_count_df.iloc[0]['row_count'] if not row_count_df.empty else 0
            
            columns = columns_df['col_name'].tolist()
            column_types = dict(zip(columns_df['col_name'], columns_df['data_type']))
            
            return TableMetadata(
                platform="hadoop",
                database=self.config.database_name,
                schema=schema or "default",
                table_name=table_name,
                column_count=len(columns),
                row_count=actual_row_count,
                columns=columns,
                column_types=column_types,
                primary_keys=[],
                indexes=[],
                last_updated=datetime.now(),
                data_size_mb=0.0
            )
        except Exception as e:
            logger.error(f"Failed to get Hadoop metadata for {table_name}: {str(e)}")
            return None
    
    def get_table_sample(self, table_name: str, sample_size: int = 1000, schema: Optional[str] = None) -> pd.DataFrame:
        """Get sample data from Hadoop table"""
        query = f"SELECT * FROM {table_name} TABLESAMPLE(BUCKET 1 OUT OF 100) LIMIT {sample_size}"
        return self.execute_query(query)
    
    def _build_fingerprint_query(self, table_name: str, columns: List[str], schema: Optional[str] = None) -> str:
        """Build Hadoop-specific fingerprint query"""
        column_list = ", ".join(columns)
        return f"""
        SELECT {column_list}
        FROM {table_name}
        TABLESAMPLE(BUCKET 1 OUT OF 100)
        LIMIT 1000
        """


class PostgreSQLConnector(BaseDatabaseConnector):
    """Connector for PostgreSQL"""
    
    def connect(self) -> bool:
        """Establish connection to PostgreSQL"""
        try:
            import psycopg2
            
            self.connection = psycopg2.connect(
                host=self.config.connection_string,
                database=self.config.database_name,
                user=self.config.username,
                password=self.config.password
            )
            logger.info(f"Connected to PostgreSQL: {self.config.database_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            return False
    
    def disconnect(self):
        """Close PostgreSQL connection"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from PostgreSQL")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute query on PostgreSQL"""
        try:
            return pd.read_sql(query, self.connection)
        except Exception as e:
            logger.error(f"PostgreSQL query execution failed: {str(e)}")
            return pd.DataFrame()
    
    def get_table_metadata(self, table_name: str, schema: Optional[str] = None) -> TableMetadata:
        """Get PostgreSQL table metadata"""
        schema = schema or 'public'
        
        columns_query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable
        FROM information_schema.columns 
        WHERE table_name = '{table_name}'
        AND table_schema = '{schema}'
        ORDER BY ordinal_position
        """
        
        try:
            columns_df = self.execute_query(columns_query)
            
            # Get row count
            row_count_query = f"SELECT COUNT(*) as row_count FROM {schema}.{table_name}"
            row_count_df = self.execute_query(row_count_query)
            actual_row_count = row_count_df.iloc[0]['row_count'] if not row_count_df.empty else 0
            
            columns = columns_df['column_name'].tolist()
            column_types = dict(zip(columns_df['column_name'], columns_df['data_type']))
            
            return TableMetadata(
                platform="postgresql",
                database=self.config.database_name,
                schema=schema,
                table_name=table_name,
                column_count=len(columns),
                row_count=actual_row_count,
                columns=columns,
                column_types=column_types,
                primary_keys=[],
                indexes=[],
                last_updated=datetime.now(),
                data_size_mb=0.0
            )
        except Exception as e:
            logger.error(f"Failed to get PostgreSQL metadata for {table_name}: {str(e)}")
            return None
    
    def get_table_sample(self, table_name: str, sample_size: int = 1000, schema: Optional[str] = None) -> pd.DataFrame:
        """Get sample data from PostgreSQL table"""
        schema = schema or 'public'
        query = f"SELECT * FROM {schema}.{table_name} TABLESAMPLE SYSTEM (10) LIMIT {sample_size}"
        return self.execute_query(query)
    
    def _build_fingerprint_query(self, table_name: str, columns: List[str], schema: Optional[str] = None) -> str:
        """Build PostgreSQL-specific fingerprint query"""
        schema = schema or 'public'
        column_list = ", ".join(columns)
        return f"""
        SELECT {column_list}
        FROM {schema}.{table_name}
        TABLESAMPLE SYSTEM (10)
        LIMIT 1000
        """


class DatabaseConnectorFactory:
    """Factory for creating database connectors"""
    
    _connectors = {
        'snowflake': SnowflakeConnector,
        'oracle': OracleConnector,
        'hadoop': HadoopConnector,
        'hive': HadoopConnector,
        'postgresql': PostgreSQLConnector,
        'postgres': PostgreSQLConnector,
    }
    
    @classmethod
    def create_connector(cls, config: ConnectionConfig) -> BaseDatabaseConnector:
        """Create a database connector based on platform type"""
        platform = config.platform_type.lower()
        
        if platform not in cls._connectors:
            raise ValueError(f"Unsupported platform: {platform}")
        
        connector_class = cls._connectors[platform]
        return connector_class(config)
    
    @classmethod
    def get_supported_platforms(cls) -> List[str]:
        """Get list of supported platforms"""
        return list(cls._connectors.keys())
