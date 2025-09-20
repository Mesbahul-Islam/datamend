"""
Data Connectors Module

This module provides simple data connectors for basic data sources:
- CSV files
- Excel files
- Snowflake database
"""

import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os

try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import pd_writer
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DataSourceInfo:
    """Information about a data source"""
    source_type: str
    file_path: Optional[str] = None
    row_count: int = 0
    column_count: int = 0
    columns: List[str] = None

    def __post_init__(self):
        if self.columns is None:
            self.columns = []


class DataConnector(ABC):
    """Abstract base class for data connectors"""
    
    def __init__(self):
        self.is_connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the data source"""
        pass
    
    @abstractmethod
    def get_data(self, query: str = None, limit: int = None) -> pd.DataFrame:
        """Get data from the source"""
        pass
    
    @abstractmethod
    def get_tables(self) -> List[str]:
        """Get list of available tables/sheets"""
        pass
    
    def disconnect(self):
        """Disconnect from the data source"""
        self.is_connected = False
    
    def get_info(self) -> DataSourceInfo:
        """Get information about the data source"""
        return DataSourceInfo(source_type="unknown")

class OracleCloudConnector(DataConnector):
    """Oracle Cloud Database Connector"""
    def __init__(self, username: str, password: str, dsn: str, 
                 wallet_location: str = None):
        super().__init__()
        self.username = username
        self.password = password
        self.dsn = dsn
        self.wallet_location = wallet_location
        self.connection = None
        
    def connect(self) -> bool:
        """Connect to Oracle Cloud"""
        try:
            import oracledb
            if self.wallet_location:
                oracledb.init_oracle_client(config_dir=self.wallet_location)
            
            self.connection = oracledb.connect(
                user=self.username,
                password=self.password,
                dsn=self.dsn
            )
            return True
        except Exception as e:
            logger.error(f"Oracle connection failed: {str(e)}")
            return False
            
    def get_data(self, query: str = None, limit: int = None) -> pd.DataFrame:
        """Execute query and return results"""
        try:
            if query is None:
                return pd.DataFrame()
                
            if limit:
                query = self._add_row_limit(query, limit)
                
            return pd.read_sql(query, self.connection)
        except Exception as e:
            logger.error(f"Oracle query failed: {str(e)}")
            return pd.DataFrame()
            
    def get_tables(self) -> List[str]:
        """Get list of accessible tables"""
        query = """
        SELECT table_name 
        FROM user_tables 
        ORDER BY table_name
        """
        df = self.get_data(query)
        return df['TABLE_NAME'].tolist() if not df.empty else []

    def close(self):
        """Close Oracle connection"""
        if self.connection:
            self.connection.close()

class CSVConnector(DataConnector):
    """CSV File Connector"""
    
    def __init__(self, file_path: str, encoding: str = 'utf-8', delimiter: str = ','):
        super().__init__()
        self.file_path = file_path
        self.encoding = encoding
        self.delimiter = delimiter
        logger.info(f"CSV Connector initialized for file: {file_path}")
    
    def connect(self) -> bool:
        """Check if file exists and is readable"""
        try:
            if not os.path.exists(self.file_path):
                logger.error(f"File not found: {self.file_path}")
                return False
            
            # Try to read a small sample to validate format
            sample = pd.read_csv(self.file_path, nrows=5, encoding=self.encoding, delimiter=self.delimiter)
            self.is_connected = True
            logger.info(f"Successfully connected to CSV file: {self.file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to CSV file: {str(e)}")
            return False
    
    def get_data(self, query: str = None, limit: int = None) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            query: Not used for CSV files
            limit: Maximum number of rows to read
            
        Returns:
            DataFrame with the data
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to data source")
        
        try:
            # Read CSV with optional row limit
            df = pd.read_csv(
                self.file_path, 
                encoding=self.encoding, 
                delimiter=self.delimiter,
                nrows=limit
            )
            logger.info(f"Loaded {len(df)} rows from CSV file")
            return df
            
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            raise
    
    def get_tables(self) -> List[str]:
        """Return the filename as the 'table' name"""
        return [os.path.basename(self.file_path)]
    
    def get_info(self) -> DataSourceInfo:
        """Get information about the CSV file"""
        if not self.is_connected:
            return DataSourceInfo(source_type="csv", file_path=self.file_path)
        
        try:
            # Read just the header to get column info
            df_sample = pd.read_csv(self.file_path, nrows=0, encoding=self.encoding, delimiter=self.delimiter)
            
            # Get total row count (excluding header)
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                row_count = sum(1 for line in f) - 1  # Subtract header row
            
            return DataSourceInfo(
                source_type="csv",
                file_path=self.file_path,
                row_count=row_count,
                column_count=len(df_sample.columns),
                columns=df_sample.columns.tolist()
            )
        except Exception as e:
            logger.error(f"Error getting CSV info: {str(e)}")
            return DataSourceInfo(source_type="csv", file_path=self.file_path)


class ExcelConnector(DataConnector):
    """Excel File Connector"""
    
    def __init__(self, file_path: str, sheet_name: str = None):
        super().__init__()
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.available_sheets = []
        logger.info(f"Excel Connector initialized for file: {file_path}")
    
    def connect(self) -> bool:
        """Check if file exists and is readable"""
        try:
            if not os.path.exists(self.file_path):
                logger.error(f"File not found: {self.file_path}")
                return False
            
            # Get available sheets
            excel_file = pd.ExcelFile(self.file_path)
            self.available_sheets = excel_file.sheet_names
            
            # If no sheet specified, use the first one
            if not self.sheet_name and self.available_sheets:
                self.sheet_name = self.available_sheets[0]
            
            # Try to read a small sample to validate format
            sample = pd.read_excel(self.file_path, sheet_name=self.sheet_name, nrows=5)
            self.is_connected = True
            logger.info(f"Successfully connected to Excel file: {self.file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Excel file: {str(e)}")
            return False
    
    def get_data(self, query: str = None, limit: int = None) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Args:
            query: Not used for Excel files
            limit: Maximum number of rows to read
            
        Returns:
            DataFrame with the data
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to data source")
        
        try:
            # Read Excel with optional row limit
            df = pd.read_excel(
                self.file_path, 
                sheet_name=self.sheet_name,
                nrows=limit
            )
            logger.info(f"Loaded {len(df)} rows from Excel file")
            return df
            
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            raise
    
    def get_tables(self) -> List[str]:
        """Return available sheet names"""
        return self.available_sheets
    
    def get_info(self) -> DataSourceInfo:
        """Get information about the Excel file"""
        if not self.is_connected:
            return DataSourceInfo(source_type="excel", file_path=self.file_path)
        
        try:
            # Read just the header to get column info
            df_sample = pd.read_excel(self.file_path, sheet_name=self.sheet_name, nrows=0)
            
            # Get total row count
            df_full = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            row_count = len(df_full)
            
            return DataSourceInfo(
                source_type="excel",
                file_path=self.file_path,
                row_count=row_count,
                column_count=len(df_sample.columns),
                columns=df_sample.columns.tolist()
            )
        except Exception as e:
            logger.error(f"Error getting Excel info: {str(e)}")
            return DataSourceInfo(source_type="excel", file_path=self.file_path)


class SnowflakeConnector(DataConnector):
    """Snowflake Database Connector"""
    
    def __init__(self, account: str, username: str, password: str, 
                 warehouse: str, database: str, schema: str = "PUBLIC"):
        super().__init__()
        if not SNOWFLAKE_AVAILABLE:
            raise ImportError("Snowflake connector not available. Install snowflake-connector-python")
        
        self.account = account
        self.username = username
        self.password = password
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.connection = None
        self.tables_cache = None
        logger.info(f"Snowflake Connector initialized for account: {account}")
    
    def connect(self) -> bool:
        """Connect to Snowflake"""
        try:
            self.connection = snowflake.connector.connect(
                account=self.account,
                user=self.username,
                password=self.password,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema
            )
            
            # Test connection
            cursor = self.connection.cursor()
            cursor.execute("SELECT CURRENT_VERSION()")
            version = cursor.fetchone()
            cursor.close()
            
            self.is_connected = True
            logger.info(f"Successfully connected to Snowflake. Version: {version[0]}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            self.is_connected = False
            return False
    
    def get_data(self, query: str = None, limit: int = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query to execute
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with query results
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Snowflake")
        
        if not query:
            raise ValueError("Query is required for Snowflake connector")
        
        try:
            # Add LIMIT clause if specified
            if limit and "LIMIT" not in query.upper():
                query = f"{query.rstrip(';')} LIMIT {limit}"
            
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # Fetch column names
            columns = [desc[0] for desc in cursor.description]
            
            # Fetch data
            data = cursor.fetchall()
            cursor.close()
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=columns)
            logger.info(f"Loaded {len(df)} rows from Snowflake query")
            return df
            
        except Exception as e:
            logger.error(f"Error executing Snowflake query: {str(e)}")
            raise
    
    def get_tables(self) -> List[str]:
        """Get list of available tables in the current schema"""
        if not self.is_connected:
            return []
        
        if self.tables_cache is not None:
            return self.tables_cache
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = '{self.schema}' 
                AND TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
            self.tables_cache = tables
            logger.info(f"Found {len(tables)} tables in schema {self.schema}")
            return tables
            
        except Exception as e:
            logger.error(f"Error getting Snowflake tables: {str(e)}")
            return []
    
    def get_info(self) -> DataSourceInfo:
        """Get information about the Snowflake connection"""
        return DataSourceInfo(
            source_type="snowflake",
            file_path=f"{self.account}/{self.database}/{self.schema}"
        )
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific table"""
        if not self.is_connected:
            return {}
        
        try:
            cursor = self.connection.cursor()
            
            # Get column information
            cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = '{self.schema}' 
                AND TABLE_NAME = '{table_name}'
                ORDER BY ORDINAL_POSITION
            """)
            
            columns_info = cursor.fetchall()
            
            # Get row count
            cursor.execute(f'SELECT COUNT(*) FROM "{self.schema}"."{table_name}"')
            row_count = cursor.fetchone()[0]
            
            cursor.close()
            
            return {
                'table_name': table_name,
                'row_count': row_count,
                'column_count': len(columns_info),
                'columns': [col[0] for col in columns_info],
                'column_types': {col[0]: col[1] for col in columns_info}
            }
            
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {str(e)}")
            return {}
    
    def close(self):
        """Close the Snowflake connection"""
        if self.connection:
            self.connection.close()
            self.is_connected = False
            logger.info("Snowflake connection closed")


class DataConnectorFactory:
    """Factory class for creating data connectors"""
    
    @staticmethod
    def create_connector(source_type: str, **kwargs) -> DataConnector:
        """
        Create a data connector based on the source type.
        
        Args:
            source_type: Type of data source ('csv', 'excel', 'snowflake')
            **kwargs: Additional parameters for the connector
            
        Returns:
            DataConnector instance
        """
        source_type = source_type.lower()
        
        if source_type == 'csv':
            file_path = kwargs.get('file_path')
            if not file_path:
                raise ValueError("file_path is required for CSV connector")
            return CSVConnector(
                file_path=file_path,
                encoding=kwargs.get('encoding', 'utf-8'),
                delimiter=kwargs.get('delimiter', ',')
            )
        
        elif source_type == 'excel':
            file_path = kwargs.get('file_path')
            if not file_path:
                raise ValueError("file_path is required for Excel connector")
            return ExcelConnector(
                file_path=file_path,
                sheet_name=kwargs.get('sheet_name')
            )
        
        elif source_type == 'snowflake':
            required_params = ['account', 'username', 'password', 'warehouse', 'database']
            for param in required_params:
                if not kwargs.get(param):
                    raise ValueError(f"{param} is required for Snowflake connector")
            
            return SnowflakeConnector(
                account=kwargs['account'],
                username=kwargs['username'],
                password=kwargs['password'],
                warehouse=kwargs['warehouse'],
                database=kwargs['database'],
                schema=kwargs.get('schema', 'PUBLIC')
            )
        
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    @staticmethod
    def get_supported_types() -> List[str]:
        """Get list of supported data source types"""
        types = ['csv', 'excel']
        if SNOWFLAKE_AVAILABLE:
            types.append('snowflake')
        return types
