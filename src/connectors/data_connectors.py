"""
Data Connectors Module

This module provides simple data connectors for basic data sources:
- CSV files
- Excel files
"""

import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os

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


class DataConnectorFactory:
    """Factory class for creating data connectors"""
    
    @staticmethod
    def create_connector(source_type: str, **kwargs) -> DataConnector:
        """
        Create a data connector based on the source type.
        
        Args:
            source_type: Type of data source ('csv', 'excel')
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
        
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    @staticmethod
    def get_supported_types() -> List[str]:
        """Get list of supported data source types"""
        return ['csv', 'excel']
