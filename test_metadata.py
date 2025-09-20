#!/usr/bin/env python3
"""
Test script to verify metadata extraction works correctly
"""

import pandas as pd
import numpy as np
from src.metadata.metadata_extractor import MetadataExtractor

def test_metadata_extraction():
    """Test that metadata extraction includes all required attributes"""
    
    # Create a test DataFrame with different column types
    test_data = {
        'numeric_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'string_col': ['A', 'B', 'C', 'D', 'E'],
        'bool_col': [True, False, True, False, True],
        'datetime_col': pd.date_range('2023-01-01', periods=5)
    }
    
    df = pd.DataFrame(test_data)
    
    # Add some null values
    df.loc[0, 'string_col'] = None
    df.loc[1, 'float_col'] = None
    
    print("Test DataFrame:")
    print(df)
    print(f"\nData types:")
    print(df.dtypes)
    
    # Extract metadata
    extractor = MetadataExtractor()
    metadata = extractor.extract_metadata(df, "test_dataset", "manual")
    
    print(f"\n=== Metadata Extraction Results ===")
    print(f"Name: {metadata.name}")
    print(f"Rows: {metadata.row_count}")
    print(f"Columns: {metadata.column_count}")
    print(f"Total nulls: {metadata.total_null_count}")
    
    print(f"\nColumn categorization:")
    print(f"Numeric columns: {metadata.numeric_columns}")
    print(f"Categorical columns: {metadata.categorical_columns}")
    print(f"DateTime columns: {metadata.datetime_columns}")
    
    print(f"\nColumn types: {metadata.column_types}")
    
    # Verify all required attributes exist
    required_attrs = [
        'name', 'source_type', 'row_count', 'column_count', 'creation_time',
        'column_names', 'column_types', 'numeric_columns', 'categorical_columns',
        'datetime_columns', 'total_null_count', 'columns', 'source_metadata'
    ]
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(metadata, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        print(f"\n❌ Missing attributes: {missing_attrs}")
        return False
    else:
        print(f"\n✅ All required attributes present!")
        return True

if __name__ == "__main__":
    success = test_metadata_extraction()
    if success:
        print("\n🎉 Metadata extraction test passed!")
    else:
        print("\n💥 Metadata extraction test failed!")
