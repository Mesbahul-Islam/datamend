#!/usr/bin/env python3
"""
Test script to verify ydata-profiling integration
"""

import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import time

def test_ydata_profiling():
    """Test ydata-profiling with sample data"""
    
    print("Testing ydata-profiling integration...")
    
    # Create sample data
    np.random.seed(42)
    n_rows = 1000
    
    data = {
        'id': range(1, n_rows + 1),
        'name': [f'Person_{i}' for i in range(1, n_rows + 1)],
        'age': np.random.randint(18, 80, n_rows),
        'salary': np.random.normal(50000, 15000, n_rows),
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_rows),
        'score': np.random.uniform(0, 100, n_rows),
        'missing_data': np.where(np.random.random(n_rows) < 0.1, None, np.random.normal(10, 3, n_rows))
    }
    
    df = pd.DataFrame(data)
    
    print(f"Created sample dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Dataset shape: {df.shape}")
    
    # Test ydata-profiling
    print("\nGenerating profile report...")
    start_time = time.time()
    
    profile = ProfileReport(
        df, 
        title="Test Profile Report",
        explorative=True,
        minimal=False
    )
    
    end_time = time.time()
    profiling_time = end_time - start_time
    
    print(f"Profiling completed in {profiling_time:.2f} seconds")
    
    # Test basic profile access
    description = profile.description_set
    print(f"\nBasic statistics:")
    print(f"- Variables: {description.variables}")
    print(f"- Observations: {description.observations:,}")
    print(f"- Missing cells: {description.missing:,}")
    print(f"- Total cells: {description.cells:,}")
    print(f"- Duplicate rows: {description.duplicates}")
    
    types_summary = description.types
    print(f"- Variable types: {types_summary}")
    
    print("\n✅ ydata-profiling integration test completed successfully!")
    
    return profile

if __name__ == "__main__":
    test_ydata_profiling()
