#!/usr/bin/env python3
"""
Simple test to verify the data quality engine works end-to-end
"""

import sys
import os
sys.path.append('/home/mesbahul/Documents/hackathon_2')

from src.data_quality.engine import DataQualityEngine
from src.connectors.data_connectors import DataConnectorFactory

def test_complete_pipeline():
    """Test the complete data quality pipeline"""
    print("🔍 Testing Data Quality Engine Pipeline")
    print("=" * 50)
    
    try:
        # Test CSV connector
        print("1. Testing CSV Connector...")
        connector = DataConnectorFactory.create_connector('csv', file_path='test_data.csv')
        
        if not connector.connect():
            print("❌ Failed to connect to test_data.csv")
            return False
            
        df = connector.get_data(limit=100)
        print(f"✅ Loaded {len(df)} rows from CSV")
        
        # Test data quality engine
        print("2. Testing Data Quality Engine...")
        engine = DataQualityEngine(chunk_size=500, max_workers=2)
        report = engine.profile_dataset(df, 'Test Dataset')
        
        print(f"✅ Generated quality report")
        print(f"   - Overall Quality Score: {report.overall_quality_score:.1f}/100")
        print(f"   - Critical Issues: {len(report.critical_issues)}")
        print(f"   - Columns Analyzed: {len(report.column_profiles)}")
        
        # Test specific column analysis
        print("3. Testing Column Analysis...")
        for col_name, profile in list(report.column_profiles.items())[:2]:  # Test first 2 columns
            print(f"   - {col_name}: {profile.null_count} nulls, {profile.data_type}")
            
        print("\n🎉 All tests passed! Data Quality Engine is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    sys.exit(0 if success else 1)
