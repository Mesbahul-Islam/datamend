#!/usr/bin/env python3
"""
Test script for Snowflake integration
This script tests the Snowflake connector functionality
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from connectors.data_connectors import DataConnectorFactory, SNOWFLAKE_AVAILABLE

def test_snowflake_availability():
    """Test if Snowflake connector is available"""
    print("🧪 Testing Snowflake Availability")
    print(f"Snowflake available: {SNOWFLAKE_AVAILABLE}")
    
    supported_types = DataConnectorFactory.get_supported_types()
    print(f"Supported connector types: {supported_types}")
    
    if 'snowflake' in supported_types:
        print("✅ Snowflake connector is available")
        return True
    else:
        print("❌ Snowflake connector is not available")
        return False

def test_snowflake_connector_creation():
    """Test creating a Snowflake connector (without connecting)"""
    print("\n🧪 Testing Snowflake Connector Creation")
    
    try:
        # Test with dummy credentials (won't actually connect)
        connector = DataConnectorFactory.create_connector(
            'snowflake',
            account='test-account',
            username='test-user',
            password='test-password',
            warehouse='TEST_WH',
            database='TEST_DB',
            schema='PUBLIC'
        )
        print("✅ Snowflake connector created successfully")
        print(f"Connector type: {type(connector).__name__}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create Snowflake connector: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Snowflake Integration")
    print("=" * 50)
    
    # Test availability
    availability_ok = test_snowflake_availability()
    
    if availability_ok:
        # Test connector creation
        creation_ok = test_snowflake_connector_creation()
        
        if creation_ok:
            print("\n🎉 All tests passed! Snowflake integration is ready.")
            print("\n📝 Next steps:")
            print("1. Run the Streamlit app: streamlit run main.py")
            print("2. Go to the Data Source tab")
            print("3. Select 'Snowflake Database' from the dropdown")
            print("4. Enter your Snowflake credentials")
            print("5. Connect and load data for analysis")
        else:
            print("\n⚠️ Snowflake connector creation failed")
    else:
        print("\n❌ Snowflake is not available. Please install: pip install snowflake-connector-python")

if __name__ == "__main__":
    main()
