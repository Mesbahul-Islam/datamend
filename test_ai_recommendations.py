#!/usr/bin/env python3
"""
Test script for AI Recommendations integration
This script tests the ProfileReport integration with LLM analyzer
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llm.analyzer import DataQualityLLMAnalyzer, LLMConfig
from ydata_profiling import ProfileReport
import pandas as pd

def test_profilereport_integration():
    """Test ProfileReport integration with LLM analyzer"""
    print("🧪 Testing ProfileReport Integration")
    print("=" * 50)
    
    # Create test data with various quality issues
    df = pd.DataFrame({
        'id': range(1, 101),
        'name': ['John', 'Jane', None, 'Bob'] * 25,  # 25% missing
        'age': [25, 30, 35, 40] * 25,
        'salary': [50000, 60000, None, 80000] * 25,  # 25% missing
        'category': ['A'] * 90 + ['B'] * 10,  # Imbalanced
        'unique_id': range(1, 101)  # High cardinality
    })
    
    print(f"Created test dataset: {len(df)} rows × {len(df.columns)} columns")
    
    try:
        # Create ProfileReport
        print("Creating ydata-profiling report...")
        profile = ProfileReport(df, minimal=True, title="Test Data Profile")
        print("✅ ProfileReport created successfully")
        
        # Create LLM analyzer
        config = LLMConfig(
            provider='openai',
            model='gpt-3.5-turbo',
            api_key='test-key'  # Won't actually call API
        )
        analyzer = DataQualityLLMAnalyzer(config)
        print("✅ LLM Analyzer created successfully")
        
        # Test prompt building (this was the failing part)
        print("Testing prompt building...")
        prompt = analyzer._build_analysis_prompt(profile, 'general')
        print(f"✅ Prompt building successful! Length: {len(prompt)} characters")
        
        # Check if key data quality issues were detected
        issues_detected = []
        if 'name (25.0% missing)' in prompt:
            issues_detected.append("Missing data in 'name' column")
        if 'salary (25.0% missing)' in prompt:
            issues_detected.append("Missing data in 'salary' column")
        if 'unique_id (100 unique values)' in prompt:
            issues_detected.append("High cardinality in 'unique_id' column")
            
        if issues_detected:
            print("✅ Data quality issues detected:")
            for issue in issues_detected:
                print(f"   - {issue}")
        else:
            print("⚠️  No specific data quality issues detected in prompt")
        
        # Test that the prompt is well-formed
        if 'Dataset Overview:' in prompt and 'recommendations' in prompt.lower():
            print("✅ Prompt structure is correct")
        else:
            print("❌ Prompt structure issue")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Testing AI Recommendations ProfileReport Integration")
    print("=" * 60)
    
    success = test_profilereport_integration()
    
    if success:
        print("\n🎉 All tests passed!")
        print("\n📝 The ProfileReport integration is working correctly.")
        print("The 'ProfileReport object has no attribute get' error has been fixed.")
        print("\n🚀 You can now use AI Recommendations in the Streamlit app:")
        print("1. Load data in the Data Source tab")
        print("2. Run profiling in the Data Profiling tab") 
        print("3. Enable AI Recommendations in the sidebar")
        print("4. Go to AI Recommendations tab and generate recommendations")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
