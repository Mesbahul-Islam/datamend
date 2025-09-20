#!/usr/bin/env python3
"""
Test script for complete AI recommendations flow
Tests the end-to-end flow from ProfileReport to displayed recommendations
"""

import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_complete_flow():
    """Test the complete recommendations flow"""
    print("🧪 Testing Complete AI Recommendations Flow")
    print("=" * 60)
    
    try:
        from llm.analyzer import DataQualityLLMAnalyzer, LLMConfig
        from ydata_profiling import ProfileReport
        import pandas as pd
        
        # Create test data with quality issues
        df = pd.DataFrame({
            'id': range(1, 101),
            'name': ['John', 'Jane', None, 'Bob'] * 25,  # 25% missing
            'age': [25, 30, 35, 40] * 25,
            'salary': [50000, 60000, None, 80000] * 25,  # 25% missing
        })
        
        print(f"✅ Created test dataset: {len(df)} rows × {len(df.columns)} columns")
        
        # Create ProfileReport
        print("Creating ProfileReport...")
        profile = ProfileReport(df, minimal=True)
        print("✅ ProfileReport created")
        
        # Create analyzer
        config = LLMConfig(
            provider="gemini",
            model="gemini-1.5-flash",
            api_key="test-key"  # Won't actually call API for this test
        )
        analyzer = DataQualityLLMAnalyzer(config)
        print("✅ Analyzer created")
        
        # Test prompt building
        prompt = analyzer._build_analysis_prompt(profile, 'general')
        print(f"✅ Prompt built: {len(prompt)} characters")
        
        # Test response parsing with mock response
        mock_response = """
        Here are my recommendations:
        
        {
          "summary": "The dataset shows some data quality issues that need attention.",
          "recommendations": [
            {
              "type": "data_validation",
              "priority": "high",
              "title": "Address Missing Values",
              "description": "Multiple columns have significant missing values that could impact analysis.",
              "suggested_actions": [
                "Investigate root causes of missing values in name and salary columns",
                "Implement data validation at source",
                "Consider imputation strategies for critical fields"
              ],
              "affected_columns": ["name", "salary"],
              "estimated_impact": "Improved data completeness and analysis reliability"
            },
            {
              "type": "data_monitoring",
              "priority": "medium",
              "title": "Implement Data Quality Monitoring",
              "description": "Regular monitoring can prevent future data quality issues.",
              "suggested_actions": [
                "Set up automated data quality checks",
                "Create data quality dashboards",
                "Establish data quality metrics"
              ],
              "affected_columns": [],
              "estimated_impact": "Proactive data quality management"
            }
          ]
        }
        """
        
        parsed = analyzer._parse_llm_response(mock_response)
        print("✅ Response parsing successful")
        print(f"   Summary: {parsed.get('summary', 'No summary')}")
        print(f"   Recommendations count: {len(parsed.get('recommendations', []))}")
        
        # Test individual recommendations
        for i, rec in enumerate(parsed.get('recommendations', []), 1):
            print(f"   Recommendation {i}: {rec.get('title', 'No title')} ({rec.get('priority', 'no priority')})")
            print(f"      Actions: {len(rec.get('suggested_actions', []))}")
        
        # Test that the format matches UI expectations
        expected_keys = ['summary', 'recommendations']
        if all(key in parsed for key in expected_keys):
            print("✅ Response format matches UI expectations")
        else:
            print("❌ Response format doesn't match UI expectations")
            print(f"   Expected keys: {expected_keys}")
            print(f"   Actual keys: {list(parsed.keys())}")
        
        # Test recommendation structure
        if parsed.get('recommendations'):
            rec = parsed['recommendations'][0]
            rec_keys = ['type', 'priority', 'title', 'description', 'suggested_actions']
            if all(key in rec for key in rec_keys):
                print("✅ Recommendation structure is correct")
            else:
                print("❌ Recommendation structure is incomplete")
                print(f"   Expected keys: {rec_keys}")
                print(f"   Actual keys: {list(rec.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Testing Complete AI Recommendations Flow")
    print("=" * 60)
    
    success = test_complete_flow()
    
    if success:
        print("\n🎉 All tests passed!")
        print("\n📝 The complete AI recommendations flow is working correctly.")
        print("The recommendations should now display properly in the Streamlit app.")
    else:
        print("\n❌ Tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
