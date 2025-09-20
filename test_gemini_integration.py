#!/usr/bin/env python3
"""
Test script for Google Gemini integration
This script tests the Gemini integration with the LLM analyzer
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_gemini_availability():
    """Test if Gemini is available"""
    print("🧪 Testing Gemini Availability")
    print("=" * 50)
    
    try:
        import google.genai as genai
        print("✅ google-genai library is available")
        return True
    except ImportError as e:
        print(f"❌ google-genai library not available: {e}")
        print("Install with: pip install google-genai")
        return False

def test_gemini_integration():
    """Test Gemini integration with LLM analyzer"""
    print("\n🧪 Testing Gemini Integration")
    print("=" * 50)
    
    try:
        from llm.analyzer import DataQualityLLMAnalyzer, LLMConfig, GEMINI_AVAILABLE
        
        if not GEMINI_AVAILABLE:
            print("❌ Gemini not available in analyzer")
            return False
        
        # Test config creation
        config = LLMConfig(
            provider="gemini",
            model="gemini-1.5-flash", 
            api_key="test-key"
        )
        
        print("✅ Gemini config created successfully")
        print(f"   Provider: {config.provider}")
        print(f"   Model: {config.model}")
        print(f"   API URL: {config.api_url}")
        
        # Test analyzer creation
        analyzer = DataQualityLLMAnalyzer(config)
        print("✅ LLM Analyzer with Gemini config created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gemini_with_real_api():
    """Test with real API key (if available)"""
    print("\n🧪 Testing Real Gemini API (Optional)")
    print("=" * 50)
    
    # Check if API key is available in environment
    api_key = os.getenv('LLM_API_KEY')
    provider = os.getenv('LLM_PROVIDER', '').lower()
    
    if not api_key or provider != 'gemini':
        print("ℹ️  No Gemini API key found in environment variables")
        print("   Set LLM_PROVIDER=gemini and LLM_API_KEY=your_key to test real API")
        return True
    
    try:
        from llm.analyzer import DataQualityLLMAnalyzer, LLMConfig
        from ydata_profiling import ProfileReport
        import pandas as pd
        
        # Create test data
        df = pd.DataFrame({
            'A': [1, 2, 3, None, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Create profile
        profile = ProfileReport(df, minimal=True)
        
        # Create analyzer with Gemini
        config = LLMConfig(
            provider="gemini",
            model="gemini-1.5-flash",
            api_key=api_key
        )
        analyzer = DataQualityLLMAnalyzer(config)
        
        # Test prompt building
        prompt = analyzer._build_analysis_prompt(profile, 'general')
        print("✅ Prompt building successful")
        
        # Test API call (this will use real API)
        print("🔄 Testing real Gemini API call...")
        response = analyzer._call_llm_api(prompt[:500])  # Shortened for test
        print("✅ Gemini API call successful!")
        print(f"   Response length: {len(response)} characters")
        print(f"   Response preview: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Real API test failed: {e}")
        if "API_KEY" in str(e):
            print("💡 Check your Google AI API key")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Google Gemini Integration")
    print("=" * 60)
    
    # Test 1: Check availability
    availability_ok = test_gemini_availability()
    
    if not availability_ok:
        print("\n❌ Gemini library not available. Install with:")
        print("pip install google-genai")
        return
    
    # Test 2: Test integration
    integration_ok = test_gemini_integration()
    
    if not integration_ok:
        print("\n❌ Gemini integration test failed")
        return
    
    # Test 3: Test real API (optional)
    real_api_ok = test_gemini_with_real_api()
    
    print("\n" + "=" * 60)
    if availability_ok and integration_ok:
        print("🎉 Gemini integration is ready!")
        print("\n📝 How to use Gemini in the app:")
        print("1. Get your API key from: https://aistudio.google.com/app/apikey")
        print("2. Run the Streamlit app: streamlit run main.py")
        print("3. In the sidebar, enable 'AI Recommendations'")
        print("4. Select 'Google Gemini' as the provider")
        print("5. Enter your API key and select a model")
        print("6. Go to AI Recommendations tab and generate recommendations")
        
        if real_api_ok and os.getenv('LLM_API_KEY'):
            print("\n✅ Real API test also passed!")
        else:
            print("\nℹ️  Set environment variables to test real API:")
            print("   export LLM_PROVIDER=gemini")
            print("   export LLM_API_KEY=your_api_key")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
