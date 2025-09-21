"""
Data Quality Engine - Main Application Entry Point

A modular data quality management interface focused on:
- Data upload (CSV/Excel)
- Interactive data profiling with ydata-profiling
- Statistical anomaly detection
- AI-driven recommendations
"""

import streamlit as st
import sys
import os


# Add src to path for imports
# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import UI modules
from src.ui.session_state import initialize_session_state
from src.ui.data_source import (
    data_overview_tab, 
    handle_sidebar_csv_upload, 
    handle_sidebar_excel_upload, 
    handle_sidebar_snowflake_connection
)
from src.ui.data_profiling import data_profiling_tab
from src.ui.anomaly_detection import anomaly_detection_tab
from src.ui.data_lineage import data_lineage_tab

# Configure Streamlit page
st.set_page_config(
    page_title="DataMend",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    /* Alternative selector in case the above doesn't work */
    .stTabs [data-baseweb="tab-list"] {
        font-size: 18px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    st.title("DataMend - Data Quality Management")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for data uploading and configuration
    with st.sidebar:
        st.header("Data Upload")
        
        # Show connection status if Snowflake is connected
        if st.session_state.get('snowflake_connected', False):
            st.success("Snowflake Connected")
        
        # Data source selection
        st.subheader("Select Data Source")
        source_type = st.selectbox(
            "Choose data source type:",
            ["CSV Files", "Excel Files", "Snowflake Database"],
            help="Select the type of data you want to upload"
        )
        
        # Handle different data source types
        if source_type == "CSV Files":
            handle_sidebar_csv_upload()
        elif source_type == "Excel Files":
            handle_sidebar_excel_upload()
        elif source_type == "Snowflake Database":
            handle_sidebar_snowflake_connection()
        
        # Configuration section
        st.markdown("---")
        st.header("Configuration")
        
        # Analysis configuration (simplified)
        anomaly_threshold = st.slider("Anomaly Threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.1,
                                    help="Z-score threshold for anomaly detection. Higher values mean stricter anomaly detection.")
        
        # LLM configuration
        st.subheader("AI Recommendations")
        
        # Try to automatically configure from environment
        try:
            from src.llm.analyzer import DataQualityLLMAnalyzer
            analyzer = DataQualityLLMAnalyzer()
            env_config = analyzer._load_config_from_env()
            auto_configured = bool(env_config.api_key)
        except Exception:
            auto_configured = False
            env_config = None
        
        # Enable by default if API key is available in environment
        use_llm = st.checkbox("Enable AI Recommendations", value=auto_configured)
        
        if use_llm:
            if auto_configured:
                # Show current auto-configuration with option to override
                st.success(f"Auto-configured with {env_config.provider.upper()} from environment")
                
                # Option to use custom settings instead
                use_custom = st.checkbox("Use custom API settings", value=False, 
                                       help="Check this to override the environment configuration")
                
                if not use_custom:
                    # Use environment configuration
                    st.info(f"**Provider:** {env_config.provider.upper()} | **Model:** {env_config.model}")
                    st.session_state['llm_auto_config'] = {
                        'provider': env_config.provider,
                        'model': env_config.model,
                        'api_key': env_config.api_key
                    }
                else:
                    # Show custom configuration options - inline implementation
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        provider = st.selectbox(
                            "LLM Provider", 
                            ["Google Gemini", "OpenAI"], 
                            index=0,  # Default to Gemini
                            help="Choose your preferred LLM provider",
                            key="custom_provider"
                        )
                    
                    with col2:
                        if provider == "OpenAI":
                            model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
                            default_model = "gpt-3.5-turbo"
                            provider_key = "openai"
                        else:  # Google Gemini
                            model_options = ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-flash"]
                            default_model = "gemini-2.0-flash"
                            provider_key = "gemini"
                        
                        model = st.selectbox("Model", model_options, 
                                           index=model_options.index(default_model),
                                           key="custom_model")
                    
                    # API Key input
                    if provider == "OpenAI":
                        env_api_key = os.getenv("OPENAI_API_KEY", "")
                        api_key = st.text_input(
                            "OpenAI API Key", 
                            value=env_api_key,
                            type="password", 
                            help="Enter your OpenAI API key",
                            key="custom_openai_key"
                        )
                    else:  # Google Gemini
                        env_api_key = os.getenv("GOOGLE_AI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
                        api_key = st.text_input(
                            "Google AI API Key", 
                            value=env_api_key,
                            type="password", 
                            help="Enter your Google AI Studio API key",
                            key="custom_gemini_key"
                        )
                    
                    # Store custom configuration
                    if api_key:
                        st.session_state['llm_auto_config'] = {
                            'provider': provider_key,
                            'model': model,
                            'api_key': api_key
                        }
            else:
                # No environment config available, show manual configuration
                col1, col2 = st.columns(2)
                
                with col1:
                    provider = st.selectbox(
                        "LLM Provider", 
                        ["Google Gemini", "OpenAI"], 
                        index=0,  # Default to Gemini
                        help="Choose your preferred LLM provider",
                        key="manual_provider"
                    )
                
                with col2:
                    if provider == "OpenAI":
                        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
                        default_model = "gpt-3.5-turbo"
                        provider_key = "openai"
                    else:  # Google Gemini
                        model_options = ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-flash"]
                        default_model = "gemini-2.0-flash"
                        provider_key = "gemini"
                    
                    model = st.selectbox("Model", model_options, 
                                       index=model_options.index(default_model),
                                       key="manual_model")
                
                # API Key input
                if provider == "OpenAI":
                    env_api_key = os.getenv("OPENAI_API_KEY", "")
                    api_key = st.text_input(
                        "OpenAI API Key", 
                        value=env_api_key,
                        type="password", 
                        help="Enter your OpenAI API key",
                        key="manual_openai_key"
                    )
                else:  # Google Gemini
                    env_api_key = os.getenv("GOOGLE_AI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
                    api_key = st.text_input(
                        "Google AI API Key", 
                        value=env_api_key,
                        type="password", 
                        help="Enter your Google AI Studio API key",
                        key="manual_gemini_key"
                    )
                
                # Store manual configuration
                if api_key:
                    st.session_state['llm_auto_config'] = {
                        'provider': provider_key,
                        'model': model,
                        'api_key': api_key
                    }

    # User guidance section
    if st.session_state.data is not None:
        lineage_text = "\n4. **Data Lineage** - Visualize Snowflake data dependencies with AI insights" if st.session_state.get('data_source') == 'snowflake' else ""
        
        st.info("**Quick Start Guide:**\n"
                "1. **Data Overview** - Basic dataset information\n"
                "2. **Data Profiling** - Comprehensive analysis with AI insights\n"
                "   • Quick Summary | Detailed Report | Statistical Outliers | Analytics Quality\n"
                "   • Get AI-powered recommendations after profiling\n"
                "3. **Statistical Outliers** - View outlier results with AI explanations" + lineage_text)
        
        # Add the small text separately
        st.markdown("<small style='color: #6c757d;'>😉 Make sure to clear AI assessment after use (just because)</small>", unsafe_allow_html=True)
    else:
        st.warning("Please load a dataset using the sidebar to get started.")

    # Create tabs for analysis (conditionally include Data Lineage for Snowflake)
    tab_names = [
        "Data Overview", 
        "Data Profiling", 
        "Anomaly Detection Results Overview"
    ]
    
    # Add Data Lineage tab if data source is Snowflake
    if st.session_state.get('data_source') == 'snowflake':
        tab_names.append("Data Lineage")
    
    tabs = st.tabs(tab_names)
    
    # Data Overview tab
    with tabs[0]:
        data_overview_tab()
    
    # Data Profiling tab  
    with tabs[1]:
        data_profiling_tab(anomaly_threshold)
    
    # Statistical Outliers tab
    with tabs[2]:
        anomaly_detection_tab(anomaly_threshold)
    
    # Data Lineage tab (only for Snowflake)
    if st.session_state.get('data_source') == 'snowflake':
        with tabs[3]:
            data_lineage_tab()


if __name__ == "__main__":
    main()
