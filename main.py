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
from src.ui.ai_recommendations import ai_recommendations_tab

# Configure Streamlit page
st.set_page_config(
    page_title="Data Quality Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main application function"""
    st.title("Data Quality Engine")
    st.markdown("---")
    
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
        else:  # Snowflake Database
            handle_sidebar_snowflake_connection()
        
        # Configuration section
        st.markdown("---")
        st.header("Configuration")
        
        # Analysis configuration (simplified)
        anomaly_threshold = st.slider("Anomaly Threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.1,
                                    help="Z-score threshold for anomaly detection")
        
        # LLM configuration
        st.subheader("AI Recommendations")
        use_llm = st.checkbox("Enable AI Recommendations", value=False)
        if use_llm:
            provider = st.selectbox(
                "LLM Provider", 
                ["OpenAI", "Google Gemini"], 
                index=0,
                help="Choose your preferred LLM provider"
            )
            
            if provider == "OpenAI":
                api_key = st.text_input(
                    "OpenAI API Key", 
                    type="password", 
                    help="Enter your OpenAI API key"
                )
                model = st.selectbox(
                    "Model", 
                    ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"], 
                    index=0
                )
            else:  # Google Gemini
                api_key = st.text_input(
                    "Google AI API Key", 
                    type="password", 
                    help="Enter your Google AI Studio API key"
                )
                model = st.selectbox(
                    "Model", 
                    ["gemini-2.0-flash", "gemini-2.0-pro"], 
                    index=0,
                    help="Gemini models available through Google AI"
                )

    # Create tabs for analysis
    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Data Profiling", "Anomaly Detection", "AI Recommendations"])
    
    with tab1:
        data_overview_tab()
    
    with tab2:
        data_profiling_tab(anomaly_threshold)
    
    with tab3:
        anomaly_detection_tab(anomaly_threshold)
    
    with tab4:
        ai_recommendations_tab(
            use_llm, 
            api_key if 'api_key' in locals() else "", 
            model if 'model' in locals() else "gpt-3.5-turbo",
            provider.lower() if 'provider' in locals() else "openai"
        )


if __name__ == "__main__":
    main()
