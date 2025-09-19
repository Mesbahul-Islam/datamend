"""
AI Recommendations Module
Handles AI-powered data quality recommendations
"""

import streamlit as st
from typing import Dict, Any
from src.llm.analyzer import DataQualityLLMAnalyzer, LLMConfig


def ai_recommendations_tab(use_llm: bool, api_key: str, model: str):
    """AI recommendations tab"""
    st.header("🤖 AI Recommendations")
    
    if not use_llm:
        st.info("💡 Enable AI Recommendations in the sidebar to get intelligent suggestions")
        return
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Source tab")
        return
    
    # Show current dataset info
    if st.session_state.current_dataset:
        st.info(f"🤖 **Analyzing Dataset**: {st.session_state.current_dataset}")
    
    if not st.session_state.get('ydata_profile'):
        st.warning("⚠️ Please run data profiling first")
        return
    
    # Recommendations controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Generate AI-powered recommendations based on your data quality analysis")
    
    with col2:
        if st.button("🤖 Get Recommendations", type="primary", key="get_recommendations_button"):
            generate_recommendations(api_key, model)
    
    # Show recommendations if available
    if st.session_state.get('recommendations'):
        display_recommendations(st.session_state.recommendations)


def generate_recommendations(api_key: str, model: str):
    """Generate AI recommendations"""
    try:
        with st.spinner("🤖 Generating AI recommendations..."):
            config = LLMConfig(
                provider="openai",
                model=model,
                api_key=api_key
            )
            analyzer = DataQualityLLMAnalyzer(config)
            
            # Build the prompt
            quality_report = st.session_state.ydata_profile  # Ensure this is the correct data structure
            context = "general"  # Adjust context as needed
            prompt = analyzer._build_analysis_prompt(quality_report, context)
            
            # Call the LLM API
            response = analyzer._call_llm_api(prompt)
            
            # Parse the response
            recommendations = analyzer._parse_llm_response(response)
            
            # Store recommendations in session state
            st.session_state.recommendations = recommendations
        
        st.success("✅ AI recommendations generated successfully!")
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {str(e)}")


def display_recommendations(recommendations: Dict[str, Any]):
    """Display AI recommendations"""
    st.subheader("💡 AI-Powered Recommendations")
    
    # Summary
    if 'summary' in recommendations:
        st.info(f"📋 **Summary:** {recommendations['summary']}")
    
    # Individual recommendations
    if 'recommendations' in recommendations:
        for i, rec in enumerate(recommendations['recommendations'], 1):
            priority = rec.get('priority', 'medium')
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "🟡")
            
            with st.expander(f"{priority_color} {rec.get('title', f'Recommendation {i}')}"):
                st.write(f"**Priority:** {priority.title()}")
                st.write(f"**Type:** {rec.get('type', 'General').replace('_', ' ').title()}")
                
                if 'description' in rec:
                    st.write(f"**Description:** {rec['description']}")
                
                if 'suggested_actions' in rec and rec['suggested_actions']:
                    st.write("**Suggested Actions:**")
                    for action in rec['suggested_actions']:
                        st.write(f"• {action}")
