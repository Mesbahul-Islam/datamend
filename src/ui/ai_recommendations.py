"""
AI Recommendations Module
Handles AI-powered data quality recommendations
"""

import streamlit as st
from typing import Dict, Any
from llm.analyzer import DataQualityLLMAnalyzer, LLMConfig


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
            if not api_key:
                # Use mock recommendations if no API key
                mock_recommendations = {
                    "recommendations": [
                        {
                            "type": "data_quality",
                            "priority": "high",
                            "title": "Address Missing Values",
                            "description": "Several columns have significant missing values that could impact analysis quality.",
                            "suggested_actions": [
                                "Consider imputation strategies for numerical columns",
                                "Investigate the root cause of missing data",
                                "Document data collection processes"
                            ]
                        },
                        {
                            "type": "data_validation",
                            "priority": "medium", 
                            "title": "Standardize Data Types",
                            "description": "Some columns may benefit from consistent data type formatting.",
                            "suggested_actions": [
                                "Convert string numbers to numeric types",
                                "Standardize date formats",
                                "Review categorical variable encoding"
                            ]
                        }
                    ],
                    "summary": "Your dataset shows good overall quality with some areas for improvement in completeness and consistency."
                }
                st.session_state.recommendations = mock_recommendations
            else:
                # Use actual LLM API
                config = LLMConfig(
                    provider="openai",
                    model=model,
                    api_key=api_key
                )
                
                analyzer = DataQualityLLMAnalyzer(config)
                recommendations = analyzer.analyze_data_quality(
                    st.session_state.data,
                    st.session_state.ydata_profile
                )
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
