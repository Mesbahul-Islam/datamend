"""
Session State Management for Data Quality Engine
"""

import streamlit as st


def initialize_session_state():
    """Initialize session state variables"""
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}  # Dictionary to store multiple datasets
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None  # Currently selected dataset
    if 'data' not in st.session_state:
        st.session_state.data = None  # For backward compatibility
    if 'ydata_profiles' not in st.session_state:
        st.session_state.ydata_profiles = {}  # ydata-profiling reports for each dataset
    if 'ydata_profile' not in st.session_state:
        st.session_state.ydata_profile = None  # Currently selected profile
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'connectors' not in st.session_state:
        st.session_state.connectors = {}  # Connectors for each dataset
    if 'connector' not in st.session_state:
        st.session_state.connector = None  # For backward compatibility
    if 'profiling_complete' not in st.session_state:
        st.session_state.profiling_complete = False
