import streamlit as st
import json
import os
import sys

# Add the project root to sys.path to resolve imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib
import app.core.rag_pipeline
importlib.reload(app.core.rag_pipeline)

import evaluation.evaluate
importlib.reload(evaluation.evaluate)

from evaluation.evaluate import run_evaluation_suite

st.set_page_config(
    page_title="Evaluation Metrics",
    page_icon=":material/analytics:",
    layout="wide"
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "styles.css")
try:
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.title(":material/analytics: Evaluation Metrics")
st.markdown("Run an evaluation on the test question set to calculate performance metrics of the RAG pipeline.")

if st.button("Run Evaluation", type="primary"):
    from evaluation.evaluate import render_evaluation_ui
    render_evaluation_ui(st, project_root)
