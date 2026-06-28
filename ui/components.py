import streamlit as st
from ui.constants import CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD

def render_source_card(citation: dict):
    """
    Renders a clean, reusable Streamlit component for a single citation.
    The citation dict should match the API shape: {"document": "...", "page": ...}
    """
    doc_name = citation.get("document", "Unknown Document")
    page_num = citation.get("page")
    
    # Format the citation string nicely inside the custom CSS class
    page_str = f" <b>(Page {page_num})</b>" if page_num is not None else ""
    card_html = f"""
    <div class="source-card">
        📄 {doc_name}{page_str}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_confidence_badge(score: float):
    """
    Renders a color-coded confidence badge (green/orange/red) based on 
    the system's predefined thresholds.
    """
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        color = "#28a745" # Green
        label = "High Confidence"
    elif score >= CONFIDENCE_MEDIUM_THRESHOLD:
        color = "#ffc107" # Yellow/Orange
        label = "Medium Confidence"
    else:
        color = "#dc3545" # Red
        label = "Low Confidence"
        
    # Render using raw HTML/CSS to construct a nice visual pill badge
    badge_html = f"""
    <div style="display: inline-block; background-color: {color}; color: #fff; 
                padding: 4px 8px; border-radius: 12px; font-size: 12px; 
                font-weight: 600; margin-bottom: 8px;">
        {label}: {score:.2f}
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)
