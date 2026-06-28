import streamlit as st
import requests
import uuid

import os

# Import our custom UI components
from ui.components import render_source_card, render_confidence_badge

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# Load custom CSS
css_path = os.path.join(os.path.dirname(__file__), "styles.css")
try:
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Initialize Streamlit session state to hold the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Initialize a unique session ID for the backend conversation memory
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("🧠 Enterprise Knowledge Assistant")
st.markdown("Ask questions based on your ingested documents.")

# 1. Render all historical messages from the session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display the custom components for assistant responses
        if msg["role"] == "assistant":
            if "confidence" in msg:
                render_confidence_badge(msg["confidence"])
            if "sources" in msg and msg["sources"]:
                with st.expander("View Sources"):
                    for source in msg["sources"]:
                        render_source_card(source)

# 2. Capture new user input
if prompt := st.chat_input("Ask something..."):
    # Display the new user message instantly
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Save user message to frontend history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 3. Request an answer from the FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Assuming the FastAPI router is mounted at /api/v1 as we configured in main.py
                response = requests.post(
                    "http://localhost:8000/api/v1/ask",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=45
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    confidence = data.get("confidence", 0.0)
                    sources = data.get("sources", [])
                    
                    # Display the response
                    st.markdown(answer)
                    render_confidence_badge(confidence)
                    
                    if sources:
                        with st.expander("View Sources"):
                            for source in sources:
                                render_source_card(source)
                                
                    # Save assistant message and metadata to frontend history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "confidence": confidence,
                        "sources": sources
                    })
                else:
                    st.error(f"Backend API Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the backend server. Is it running on port 8000? Error: {e}")