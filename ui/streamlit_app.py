import sys
import os

# Add the project root to sys.path to resolve 'ui' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import uuid

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
st.markdown("Ask questions based on documents.")

# 1. Render all historical messages from the session state
for i, msg in enumerate(st.session_state.messages):
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
                        
            # Add interactive feedback widget
            feedback = st.feedback("thumbs", key=f"fb_{i}")
            if feedback is not None and not msg.get("feedback_submitted"):
                # Find the corresponding user question
                question = "Unknown Question"
                if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                    question = st.session_state.messages[i-1]["content"]
                    
                try:
                    requests.post(
                        "http://localhost:8000/api/v1/feedback",
                        json={
                            "session_id": st.session_state.session_id,
                            "question": question,
                            "answer": msg["content"],
                            "score": feedback
                        },
                        timeout=5
                    )
                    # Mark as submitted so we don't resend on every script rerun
                    msg["feedback_submitted"] = True
                except Exception as e:
                    st.error(f"Failed to submit feedback: {e}")

# 2. Capture new user input and optional file attachments
if prompt_data := st.chat_input("Ask something...", accept_file="multiple", file_type=["pdf", "txt", "docx", "csv"]):
    prompt = prompt_data.text
    uploaded_files = prompt_data.files
    
    # Process files if any were uploaded
    if uploaded_files:
        with st.spinner("Processing attached document(s)..."):
            DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "documents")
            os.makedirs(DOCUMENTS_DIR, exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Trigger ingestion pipeline
            try:
                response = requests.post(
                    "http://localhost:8000/api/v1/ingest",
                    json={"directory": DOCUMENTS_DIR},
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        st.success(f"Successfully processed! Generated {data.get('total_chunks', 0)} chunks.")
                    else:
                        st.error(f"Ingestion failed: {data.get('error')}")
                else:
                    st.error(f"Backend API Error ({response.status_code}): {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend for ingestion: {e}")

    # If the user typed a question, process it
    if prompt:
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
                        
                        # Force a rerun to display the feedback widget for this new message
                        st.rerun()
                    else:
                        st.error(f"Backend API Error ({response.status_code}): {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to the backend server. Is it running on port 8000? Error: {e}")