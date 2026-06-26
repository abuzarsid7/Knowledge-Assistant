import streamlit as st

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    layout="wide"
)

st.title("Enterprise Knowledge Assistant")

st.success("Streamlit is working!")

question = st.text_input("Ask something")

if st.button("Submit"):
    st.write(f"You asked: {question}")