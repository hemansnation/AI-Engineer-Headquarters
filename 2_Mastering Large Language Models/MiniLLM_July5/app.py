import streamlit as st
from query_model import generate_text

st.title("Mini GPT-Like LLM")
prompt = st.text_input("Enter your prompt:", "Once upon a time")

if st.button("Generate"):
    output = generate_text(prompt)
    st.write(output)
