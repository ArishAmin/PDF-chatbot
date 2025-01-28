
import streamlit as st
import os
from transformers import pipeline
from model import PDFChatbotHF

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = PDFChatbotHF()
    st.session_state.messages = []

st.title("📚 PDF Chatbot with Hugging Face")


pdf_file = st.file_uploader("Upload your PDF", type=['pdf'])

if pdf_file:
    try:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getvalue())
        
        with st.spinner("Processing PDF... This may take a moment."):
            st.session_state.chatbot.extract_text_from_pdf("temp.pdf")
        st.success("PDF processed successfully!")
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
    finally:
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")


st.markdown("### Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("Thinking..."):
        response = st.session_state.chatbot.get_response(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)

with st.sidebar:
    st.markdown("""
    ### How to use
    1. Upload a PDF document
    2. Wait for the processing to complete
    3. Ask questions about the content
    
    ### Features
    - Hugging Face models for response generation
    - FAISS for efficient similarity search
    - Smart context selection
    - Multi-factor relevance scoring
    """)
    

