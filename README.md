# PDF Chatbot Using Hugging Face and FAISS

This repository contains a chatbot that allows you to upload PDF documents and ask questions about their content. The chatbot uses advanced Natural Language Processing (NLP) techniques with Hugging Face transformers and FAISS for semantic search to provide accurate and context-aware answers.

## Features

- **PDF Upload**: Supports uploading PDF files for content extraction.
- **Hugging Face Integration**: Utilizes the `t5-small` model for natural language generation.
- **Semantic Search**: Implements FAISS for fast and efficient similarity search.
- **Smart Context Selection**: Retrieves and ranks the most relevant sentences from the document.
- **Streamlit UI**: Easy-to-use web interface for interactions.


## File Structure

- **`app.py`**: Streamlit application for the user interface.
- **`model.py`**: Core logic for text extraction, semantic search, and response generation.
- **`requirements.txt`**: List of dependencies and their versions.


## Technologies Used

- **Hugging Face Transformers**: For text generation (`t5-small`).
- **FAISS**: For efficient similarity search.
- **Sentence Transformers**: For embedding generation (`all-mpnet-base-v2`).
- **Streamlit**: For building the web interface.

---




