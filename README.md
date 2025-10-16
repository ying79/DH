# DH 

#### 📚 *An Assignment of the Digital Humanities Course* 

## 💬 Japanese Literature Name Finder (RAG-based Web App)

This project is a simple Retrieval-Augmented Generation (RAG) web application that allows users to search for where a given kanji name appears in classical Japanese literature. It uses 青空文庫 (Aozora Bunko) as the text source and demonstrates the idea of combining NLP search with conversational AI interaction.

### 🌸 Project Overview

When a user inputs a kanji name (e.g. 「瑩」), the system searches downloaded Japanese literary works from Aozora Bunko, finds matching contexts, and summarizes the results in natural Japanese sentences.

This project is developed for the Digital Humanities course and also serves as a base framework for future expansion.

### 🧩 System Architecture

*Streamlit Frontend (UI)*

- Chat-style user input
- Display matched excerpts
- Show source references

*RAG Core (Python Backend)*

- Text Preprocessing
- Chunking & Embeddings
- Vector Search (FAISS)
- Gemini LLM Generation

*Aozora Bunko Dataset*

- .txt files (UTF-8)
- Metadata (title, author)

### ⚙️ Tech Stack

| Layer | Tool / Library | Description |
|-------|----------------|--------------|
| Frontend | Streamlit | Interactive web app interface |
| Backend | Python| Main development language |
| Embeddings | Google Gemini Embeddings API | Convert text chunks into vectors |
| Vector DB | FAISS or Chroma | Efficient similarity search |
| LLM | Gemini | Generate summarized answers |
| Data | Aozora Bunko | Public domain Japanese literature |


### 📂 Project Structure
```
project_root/
├── app.py — Streamlit web interface
├── rag_core.py — Main RAG logic (embedding, search, generation)
├── requirements.txt — Python dependencies
├── config.yaml — API keys & path settings (excluded from repo)
├── data/ — Text corpus folder
│ ├── botchan.txt
│ ├── rashomon.txt
│ └── ...
└── vectorstore/ — Saved embeddings or FAISS index
```




### 🧠 Workflow Summary

*Preprocessing* – Parse and preserve ruby annotations from Aozora texts to retain the original readings given by authors, while cleaning unnecessary markup and splitting the text into semantic chunks.

*Embedding* – Generate semantic embeddings for each chunk using Gemini Embeddings API.

*Storage* – Save all embeddings to FAISS or Chroma vector database.

*Retrieval* – Convert user query into embedding and search for similar text.

*Generation* – Feed top-k matched passages into Gemini model to summarize and display sources.



### 💡 References


青空文庫: https://www.aozora.gr.jp/   
FAISS: https://github.com/facebookresearch/faiss  
Streamlit Docs: https://docs.streamlit.io/  
Google Generative AI SDK: https://ai.google.dev/  
