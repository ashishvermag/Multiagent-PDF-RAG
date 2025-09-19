# 🤖 Multi-Agent PDF Chatbot

A sophisticated, interactive chatbot application built with **Streamlit** that allows you to turn any PDF document into a conversational AI agent. You can create multiple agents from different PDFs, switch between them, and have persistent, source-cited conversations.

---

## ✨ Features

- **Dynamic Agent Creation**: Upload any PDF and instantly create a specialized chatbot agent knowledgeable about its contents.  
- **Multi-Agent Management**: Seamlessly create and switch between multiple agents, each with its own memory and context.  
- **Persistent Conversations**: Chat histories are automatically saved and loaded across sessions.  
- **Source-Cited Answers**: Every answer includes exact text chunks from the source document for trust and verifiability.  
- **Full Chat Control**:  
  - Clear history for a specific agent  
  - Delete an agent and its associated database & history  
- **Intuitive UI**: Clean and user-friendly, powered by Streamlit.  

---

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)  
- **LLM**: [Google Gemini 1.5 Flash](https://deepmind.google/technologies/gemini/)  
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (persistent)  
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`  
- **PDF Parsing**: [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/en/latest/)  
- **Text Processing**: [LangChain](https://www.langchain.com/)  

---

## 🚀 Setup and Installation (Windows)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
