# FAQ Chatbot: Multi-Source AI Assistant

A visually stunning, modern AI chatbot that can deeply analyze and summarize content from:
- **PDF and DOCX documents**
- **YouTube videos** (via URL)
- **Websites** (via URL)

Built with free Hugging Face models and a beautiful Streamlit UI inspired by top AI landing pages.

---

## ✨ Features
- **Modern, attractive UI** with gradient backgrounds and animated robot art
- **Upload PDFs/DOCX** or paste YouTube/website URLs
- **Deep analysis & summarization** of all sources
- **Conversational Q&A**: Ask questions about your content
- **Automatic YouTube transcript summarization**
- **Free & offline-capable** (uses open-source Hugging Face models)
- **No API keys required**

---

## 🚀 Quickstart

### 1. **Clone the repository**
```bash
git clone <your-repo-url>
cd RAG-Document-summarizer-using-RAG
```

### 2. **Install dependencies**
```bash
pip install -r requirements.txt
# Also install:
pip install youtube-transcript-api trafilatura
```

### 3. **(Optional) Download models for offline use**
If you want to run fully offline, pre-download the models:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", repo_type="model", local_dir="./models/sentence-transformers_all-MiniLM-L6-v2")
snapshot_download(repo_id="google/flan-t5-base", repo_type="model", local_dir="./models/google-flan-t5-base")
```

### 4. **Run the app**
```bash
streamlit run app.py
```

---

## 🖥️ Usage
1. **Landing Page:**
   - See a beautiful hero section with a robot and "Start Chatting" button.
2. **Start Chatting:**
   - Upload PDF/DOCX files, or enter a YouTube/website URL in the sidebar.
   - Click "Process Inputs" to analyze your content.
3. **Ask Questions:**
   - Type questions about your documents, videos, or websites.
   - The bot will answer, summarize, and explain content in depth.
   - For YouTube, it will summarize/explain the transcript automatically.
4. **Summarize All:**
   - Use the "Summarize All Documents" button to get a full summary.

---

## 🛠️ Tech Stack
- **Streamlit** (UI)
- **LangChain** (RAG pipeline)
- **Hugging Face Transformers** (Flan-T5, MiniLM)
- **FAISS** (vector search)
- **youtube-transcript-api** (YouTube ingestion)
- **trafilatura** (website scraping)

---

## 📄 License
MIT

---

## 🙏 Credits
- Robot vector art: [Your image source or credit]
- UI inspiration: Modern AI landing pages

---

## 💡 Tips
- For best results, use clear, well-structured documents and URLs.
- The app works offline if models are pre-downloaded.
- You can customize the UI and prompts in `app.py`.
