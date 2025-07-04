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
- **No API keys required for core features**
- **Works with local files and online content**
- **Source highlighting**: See which document or URL your answer came from
- **Easy model management**: Use your own downloaded models for privacy/offline use

---

## ⚠️ Required: Download Models Manually

> **The required models are NOT included in this repository. You must download them yourself before running the app.**

Download the following models from Hugging Face and place them in the `models/` directory as shown:

1. **Sentence Transformers (Embeddings):**
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Download URL: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
   - Place in: `models/sentence-transformers_all-MiniLM-L6-v2/`

2. **Flan-T5 (LLM):**
   - Model: `google/flan-t5-base`
   - Download URL: https://huggingface.co/google/flan-t5-base
   - Place in: `models/google-flan-t5-base/`

You can use the following Python code to download them:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", repo_type="model", local_dir="./models/sentence-transformers_all-MiniLM-L6-v2")
snapshot_download(repo_id="google/flan-t5-base", repo_type="model", local_dir="./models/google-flan-t5-base")
```

---

## 🚀 Quickstart

### 1. **Clone the repository**
```bash
git clone https://github.com/Purnima2004/FAQ-Chatbot.git
cd FAQ-Chatbot
```

### 2. **Install dependencies**
```bash
pip install -r requirements.txt
# Also install:
pip install youtube-transcript-api trafilatura
```

### 3. **Download models (see above)**
- Make sure the models are in the correct folders as described above.

### 4. **Run the app**
```bash
streamlit run app.py
```

---

## 🖥️ Usage
1. **Landing Page:**
   - See a beautiful hero section with a robot and "Start Chatting" button.
2. **Start Chatting:**
   - Upload PDF/DOCX files, or enter a YouTube/website URL in the chat bar.
   - The app will process your content and build a knowledge base.
3. **Ask Questions:**
   - Type questions about your documents, videos, or websites.
   - The bot will answer, summarize, and explain content in depth.
   - For YouTube, it will summarize/explain the transcript automatically.
4. **Summarize All:**
   - Use the "Summarize All Documents" button to get a full summary.
5. **Source Attribution:**
   - Answers include references to the source document or URL.

---

## 🛠️ Tech Stack
- **Streamlit** (UI)
- **LangChain** (RAG pipeline)
- **Hugging Face Transformers** (Flan-T5, MiniLM)
- **FAISS** (vector search)
- **youtube-transcript-api** (YouTube ingestion)
- **trafilatura** (website scraping)

---

## ❓ Troubleshooting
- **Models not found error:**
  - Make sure you have downloaded the required models and placed them in the correct folders as described above.
- **No answer found:**
  - Try uploading more relevant documents or check if your question matches the content.
- **App fails to start:**
  - Ensure all dependencies are installed and you are using a compatible Python version (3.8+ recommended).
- **Internet required for first run:**
  - If you haven't downloaded the models, the app will try to fetch them online. For offline use, always pre-download the models.

---

## 📄 License
MIT

---
