import os
import streamlit as st
from dotenv import load_dotenv
import time
import shutil
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    DirectoryLoader,
    UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Fix deprecation warning by using the updated import
try:
    # Try the new recommended import path first
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fall back to the deprecated import path if needed
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
import trafilatura
import base64
import requests

# Load environment variables
load_dotenv()

# Check if running on Render
IS_RENDER = os.environ.get('RENDER', '') == 'true'

# Create necessary directories
os.makedirs("faqs", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)
os.makedirs("models", exist_ok=True)  # Directory to cache models

# Set flags to track if models are pre-downloaded
SENTENCE_TRANSFORMER_DOWNLOADED = os.path.exists(os.path.join("models", "sentence-transformers_all-MiniLM-L6-v2"))
# Hugging Face caches models with a specific naming pattern
FLAN_T5_DOWNLOADED = os.path.exists(os.path.join("models", "google-flan-t5-base"))

# Load environment variables (API keys)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # If needed elsewhere

# Check if we have internet connection
HAS_INTERNET = check_internet_connection()

# Determine if we should use offline mode
USE_OFFLINE_MODE = (SENTENCE_TRANSFORMER_DOWNLOADED and FLAN_T5_DOWNLOADED) or not HAS_INTERNET

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./models"  # Cache models locally
        )

    def load_documents(self, directory: str = "faqs") -> List:
        """Load documents from the specified directory."""
        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
        }
        
        documents = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in loaders:
                try:
                    st.info(f"Loading {filename}...")
                    loader = loaders[ext](file_path)
                    loaded_docs = loader.load()
                    if not loaded_docs:
                        st.warning(f"No content extracted from {filename}")
                    else:
                        documents.extend(loaded_docs)
                        st.success(f"Successfully loaded {filename}")
                except Exception as e:
                    st.error(f"Error loading {filename}: {str(e)}")
                    # More detailed error information for debugging
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")
            else:
                st.warning(f"Unsupported file type: {filename}")
        
        return documents

    def process_documents(self, documents: List) -> None:
        """Process documents and create vector store."""
        if not documents:
            st.warning("No documents found in the 'faqs' directory.")
            return None
            
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Create and save vector store
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local("vectorstore/faiss_index")
        return vectorstore

    def load_youtube_transcript(self, url: str) -> List:
        """Extract transcript from a YouTube video URL and return as a document."""
        import re
        video_id_match = re.search(r"(?:v=|youtu.be/)([\w-]+)", url)
        if not video_id_match:
            st.error("Invalid YouTube URL.")
            return []
        video_id = video_id_match.group(1)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([entry['text'] for entry in transcript])
            from langchain.docstore.document import Document
            return [Document(page_content=text, metadata={"source": url, "type": "youtube"})]
        except Exception as e:
            st.error(f"Could not fetch transcript: {str(e)}")
            return []

    def load_website_content(self, url: str) -> List:
        """Extract main text from a website URL and return as a document."""
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                st.error("Failed to fetch website content.")
                return []
            text = trafilatura.extract(downloaded)
            if not text:
                st.error("Failed to extract main content from website.")
                return []
            from langchain.docstore.document import Document
            return [Document(page_content=text, metadata={"source": url, "type": "website"})]
        except Exception as e:
            st.error(f"Error extracting website: {str(e)}")
            return []

class Chatbot:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Singleton pattern to ensure models are loaded only once."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        # Show a loading message during initialization
        loading_placeholder = st.empty()
        
        # Define model paths
        self.sentence_transformer_path = os.path.join("models", "sentence-transformers_all-MiniLM-L6-v2")
        self.t5_model_path = os.path.join("models", "google-flan-t5-base")
        
        # Check if models exist locally
        models_exist = os.path.exists(self.sentence_transformer_path) and os.path.exists(self.t5_model_path)
        
        if not models_exist and not HAS_INTERNET:
            loading_placeholder.error(
                "Models not found locally and no internet connection available.\n"
                "Please run 'python download_models.py' first to download the models."
            )
            st.stop()
        
        loading_placeholder.info("Initializing models (this may take a few minutes)...")
        start_time = time.time()
        
        try:
            # Initialize embeddings with local path if available
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.sentence_transformer_path if models_exist else "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder="./models"
            )
            
            # Load tokenizer and model from local path if available
            model_path = self.t5_model_path if models_exist else "google-flan-t5-base"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir="./models"
            )
            
            # Load model with device_map for better memory management
            device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                cache_dir="./models",
                device_map="auto" if device == 0 else None,  # Use device_map for GPU
                torch_dtype=torch.float16 if device == 0 else torch.float32  # Use half precision on GPU
            )
            
            if device == -1:  # If using CPU
                self.model = self.model.to(torch.float32)  # Ensure full precision on CPU
                
            # Create a text generation pipeline
            self.pipe = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=0.1,
                repetition_penalty=1.2
            )
            
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            
        except Exception as e:
            loading_placeholder.error(f"Error loading models: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            st.error(
                "If you're having connection issues, please download the models first using:\n\n"
                "```python\n"
                "from huggingface_hub import snapshot_download\n\n"
                "# Download sentence transformer model\n"
                "snapshot_download(repo_id=\"sentence-transformers/all-MiniLM-L6-v2\", repo_type=\"model\", local_dir=\"./models/sentence-transformers_all-MiniLM-L6-v2\")\n\n"
                "# Download flan-t5 model\n"
                "snapshot_download(repo_id=\"google/flan-t5-base\", repo_type=\"model\", local_dir=\"./models/google-flan-t5-base\")\n"
                "```"
            )
            st.stop()
        
        # Initialize memory and QA chain
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.qa_chain = None
        
        # Clear the loading message
        loading_time = time.time() - start_time
        loading_placeholder.success(f"Models initialized in {loading_time:.2f} seconds!")
        time.sleep(1)  # Give users time to see the success message
        loading_placeholder.empty()

    def setup_qa_chain(self):
        """Set up the QA chain with the vector store."""
        if not os.path.exists("vectorstore/faiss_index"):
            st.error("Please upload and process documents first.")
            return False
            
        vectorstore = FAISS.load_local(
            "vectorstore/faiss_index",
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        prompt_template = """Use the following pieces of context to answer the question at the end. Your task is to provide a helpful and informative answer based ONLY on the provided context. Do not use any external knowledge.

If you don't know the answer or if the context doesn't contain the answer, just say that you don't have enough information to answer the question. Don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
        QA_PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True,
            output_key="answer",
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        return True

    def answer_from_internet(self, query: str) -> str:
        """Use Gemini API to answer general questions when RAG has no answer."""
        api_key = GEMINI_API_KEY
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": query}]}]
        }
        params = {"key": api_key}
        try:
            response = requests.post(url, headers=headers, params=params, json=data, timeout=15)
            response.raise_for_status()
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"Error getting answer from Gemini: {e}"

    def get_response(self, query: str) -> Dict:
        """Get response from the chatbot."""
        if not self.qa_chain:
            return {"answer": "Please process documents first."}
        
        # Handle simple greetings
        greetings = ["hello", "hi", "hey", "greetings"]
        if query.lower().strip() in greetings:
            return {"answer": "Hello! How can I help you today?"}
        
        # Check if the user is asking for a summary
        summary_keywords = ["summarize", "summary", "main points", "in short", "in brief"]
        is_summary_request = any(word in query.lower() for word in summary_keywords)

        try:
            result = self.qa_chain.invoke({"question": query})
            answer = result.get("answer", "").strip().lower()

            # Only summarize if the user explicitly asks for it
            if is_summary_request and "source_documents" in result and result["source_documents"]:
                doc_content = result["source_documents"][0].page_content
                summary_prompt = f"Summarize the following content:\n\n{doc_content[:3000]}"
                summary = self.llm.invoke(summary_prompt)
                result["answer"] = summary
                return result

            # If answer is empty or generic, use Gemini
            if not answer or answer in ["i don't know.", "no answer found.", "sorry, i couldn't find an answer in the documents."]:
                general_answer = self.answer_from_internet(query)
                return {"answer": general_answer}

            # Otherwise, return the RAG answer
            return result
        except Exception as e:
            return {"answer": f"Error: {str(e)}"}

    def get_summary(self, documents: List) -> str:
        """Generate a summary of the documents."""
        if not documents:
            return "No documents available for summarization."
        
        try:
            # Create a summarization chain
            chain = load_summarize_chain(self.llm, chain_type="map_reduce")
            
            # Run the chain on the documents
            result = chain.invoke({"input_documents": documents})
            
            return result['output_text']
        except Exception as e:
            return f"Error during summarization: {str(e)}"

def main():
    st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖", layout="wide")
    # Custom CSS for modern UI (no header bar)
    st.markdown(
        """
        <style>
        body, .stApp {background: linear-gradient(135deg, #e0e7ff 0%, #a5b4fc 100%) !important;}
        .hero {display: flex; align-items: center; justify-content: space-between; padding: 0;}
        .hero-text {max-width: 50%;}
        .hero-title {font-size: 4rem; font-weight: 800; line-height: 1.1; color: #222; margin-bottom: 1.5rem;}
        .hero-desc {font-size: 1.2rem; color: #444; margin-bottom: 2rem;}
        /* Style for Streamlit button */
        .stButton > button {
            font-size: 1.2rem !important;
            padding: 0.8rem 2.5rem !important;
            border-radius: 2rem !important;
            border: none !important;
            background: linear-gradient(90deg, #38bdf8 0%, #a78bfa 100%) !important;
            color: #fff !important;
            font-weight: 700 !important;
            box-shadow: 0 2px 12px #a5b4fc55 !important;
            transition: background 0.2s !important;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #a78bfa 0%, #38bdf8 100%) !important;
        }
        .robot-container {
            position: relative;
            width: 100%;
            height: 600px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
.oval-bg {
    position: absolute;
    width: 500px;
    height: 500px;
    background: linear-gradient(135deg, ##7F00FF 0%, #0096FF 100%); /* brighter purple & blue */
    border-radius: 50%;
    top: 50%;
    left: 57%;
    transform: translate(-50%, -50%);
    z-index: 1;
    opacity: 1;  /* Full brightness */
    filter: blur(1px); /* Soft edge */
    box-shadow: 0 0 80px rgba(147, 197, 253, 0.6); /* Bright glowing effect */
}



        .robot-img {
            position: relative;
            width: 100%;
            max-width: 400px;
            z-index: 2;
            margin-right: -20%;
        }
        .robot-img img {
            width: 100%;
            height: auto;
            object-fit: contain;
            filter: drop-shadow(0 8px 24px rgba(0,0,0,0.1));
        }
        /* Container for the hero content */
        .hero-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem 0;
        }
        /* Adjust column padding */
        .main .block-container {padding-left: 2rem; padding-right: 0;}
        
        /* Add some floating animation to the robot */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
            100% { transform: translateY(0px); }
        }
        .robot-img {
            animation: float 6s ease-in-out infinite;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Session state for switching views
    if 'show_chat' not in st.session_state:
        st.session_state.show_chat = False

    if not st.session_state.show_chat:
        # Create two columns with more space for the right column
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown(
                '''<div class="hero-title">FAQ Chatbot</div>
                <div class="hero-desc">Your intelligent assistant for document, YouTube, and website Q&A. Upload your files or paste a URL, and get instant, in-depth answers and summaries. Perfect for students, professionals, and anyone seeking knowledge from their content!</div>''',
                unsafe_allow_html=True
            )
            # Streamlit button in the hero section
            if st.button("Start Chatting", key="start_chat_btn", use_container_width=False):
                st.session_state.show_chat = True
                st.experimental_rerun()
        
        with col2:
            # Use the local image file with oval background
            st.markdown(
                '''<div class="robot-container">
                    <div class="oval-bg"></div>
                    <div class="robot-img">
                        <img src="data:image/png;base64,''' + 
                        base64.b64encode(open("E:/Projects/RAG Document summarizer using RAG/Graident Ai Robot.png", "rb").read()).decode() +
                        '''" alt="Gradient AI Robot Vectorart" />
                    </div>
                </div>''',
                unsafe_allow_html=True
            )

    else:
        # Ensure processor is initialized
        if "processor" not in st.session_state:
            st.session_state.processor = DocumentProcessor()
        # Ensure chatbot is initialized
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = Chatbot.get_instance()
        # Show the chat interface (rest of your app logic)
        if USE_OFFLINE_MODE:
            st.info("Running in offline mode with locally cached models.")
        st.title("FAQ Chatbot")
        st.write("Upload your FAQ documents (PDF/DOCX) or simply paste a YouTube/website URL in the chat bar below.")

        # Remove sidebar, move file uploader to main area
        uploaded_files = st.file_uploader(
            "Upload FAQ documents (PDF/DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        # Reset flag if new files are uploaded
        if uploaded_files:
            st.session_state.documents_processed = False

        # Only process files if not already processed
        if uploaded_files and not st.session_state.get("documents_processed", False):
            faqs_dir = "faqs"
            if os.path.exists(faqs_dir):
                shutil.rmtree(faqs_dir)
            os.makedirs(faqs_dir, exist_ok=True)
            documents = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(faqs_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                if ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                else:
                    continue
                loaded_docs = loader.load()
                if loaded_docs:
                    documents.extend(loaded_docs)
            st.session_state.documents = documents
            with st.spinner("Processing uploaded files..."):
                if os.path.exists("vectorstore"):
                    shutil.rmtree("vectorstore")
                if documents:
                    st.info(f"Found {len(documents)} document(s) to process")
                    vectorstore = st.session_state.processor.process_documents(documents)
                    if vectorstore:
                        st.session_state.chatbot.memory.clear()
                        if st.session_state.chatbot.setup_qa_chain():
                            st.session_state.documents_processed = True  # Set flag before rerun
                            st.session_state.messages = []
                            st.success("You can now ask your questions :)")
                            st.session_state.show_chat = True
                    else:
                        st.error("Failed to create vector store. Check logs for details.")
                else:
                    st.error("No valid documents or content found. Please check your inputs.")

        # Display chat messages
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        prompt = st.chat_input("How was your day? How can I help you today?")
        if prompt:
            import re
            url_pattern = r"(https?://[\w\.-]+(?:/[^\s]*)?)"
            url_match = re.search(url_pattern, prompt)
            if url_match:
                url = url_match.group(1)
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner("Processing URL..."):
                        documents = []
                        if "youtube.com" in url or "youtu.be" in url:
                            documents.extend(st.session_state.processor.load_youtube_transcript(url))
                        else:
                            documents.extend(st.session_state.processor.load_website_content(url))
                        st.session_state.documents = documents
                        if os.path.exists("vectorstore"):
                            shutil.rmtree("vectorstore")
                        if documents:
                            vectorstore = st.session_state.processor.process_documents(documents)
                            if vectorstore:
                                st.session_state.chatbot.memory.clear()
                                if st.session_state.chatbot.setup_qa_chain():
                                    st.session_state.documents_processed = True
                                    st.session_state.messages = []
                                    message_placeholder.success("You can now ask your questions :)")
                                    st.rerun()
                                else:
                                    message_placeholder.error("Failed to set up QA chain.")
                            else:
                                message_placeholder.error("Failed to create vector store.")
                        else:
                            message_placeholder.error("No valid content found at the URL.")
            else:
                if not st.session_state.get('documents_processed', False):
                    st.warning("Please upload and process documents or enter a valid URL first.")
                    st.stop()
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    with st.spinner("Thinking..."):
                        try:
                            if not hasattr(st.session_state.chatbot, 'qa_chain') or st.session_state.chatbot.qa_chain is None:
                                if not st.session_state.chatbot.setup_qa_chain():
                                    error_msg = "Failed to initialize QA chain. Please try processing documents again."
                                    message_placeholder.error(error_msg)
                                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                                    st.stop()
                            result = st.session_state.chatbot.get_response(prompt)
                            response = result.get("answer", "No answer found.")
                            message_placeholder.markdown(response)
                            if "source_documents" in result and result["source_documents"]:
                                with st.expander("Sources"):
                                    for doc in result["source_documents"]:
                                        st.markdown(f"**Source:** `{doc.metadata.get('source', 'N/A')}` (Page: {doc.metadata.get('page', 'N/A')})")
                                        st.markdown(doc.page_content)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_msg = f"Error getting response: {str(e)}"
                            message_placeholder.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()