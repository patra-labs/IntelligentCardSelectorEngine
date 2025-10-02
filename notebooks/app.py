# ========================
# APP: Intelligent Card Selector Engine
# Author: Soumya Patra
# ========================

import os
import json
import textwrap
from pathlib import Path
from datetime import datetime

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
import google.generativeai as genai

# ========================
# CONFIGURATION
# ========================
from pathlib import Path

# ========================
# Folders
# ========================

# Base folder for PDFs (deployment-safe)
BASE_FOLDER = Path(__file__).parent / "Data" / "Cards"       # PDF input folder
if not BASE_FOLDER.exists():
    BASE_FOLDER.mkdir(parents=True, exist_ok=True)            # Ensure folder exists

# Output folder for JSONL & Chroma DB (writable in deployment)
OUTPUT_FOLDER = Path(__file__).parent / "output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ========================
# Files
# ========================

JSONL_OUTPUT = OUTPUT_FOLDER / "documents.jsonl"
CHROMA_DB_PATH = OUTPUT_FOLDER / "chroma_store"

# ========================
# Chroma / Embedding settings
# ========================

COLLECTION_NAME = "card_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
N_RESULTS = 5
TEXT_WIDTH = 80

# Chunking settings
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100

# Load Google API Key from Streamlit secrets
api_key = st.secrets["api"]["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# Gemini 2.5 Flash Model
models = genai.GenerativeModel("gemini-2.5-flash")

# Today's date
today = datetime.now().strftime("%B %d, %Y")

# # Base context about credit cards (used for grounding LLM)
# CONTEXT_FILE = (
#     "Credit card benefits can include cashback, travel rewards, low interest rates, "
#     "and no annual fees. Different cards offer different perks, so choose one "
#     "that aligns with your spending habits and financial goals."
# )

# ========================
# UTILITY FUNCTIONS
# ========================

def extract_text_by_page(pdf_path: Path):
    """Yield text page by page from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            yield page_num, text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def savejsonl():
    """Read all PDFs, split into chunks, and save as JSONL."""
    with open(JSONL_OUTPUT, "w", encoding="utf-8") as f:
        for pdf_file in BASE_FOLDER.rglob("*.pdf"):
            for page_num, page_text in extract_text_by_page(pdf_file):
                if not page_text:
                    continue

                chunks = chunk_text(page_text)

                for chunk_idx, chunk in enumerate(chunks):
                    card_name = pdf_file.parents[1].name  # e.g., DiscoverIt
                    date = pdf_file.parents[0].name       # e.g., 2025-09-01
                    record = {
                        "card": card_name,
                        "date": date,
                        "filename": pdf_file.name,
                        "path": str(pdf_file.resolve()),
                        "page": page_num,
                        "chunk_index": chunk_idx,
                        "text": chunk
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

def loadchroma():
    """Load JSONL into Chroma DB with embeddings."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    with open(JSONL_OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunk_id = (
                f"{record['card']}_{record['date']}_{record['filename']}_"
                f"page{record['page']}_chunk{record['chunk_index']}"
            )
            collection.add(
                ids=[chunk_id],
                documents=[record["text"]],
                metadatas=[{
                    "card": record["card"],
                    "date": record["date"],
                    "filename": record["filename"],
                    "path": record["path"],
                    "page": record["page"],
                    "chunk_index": record["chunk_index"]
                }]
            )

def refresh_chroma():
    """Refresh Chroma DB by regenerating JSONL and loading embeddings."""
    savejsonl()
    loadchroma()

# ========================
# CONNECT TO CHROMA
# ========================

client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

# ========================
# QUERY & SUMMARIZE
# ========================

def query_and_summarize(question: str, n_results: int = N_RESULTS, width: int = TEXT_WIDTH):
    """Query Chroma DB and generate concise summary using Gemini 2.5 Flash."""
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["metadatas", "documents", "distances"]
    )

    # Combine and format results
    combined_text = ""
    for idx, doc in enumerate(results['documents'][0]):
        metadata = results['metadatas'][0][idx]
        score = 1 - results['distances'][0][idx]  # higher = more relevant
        header = (
            f"[Card: {metadata['card']} | Date: {metadata['date']} | "
            f"File: {metadata['filename']} | Page: {metadata['page']} | "
            f"Chunk: {metadata['chunk_index']} | Score: {score:.3f}]"
        )
        wrapped_text = textwrap.fill(doc, width=width)
        combined_text += f"{header}\n{wrapped_text}\n\n"

    # Prompt for LLM
    summary_prompt = (
        f"Today's date is {today}.\n"
        f"Summarize the following information to answer the question:\n\n"
        f"Question: {question}\n\n"
        f"{combined_text}\n\n"
        f"Provide a short, clear recommendation as bullet points. "
        f"Order multiple cards by relevance and keep it concise."
        f"limit the summary to provided text, not from outside knowledge.\n"
    )

    response = models.generate_content(
        [
            {"role": "model", "parts": "You summarize credit card benefits clearly and concisely."},
            {"role": "user", "parts": summary_prompt}
        ],
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=500,
            temperature=0.7
        )
    )

    return combined_text, response.text.strip()

# ========================
# STREAMLIT APP
# ========================

# Page setup
st.set_page_config(
    page_title="Intelligent Card Selector Engine",
    page_icon="💳",
    layout="centered"
)

# Header with explanatory note
st.title("💳 Intelligent Card Selector Engine")
st.markdown(
    """
    This app helps you instantly determine which credit card to use to maximize cashback and rewards.
    No more guessing in the store! Simply type your spending scenario and get a ranked recommendation.
    """
)


# ===== NOTE =====
# File upload and Chroma DB refresh functionality is currently disabled.
# The database is expected to be preloaded with card PDFs in ../Data/Cards.
# ==================
# ============TRIAL 1==========
# Sidebar: commented out staging
st.sidebar.header("Chroma DB Options")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
if st.sidebar.button("Upload Files"):
    if uploaded_files:
        st.sidebar.success(f"{len(uploaded_files)} uploaded")
    else:
        st.sidebar.warning("No files selected")
if st.sidebar.button("Refresh Chroma DB"):
    refresh_chroma()
    st.sidebar.success("Chroma DB refreshed")
# ============================


# User input
user_question = st.text_area("Type your question here...", height=100)

if st.button("Get Answer"):
    if user_question.strip():
        combined_text, answer = query_and_summarize(user_question)
        st.success(combined_text)
        st.success(answer)
    else:
        st.warning("⚠️ Please type a question.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 14px;'>
        Developed by Soumya Patra | 
        <a href="https://github.com/patra-labs" target="_blank">GitHub</a> | 
        <a href="https://www.linkedin.com/in/patrasoumya/" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)
