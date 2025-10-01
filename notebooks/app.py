# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import textwrap
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pdfplumber
from pathlib import Path
import json
import toml

# Go one level up from Notebook/ → into Data/Cards
BASE_FOLDER = Path("..") / "Data" / "Cards"

# Output folder (relative to Notebook/)
OUTPUT_FOLDER = Path("..") / "output"
OUTPUT_FOLDER.mkdir(exist_ok=True)  # create folder if it doesn’t exist

# JSONL file path
JSONL_OUTPUT = OUTPUT_FOLDER / "documents.jsonl"

# Chunking parameters
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100

def extract_text_by_page(pdf_path: Path):
    """Yield text page by page from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            yield page_num, text.strip()

def chunk_text(text, chunk_size=300, overlap=100):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:  # skip empty
            chunks.append(chunk)
    return chunks

def savejsonl():
    with open(JSONL_OUTPUT, "w", encoding="utf-8") as f:
        for pdf_file in BASE_FOLDER.rglob("*.pdf"):
            for page_num, page_text in extract_text_by_page(pdf_file):
                if not page_text:
                    continue

                chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

                for chunk_idx, chunk in enumerate(chunks):
                    card_name = pdf_file.parents[1].name  # e.g. DiscoverIt
                    date = pdf_file.parents[0].name       # e.g. 2025-09-01

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
    # ---------------- SETTINGS ----------------
    JSONL_FILE = Path("..") / "output" / "documents.jsonl"  # adjust path from Notebook/
    CHROMA_DB_PATH = Path("..") / "output" / "chroma_store"
    COLLECTION_NAME = "card_docs"

    # Embedding model
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

    # Initialize Chroma with persistent storage
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # Create or get collection with embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    # Load JSONL and insert into Chroma
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            # Unique ID for each chunk
            chunk_id = f"{record['card']}_{record['date']}_{record['filename']}_page{record['page']}_chunk{record['chunk_index']}"

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
    savejsonl()
    loadchroma()

# Load variables from .env
# load_dotenv()
api_key = st.secrets["api"]["GOOGLE_API_KEY"]

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key=api_key)
models = genai.GenerativeModel("gemini-2.5-flash")

from datetime import datetime
today = datetime.now().strftime("%B %d, %Y")

# ---------------- SETTINGS ----------------
CHROMA_DB_PATH = Path("..") / "output" / "chroma_store"
COLLECTION_NAME = "card_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
N_RESULTS = 5
TEXT_WIDTH = 80
# ------------------------------------------


# Connect to Chroma
client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

context_file = "Credit card benefits can include cashback, travel rewards, low interest rates, and no annual fees. Different cards offer different perks, so it's important to choose one that aligns with your spending habits and financial goals."

def query_and_summarize(question: str, n_results: int = N_RESULTS, width: int = TEXT_WIDTH):
    """
    Query Chroma, return top chunks with metadata, wrapped text, scores,
    and generate a concise summary using OpenAI LLM.
    """
    # query_embedding = embedding_fn.embed_query(question)

    # # Ensure it is a list of floats
    # if hasattr(query_embedding, "tolist"):  # e.g., numpy array
    #     query_embedding = query_embedding.tolist()

    results = collection.query(
        query_texts=[question],
        # query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "documents", "distances"]
    )

    # Wrap text and show relevance scores
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
    
    # Define your prompt
    summary_prompt = (
        f"Today's date is {today}.\n"
        f"Use this date as context when answering.\n\n"
        f"To understand credit card benefits, use this as a base context - {context_file}\n\n"
        f"Summarize the following information to answer the question:\n\n"
        f"Question: {question}\n\n"
        f"{combined_text}\n\n"
        f"Provide a short, clear recommendation if applicable. Also point it as bullet points.\n"
        f"Not too many words"
        f"If multiple cards are mentioned, order them based on relevance and keep it concise"
        )

    # Create the model (Gemini 2.5 Flash)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Generate response
    response = model.generate_content(
        [
            {"role": "model", "parts": "You summarize credit card benefits clearly and concisely."},
            {"role": "user", "parts": summary_prompt}
        ],
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=300,
            temperature=0.7
        )
    )

    # Extract text
    summary = response.text.strip()
        
    return combined_text, summary

# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="Intelligent Card Selector Engine",
    page_icon="💳",
    layout="centered",
)

# ========================
# Sidebar for Chroma DB Options
# ========================
st.sidebar.header("Chroma DB Options")

# File upload
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files to add to Chroma DB", type=["pdf"], accept_multiple_files=True
)
if st.sidebar.button("Upload Files"):
    if uploaded_files:
        st.sidebar.success(f"✅ {len(uploaded_files)} file(s) uploaded (demo).")
    else:
        st.sidebar.warning("⚠️ No files selected.")

# Refresh DB
if st.sidebar.button("Refresh Chroma DB"):
    refresh_chroma()
    st.sidebar.success("✅ Chroma DB refreshed.")

st.sidebar.markdown("---")
st.sidebar.info("This is a demo tool. Backend functionality will be added in future versions.")

# ========================
# Main Content
# ========================
st.title("💳 Intelligent Card Selector Engine")
st.markdown(
    """
    **Demo Tool**  
    Ask questions about credit cards to get recommendations.
    """
)

st.header("Ask a Question")
user_question = st.text_area("Type your question here...", height=100)

#question = "Which card should I use for groceries to get the best cashback?"
combined_text, answer = query_and_summarize(user_question, 5, 300)

if st.button("Get Answer"):
    if user_question.strip() != "":
        st.success(answer)
    else:
        st.warning("⚠️ Please type a question.")


# ========================
# Footer
# ========================
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
# ========================