"""
Simple RAG Document Assistant
=============================
A beginner-friendly Retrieval-Augmented Generation (RAG) app.

Pipeline (all in this one file):
    1. Upload   -> extract_text()
    2. Chunk    -> split_text()
    3. Embed    -> generate_embeddings()
    4. Store    -> create_vector_store()      (FAISS, in memory)
    5. Retrieve -> retrieve_relevant_chunks() (top 3)
    6. Generate -> generate_answer()          (Groq LLM)

Run with:  streamlit run app.py
"""

import os

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()  # Reads GROQ_API_KEY from the .env file

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b"
CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 100   # characters shared between neighbouring chunks
TOP_K = 3             # number of chunks retrieved per question


# -----------------------------------------------------------------------------
# STEP 1: Document upload -> text extraction
# -----------------------------------------------------------------------------
def extract_text(uploaded_file):
    """Return the text of an uploaded PDF or TXT file.

    Raises ValueError for unsupported formats, empty files, or read errors.
    """
    filename = uploaded_file.name
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        try:
            reader = PdfReader(uploaded_file)
            pages = [(page.extract_text() or "") for page in reader.pages]
            text = "\n".join(pages)
        except Exception as error:
            raise ValueError(f"Could not read PDF '{filename}': {error}")
    elif lower_name.endswith(".txt"):
        try:
            raw_bytes = uploaded_file.read()
            text = raw_bytes.decode("utf-8", errors="ignore")
        except Exception as error:
            raise ValueError(f"Could not read TXT '{filename}': {error}")
    else:
        raise ValueError(f"Unsupported file format: '{filename}'. Use PDF or TXT.")

    if not text.strip():
        raise ValueError(
            f"'{filename}' contains no readable text "
            "(it may be empty or a scanned PDF)."
        )
    return text


# -----------------------------------------------------------------------------
# STEP 2: Text chunking
# -----------------------------------------------------------------------------
def split_text(text, filename, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of roughly `chunk_size` characters.

    Overlap keeps sentences that fall on a boundary intact in at least one chunk.
    Each chunk is stored with its filename as metadata.
    """
    text = " ".join(text.split())  # collapse whitespace/newlines
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append({"text": piece, "source": filename})
        if start + chunk_size >= len(text):
            break
    return chunks


# -----------------------------------------------------------------------------
# STEP 3: Embeddings
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Load the Sentence Transformer once and reuse it across reruns."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def generate_embeddings(texts):
    """Convert a list of strings into a 2D numpy array of embeddings.

    An embedding is a list of numbers (384 for MiniLM) that represents the
    MEANING of a text. Texts with similar meaning get similar vectors, which
    lets us search by meaning instead of exact keywords.
    """
    model = load_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(embeddings, dtype="float32")


# -----------------------------------------------------------------------------
# STEP 4: Vector database (FAISS, in memory)
# -----------------------------------------------------------------------------
def create_vector_store(chunks):
    """Embed all chunks and store them in a FAISS index.

    We L2-normalise the vectors and use IndexFlatIP (inner product), which
    makes the inner product equal to cosine similarity.
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index  # chunk text + filenames are kept alongside in session state


# -----------------------------------------------------------------------------
# STEP 5: Retrieval
# -----------------------------------------------------------------------------
def retrieve_relevant_chunks(question, index, chunks, top_k=TOP_K):
    """Embed the question and return the top_k most similar chunks."""
    query_embedding = generate_embeddings([question])
    faiss.normalize_L2(query_embedding)  # same normalisation as the chunks

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append(
            {
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "score": float(score),
            }
        )
    return results


# -----------------------------------------------------------------------------
# STEP 6: LLM answer generation (Groq)
# -----------------------------------------------------------------------------
def generate_answer(question, retrieved_chunks):
    """Ask the Groq LLM to answer using ONLY the retrieved context."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is missing. Add your key to the .env file.")

    context = "\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in retrieved_chunks
    )

    system_prompt = (
        "You are a helpful document assistant. "
        "Answer only using the provided context. If the answer is not available "
        'in the context, reply exactly: "I couldn\'t find this information in your '
        'uploaded documents." Do not guess or use outside knowledge.'
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,  # deterministic, avoids made-up details
        )
        return response.choices[0].message.content
    except Exception as error:
        raise RuntimeError(f"Groq API error: {error}")


def generate_general_answer(question):
    """Ask Groq a general question without using uploaded documents."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is missing. Add your key to the .env file.")

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful general-purpose AI assistant."},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as error:
        raise RuntimeError(f"Groq API error: {error}")


# -----------------------------------------------------------------------------
# Document processing (runs steps 1-4 when the button is clicked)
# -----------------------------------------------------------------------------
def process_documents(uploaded_files):
    """Run extraction, chunking, embedding and indexing for all uploads."""
    all_chunks = []
    processed_files = 0

    for uploaded_file in uploaded_files:
        try:
            text = extract_text(uploaded_file)
        except ValueError as error:
            st.sidebar.warning(str(error))
            continue
        all_chunks.extend(split_text(text, uploaded_file.name))
        processed_files += 1

    if not all_chunks:
        raise ValueError("No usable text found in the uploaded documents.")

    try:
        index = create_vector_store(all_chunks)
    except Exception as error:
        raise RuntimeError(f"Failed to build the vector store: {error}")

    st.session_state.index = index
    st.session_state.chunks = all_chunks
    st.session_state.num_docs = processed_files
    st.session_state.num_chunks = len(all_chunks)
    st.session_state.messages = []  # new documents -> fresh conversation


# -----------------------------------------------------------------------------
# Streamlit frontend
# -----------------------------------------------------------------------------
def show_sources(sources):
    """Show retrieved snippets in an expander plus the source filenames."""
    with st.expander("Retrieved context"):
        for i, chunk in enumerate(sources, start=1):
            st.markdown(
                f"**{i}. {chunk['source']}** (similarity: {chunk['score']:.2f})"
            )
            st.write(chunk["text"])
    names = sorted({chunk["source"] for chunk in sources})
    st.caption("Sources: " + ", ".join(names))


def main():
    st.set_page_config(page_title="Simple RAG Document Assistant", page_icon="📄")

    # Session state keeps the in-memory vector store between reruns
    defaults = {
        "index": None,
        "chunks": [],
        "messages": [],
        "num_docs": 0,
        "num_chunks": 0,
        "status": "Waiting for documents",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    st.title("Simple RAG Document Assistant")
    st.write(
        "Choose Document Mode to ask questions about uploaded files, or General "
        "AI Mode to ask Groq without uploading documents."
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        st.error("GROQ_API_KEY is missing. Add your key to the .env file and restart.")

    # ---- Sidebar ----
    with st.sidebar:
        mode = st.radio(
            "Response mode",
            ["Document Mode (RAG)", "General AI Mode"],
            help="Document Mode answers from uploaded files. General AI Mode does not use documents.",
        )

        if mode == "Document Mode (RAG)":
            st.header("Documents")
            uploaded_files = st.file_uploader(
                "Upload PDF or TXT files",
                type=["pdf", "txt"],
                accept_multiple_files=True,
            )

            if st.button("Process Documents"):
                if not uploaded_files:
                    st.warning("Please upload at least one document first.")
                else:
                    try:
                        with st.spinner("Processing documents..."):
                            process_documents(uploaded_files)
                        st.session_state.status = "Documents processed successfully"
                    except Exception as error:
                        st.session_state.status = "Processing failed"
                        st.error(str(error))

            st.metric("Documents processed", st.session_state.num_docs)
            st.metric("Chunks generated", st.session_state.num_chunks)
            st.info(f"Status: {st.session_state.status}")
        else:
            st.info("Ask a question directly. Uploaded documents are not used.")

    # ---- Chat history ----
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                show_sources(message["sources"])

    # ---- User question ----
    prompt = (
        "Ask a question about your documents"
        if mode == "Document Mode (RAG)"
        else "Ask a general question"
    )
    question = st.chat_input(prompt)
    if question is None:
        return

    if not question.strip():
        st.warning("Please enter a question.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            if mode == "Document Mode (RAG)":
                if st.session_state.index is None:
                    answer = "I couldn't find this information in your uploaded documents."
                    st.warning(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "mode": mode}
                    )
                    return
                with st.spinner("Searching documents and generating answer..."):
                    retrieved = retrieve_relevant_chunks(
                        question, st.session_state.index, st.session_state.chunks
                    )
                    if not retrieved:
                        answer = "I couldn't find this information in your uploaded documents."
                    else:
                        answer = generate_answer(question, retrieved)
            else:
                retrieved = []
                with st.spinner("Generating general answer..."):
                    answer = generate_general_answer(question)
        except Exception as error:
            st.error(str(error))
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Error: {error}"}
            )
            return

        st.write(answer)
        if retrieved:
            show_sources(retrieved)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": retrieved, "mode": mode}
        )


if __name__ == "__main__":
    main()
