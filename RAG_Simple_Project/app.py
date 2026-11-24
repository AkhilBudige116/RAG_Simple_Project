"""
Simple RAG Document Assistant using LangChain
=============================================
A beginner-friendly Retrieval-Augmented Generation (RAG) app.

LangChain pipeline:
    1. Upload   -> extract_text()
    2. Chunk    -> RecursiveCharacterTextSplitter
    3. Embed    -> HuggingFaceEmbeddings
    4. Store    -> LangChain FAISS vector store (in memory)
    5. Retrieve -> similarity_search_with_score()
    6. Generate -> ChatGroq + LangChain prompt/parser chain

Run with:  streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

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
# STEP 2: LangChain text chunking
# -----------------------------------------------------------------------------
def split_document(text, filename):
    """Create LangChain Documents and split them while retaining source metadata."""
    document = Document(
        page_content=" ".join(text.split()),
        metadata={"source": filename},
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents([document])


# -----------------------------------------------------------------------------
# STEP 3: LangChain embeddings
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Load and cache LangChain's Hugging Face embedding integration."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )


# -----------------------------------------------------------------------------
# STEP 4: LangChain vector store (FAISS, in memory)
# -----------------------------------------------------------------------------
def create_vector_store(documents):
    """Embed LangChain Documents and add them to an in-memory FAISS store."""
    return FAISS.from_documents(documents, load_embedding_model())


# -----------------------------------------------------------------------------
# STEP 5: Retrieval
# -----------------------------------------------------------------------------
def retrieve_relevant_chunks(question, index, top_k=TOP_K):
    """Use the LangChain FAISS store to return the top matching source chunks."""
    matches = index.similarity_search_with_score(question, k=top_k)
    results = []
    for document, distance in matches:
        # With normalized vectors, squared L2 distance maps to cosine similarity.
        similarity = max(0.0, min(1.0, 1.0 - float(distance) / 2.0))
        results.append(
            {
                "text": document.page_content,
                "source": document.metadata.get("source", "Unknown"),
                "score": similarity,
            }
        )
    return results


# -----------------------------------------------------------------------------
# STEP 6: LLM answer generation (Groq)
# -----------------------------------------------------------------------------
def get_chat_model():
    """Create a LangChain Groq chat model using the configured API key."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is missing. Add your key to the .env file.")
    return ChatGroq(api_key=api_key, model=LLM_MODEL_NAME, temperature=0, max_retries=2)


def generate_answer(question, retrieved_chunks):
    """Run a LangChain prompt/model/parser chain over retrieved context only."""
    context = "\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in retrieved_chunks
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful document assistant. Answer only using the "
                "provided context. If the answer is not available in the context, "
                'reply exactly: "I couldn\'t find this information in your uploaded '
                'documents." Do not guess or use outside knowledge.',
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
        ]
    )
    chain = prompt | get_chat_model() | StrOutputParser()
    try:
        return chain.invoke({"context": context, "question": question})
    except Exception as error:
        raise RuntimeError(f"Groq API error: {error}")


def generate_general_answer(question):
    """Run a LangChain chat chain without retrieving or using documents."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful general-purpose AI assistant."),
            ("human", "{question}"),
        ]
    )
    chain = prompt | get_chat_model() | StrOutputParser()
    try:
        return chain.invoke({"question": question})
    except Exception as error:
        raise RuntimeError(f"Groq API error: {error}")


# -----------------------------------------------------------------------------
# Document processing (runs steps 1-4 when the button is clicked)
# -----------------------------------------------------------------------------
def process_documents(uploaded_files):
    """Extract text, create LangChain chunks and index them for all uploads."""
    all_documents = []
    processed_files = 0

    for uploaded_file in uploaded_files:
        try:
            text = extract_text(uploaded_file)
        except ValueError as error:
            st.sidebar.warning(str(error))
            continue
        all_documents.extend(split_document(text, uploaded_file.name))
        processed_files += 1

    if not all_documents:
        raise ValueError("No usable text found in the uploaded documents.")

    try:
        index = create_vector_store(all_documents)
    except Exception as error:
        raise RuntimeError(f"Failed to build the vector store: {error}")

    st.session_state.index = index
    st.session_state.chunks = all_documents
    st.session_state.num_docs = processed_files
    st.session_state.num_chunks = len(all_documents)
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
                        question, st.session_state.index
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
