# Simple RAG Document Assistant

A beginner-friendly **Retrieval-Augmented Generation (RAG)** app. Upload PDF or TXT documents, ask questions, and get answers generated only from your documents.

## Project Overview

The whole app lives in one file, `app.py`, containing both the Streamlit frontend and the RAG backend. It is designed so you can read it top to bottom and understand every stage of a RAG pipeline.

## Architecture

```
 PDF / TXT files
        |
        v
 [1] extract_text()              PyPDF / plain read
        |
        v
 [2] split_text()                500-char chunks, 100-char overlap
        |
        v
 [3] generate_embeddings()       all-MiniLM-L6-v2 -> 384-dim vectors
        |
        v
 [4] create_vector_store()       FAISS IndexFlatIP (in memory)
        |
 User question --> embed --> [5] retrieve_relevant_chunks()   top 3
                                   |
                                   v
                      [6] generate_answer()   Groq llama-3.3-70b-versatile
                                   |
                                   v
                     Answer + retrieved context + source files
```

## How the RAG Pipeline Works

1. **Upload and extract** – PDFs are read with PyPDF; TXT files are read directly. Empty or unreadable files produce clear errors.
2. **Chunk** – The text is cut into ~500-character pieces that overlap by ~100 characters, so ideas on a boundary are not lost. Each chunk remembers its filename.
3. **Embed** – Each chunk is turned into a vector (a list of numbers) that captures its meaning. Similar meanings produce similar vectors.
4. **Store** – Vectors are normalized and added to a FAISS `IndexFlatIP` index. With normalized vectors, inner product equals cosine similarity.
5. **Retrieve** – Your question is embedded with the same model, and FAISS returns the 3 most similar chunks.
6. **Generate** – The question and the 3 chunks are sent to the Groq LLM with an instruction to answer only from the context and say so when the answer is missing.

## Technology Stack

| Purpose | Tool |
|---|---|
| Language | Python |
| Frontend | Streamlit |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | FAISS (`faiss-cpu`) |
| PDF parsing | PyPDF |
| Config | python-dotenv |

## Folder Structure

```
RAG_Simple_Project/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Installation

**1. Create and activate a virtual environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

The first run downloads the embedding model (~90 MB), so it needs internet access.

**3. Configure your Groq API key**

Get a free key from https://console.groq.com, then edit `.env`:

```
GROQ_API_KEY=your_actual_key_here
```

Never commit `.env` (it is listed in `.gitignore`).

## Run the Application

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## How to Use

1. Upload one or more PDF/TXT files in the sidebar.
2. Click **Process Documents**.
3. Type a question in the chat box.
4. Read the answer, then open **Retrieved context** to see the exact snippets and source filenames.

## Example Questions to Test

- "What is this document about?"
- "Summarize the main points of the document."
- "What does the document say about [a topic you know is in it]?"
- "Who is mentioned in the document?"
- "What is the capital of Mars?" (not in your documents – the app should say the information is not available)

## Troubleshooting

- **"GROQ_API_KEY is missing"** – Put your real key in `.env` and restart Streamlit.
- **"contains no readable text"** – The PDF may be scanned images; this app does not do OCR.
- **Model decommissioned error** – Groq occasionally retires models. Change `LLM_MODEL_NAME` at the top of `app.py` to a currently supported model listed at https://console.groq.com/docs/models.
