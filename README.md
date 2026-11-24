# Simple RAG Document Assistant

A Streamlit app for asking questions about your PDF and TXT files, with an optional general AI chat mode powered by Groq.

## Features

- **Document Mode (RAG):** Upload and process PDF/TXT files, retrieve relevant passages with FAISS, and generate answers grounded in those passages. Retrieved context and source filenames are shown with each answer. If the answer is unavailable, the app responds: “I couldn't find this information in your uploaded documents.”
- **General AI Mode:** Ask Groq questions without uploading or processing documents. This mode does not retrieve document chunks.
- **In-memory document index:** Uploaded files are processed for the current app session.

## Requirements

- Python 3.14 or newer
- A Groq API key
- Internet access on first run to download the sentence-transformers embedding model

## Setup

Run these commands from the repository root:

```bash
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install the app dependencies:

```bash
pip install -r RAG_Simple_Project/requirements.txt
```

Create `RAG_Simple_Project/.env` and add your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Keep your real API key private. `.env` files and virtual environments are excluded from Git.

## Run

From the repository root, run:

```bash
streamlit run RAG_Simple_Project/app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Use the app

### Document Mode (RAG)

1. Select **Document Mode (RAG)** in the sidebar.
2. Upload one or more PDF or TXT files and select **Process Documents**.
3. Ask a question about the uploaded material.
4. Expand **Retrieved context** to see the passages used to answer and the source filenames.

### General AI Mode

1. Select **General AI Mode** in the sidebar.
2. Ask a question in the chat box. No documents are needed or used.

## How document retrieval works

The app extracts text from PDF/TXT files into LangChain `Document` objects, splits them into overlapping chunks with `RecursiveCharacterTextSplitter`, embeds each chunk with `HuggingFaceEmbeddings` using `sentence-transformers/all-MiniLM-L6-v2`, and stores normalized vectors in LangChain's FAISS vector store. For each question in Document Mode, FAISS retrieves up to three similar chunks, and a LangChain prompt/model/parser chain sends those passages and the question to Groq through `ChatGroq`. The configured Groq model is `openai/gpt-oss-120b`. General AI Mode uses a separate LangChain chat chain and skips retrieval.

## Project layout

```text
.
├── main.py                         # Small Python entry point
├── pyproject.toml
├── requirements.txt
└── RAG_Simple_Project/
    ├── app.py                      # Streamlit UI and RAG implementation
    ├── requirements.txt
    └── README.md
```

## Troubleshooting

- **GROQ_API_KEY is missing:** Check that `RAG_Simple_Project/.env` contains `GROQ_API_KEY=...`, then restart Streamlit.
- **No readable text in a PDF:** Scanned/image-only PDFs are not OCR processed. Use a text-based PDF or a TXT file.
- **Groq model or access error:** Confirm the API key has access to the configured model. Update `LLM_MODEL_NAME` in `RAG_Simple_Project/app.py` if you choose another model available to your account.
