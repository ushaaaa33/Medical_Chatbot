## 🔍 Overview

MediBot is a conversational AI assistant designed for medical question-answering. It leverages a **two-stage RAG pipeline**:

1. **Offline indexing** — Medical PDFs are chunked, embedded, and stored in a local FAISS vector database.
2. **Online retrieval** — At query time, relevant chunks are retrieved and injected into a strict prompt before being sent to the LLM.

This ensures the model answers only from verified source material, making it suitable for controlled medical information retrieval.

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════╗
║              STAGE 1 — OFFLINE INDEXING PIPELINE            ║
║                   (create_memory_for_llm.py)                ║
║                                                              ║
║   PDF Files                                                  ║
║      │                                                       ║
║      ▼                                                       ║
║   PyPDFLoader ──► RecursiveCharacterTextSplitter             ║
║   (Load Pages)     (chunk_size=500, overlap=50)              ║
║                            │                                 ║
║                            ▼                                 ║
║                   HuggingFace Embeddings                     ║
║                 (all-MiniLM-L6-v2, runs locally)             ║
║                            │                                 ║
║                            ▼                                 ║
║                   FAISS Vector Store                         ║
║                  (saved to vectorstore/db_faiss/)            ║
╚══════════════════════════════════════════════════════════════╝
                             │
                    (persisted to disk)
                             │
╔══════════════════════════════════════════════════════════════╗
║               STAGE 2 — QUERY PIPELINE                      ║
║        (connect_memory_with_llm.py  /  medibot.py)          ║
║                                                              ║
║   User Query                                                 ║
║      │                                                       ║
║      ▼                                                       ║
║   Embed Query ──► FAISS Similarity Search ──► Top-3 Chunks  ║
║                                                    │         ║
║                                                    ▼         ║
║                                          Custom Prompt       ║
║                                          Template (strict)   ║
║                                                    │         ║
║                                                    ▼         ║
║                                       Groq LLaMA 3.1 8B     ║
║                                                    │         ║
║                                                    ▼         ║
║                                     Answer + Source Docs     ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
Medical_Chatbot/
│
├── data/                           # Place your medical PDF files here
│   └── *.pdf
│
├── vectorstore/
│   └── db_faiss/                   # FAISS index generated after Step 1
│       ├── index.faiss
│       └── index.pkl
│
├── create_memory_for_llm.py        # Stage 1: Build vector store from PDFs
├── connect_memory_with_llm.py      # Stage 2: CLI-based RAG query interface
├── medibot.py                      # Stage 3: Streamlit web application
│
├── requirements.txt                # Pinned Python dependencies
├── Pipfile                         # Pipenv environment file
├── Pipfile.lock
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | [Groq](https://groq.com/) — `llama-3.1-8b-instant` (free, fast inference) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace (runs locally) |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) (CPU, local) |
| **RAG Framework** | [LangChain](https://www.langchain.com/) `v0.3.26` |
| **PDF Parsing** | LangChain `PyPDFLoader` + `DirectoryLoader` |
| **Text Splitting** | `RecursiveCharacterTextSplitter` |
| **Web UI** | [Streamlit](https://streamlit.io/) `v1.46.1` |
| **Env Management** | `python-dotenv` |
| **Package Manager** | Pipenv / pip |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- A free [Groq API key](https://console.groq.com) — for LLM inference
- A [HuggingFace token](https://huggingface.co/settings/tokens) — for embedding model access

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/ushaaaa33/Medical_Chatbot.git
cd Medical_Chatbot
```

**2. Create and activate a virtual environment**

Using `venv`:
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

Or using Pipenv (recommended, since `Pipfile` is included):
```bash
pip install pipenv
pipenv install
pipenv shell
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

---

## 📖 Usage

### Step 1 — Build the Knowledge Base

Add your medical PDF files to the `data/` directory, then run:

```bash
python create_memory_for_llm.py
```

**Expected output:**
```
Length of PDF pages: 120
Length of Text Chunks: 543
```

This generates the `vectorstore/db_faiss/` directory. Run this step again whenever you add or update PDFs.

---

### Step 2 — Test via CLI (Optional)

A lightweight terminal interface for quickly testing the RAG pipeline:

```bash
python connect_memory_with_llm.py
```

```
Write Query Here: What are the symptoms of type 2 diabetes?

RESULT: Type 2 diabetes symptoms include increased thirst, frequent urination...
SOURCE DOCUMENTS: [Document(...), Document(...), Document(...)]
```

---

### Step 3 — Launch the Web App

```bash
streamlit run medibot.py
```

Open your browser at **`http://localhost:8501`**

The app provides a full chat interface with conversation history. Each response includes the answer and the source document chunks used to generate it.

---

## ⚙️ Configuration

Key parameters you can tune directly in the source files:

| Parameter | File | Default | Description |
|---|---|---|---|
| `chunk_size` | `create_memory_for_llm.py` | `500` | Characters per text chunk |
| `chunk_overlap` | `create_memory_for_llm.py` | `50` | Character overlap between consecutive chunks |
| `k` | `connect_memory_with_llm.py`, `medibot.py` | `3` | Number of chunks retrieved per query |
| `temperature` | `connect_memory_with_llm.py`, `medibot.py` | `0.5` | LLM response randomness (`0` = deterministic) |
| `model` | `connect_memory_with_llm.py`, `medibot.py` | `llama-3.1-8b-instant` | Groq-hosted model |
| `embedding_model` | All files | `all-MiniLM-L6-v2` | HuggingFace sentence transformer |
| `DATA_PATH` | `create_memory_for_llm.py` | `data/` | Directory containing source PDFs |
| `DB_FAISS_PATH` | All files | `vectorstore/db_faiss` | Path to the FAISS index |

---

## 🧠 How RAG Works in This Project

**Retrieval-Augmented Generation (RAG)** grounds the LLM's answers in real source documents rather than relying on its pre-trained knowledge.

**`create_memory_for_llm.py` — Indexing**
- Loads all PDFs from `data/` using `PyPDFLoader`
- Splits them into 500-character chunks with 50-character overlap to preserve context at chunk boundaries
- Converts each chunk into a 384-dimensional vector using `all-MiniLM-L6-v2` (runs entirely locally — no data leaves your machine at this stage)
- Saves the vector index to disk with FAISS for fast approximate nearest-neighbour search

**`connect_memory_with_llm.py` — CLI Query**
- Loads the persisted FAISS index
- Embeds the user's query using the same model
- Retrieves the top 3 most semantically similar chunks
- Injects them into a strict prompt template that explicitly instructs the LLM not to answer from outside the provided context
- Sends the composed prompt to Groq's `llama-3.1-8b-instant` and prints the result

**`medibot.py` — Streamlit App**
- Same RAG logic as above, wrapped in an interactive chat UI
- Uses `@st.cache_resource` to load the FAISS index once per session, avoiding repeated disk reads
- Maintains full conversation history in `st.session_state`
- Displays source document chunks alongside every answer for full transparency

**Why FAISS over a cloud vector database?**
Running FAISS locally keeps sensitive medical documents off third-party servers, which is an important consideration in healthcare contexts.

---

## ⚠️ Disclaimer

MediBot is intended for **educational and research purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified and licensed healthcare provider before making any medical decisions. The accuracy of responses depends entirely on the quality and coverage of the source PDFs provided.

---

<p align="center">
  Built with ❤️ using LangChain · FAISS · Groq · Streamlit
</p>