## IntelliAgent: Multi-Agent Self-Correcting RAG Knowledge Assistant

IntelliAgent is a complete, production-grade Multi-Agent RAG-powered Knowledge Assistant designed to answer complex natural-language queries across multiple uploaded documents (PDF/TXT). Driven by a manually engineered **LangGraph StateGraph**, it couples standard similarity search with dense-sparse **Hybrid Retrieval (FAISS + BM25)**, grounded LLM generation, and an automated **Evaluator Agent** that runs factual grounding audits to eliminate hallucinations and enforce source-cited accuracy.

---

## 🛠️ System Architecture

IntelliAgent models RAG answering as an orchestrated state machine using a compiled LangGraph `StateGraph`. The system consists of 4 typed agents:

1. **Ingestion Agent**: Extracts text page-by-page from PDFs/TXTs, chunking content (recommended: `chunk_size = 800`, `chunk_overlap = 150`), generating OpenAI embeddings, and building the persistent database.
2. **Retrieval Agent**: Receives user query, queries both the dense FAISS database and the sparse lexical BM25 database, and merges their candidate documents using Reciprocal Rank Fusion (RRF).
3. **Response Agent**: Combines context, conversational memory, and user queries to generate an answer. Incorporates evaluator critique in case of correction retries.
4. **Evaluator Agent**: Evaluates the output against the original context in a self-correction loop, looking for hallucinations or missing citations, returning JSON critiques, and routing execution.

```
                  ┌───────────────────────┐
                  │   User Uploads Docs   │
                  └───────────┬───────────┘
                              ▼
                  ┌───────────────────────┐
                  │    Ingestion Agent    │
                  │ (FAISS & BM25 Index)  │
                  └───────────┬───────────┘
                              ▼
                       [User Queries]
                              │
                              ▼
                  ┌───────────────────────┐
                  │    Retrieval Agent    │
                  │   (Hybrid Search)     │
                  └───────────┬───────────┘
                              ▼
                  ┌───────────────────────┐
            ┌────►│    Response Agent     │◄───────────┐
            │     │  (Synthesis & LLM)    │            │
            │     └───────────┬───────────┘            │
            │                 ▼                        │
            │     ┌───────────────────────┐            │
            │     │    Evaluator Agent    │            │
            │     │   (JSON QA Review)    │            │
            │     └───────────┬───────────┘            │
            │                 │                        │
    [Rejected/Feedback]       │                        │
            │          [Success / Grounded]    [Retry Max = 3]
            └───────────  Is Grounded?  ───────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │   Final Cited Answer  │
                  │ (Streamlit Dashboard) │
                  └───────────────────────┘
```

---

## ⚡ Quick Start (Run within 5 Minutes)

### Prerequisites
- Python 3.10+
- OpenAI API Key (Enforced and used everywhere for LLM generation, evaluation, and embeddings)

### 1. Installation
Clone this repository (or open its workspace) and navigate to the directory:
```bash
cd https://github.com/shashi0011/IntelliAgent
```

Install python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the project root based on the provided `.env.example`:
```bash
copy .env.example .env
```

Open `.env` and fill in your OpenAI API Key:
```env
OPENAI_API_KEY=sk-proj-...
DEFAULT_LLM_PROVIDER=openai
DEFAULT_EMBEDDING_PROVIDER=openai
VECTOR_STORE_DIR=.intelliagent_db
```

### 3. Launch the Application
Run the Streamlit frontend locally:
```bash
streamlit run app.py
```
This launches a browser tab at `http://localhost:8501`.

---

## 🧪 Automated Testing & Verification
We have built an automated testing suite to verify the LangGraph state machine, hybrid dense-sparse search, and self-correcting evaluation loops without launching the Streamlit interface. 

To execute the automated verification test, run:
```bash
python scratch/verify_graph.py
```
This creates a mock document, indexes it, queries it using the StateGraph, and verifies assertions for factual grounding and source retrieval.

---

## 🐳 Docker Deployment

To build and run the entire production-grade assistant inside an isolated Docker container, run:

```bash
# Build the Docker image
docker build -t intelliagent:latest .

# Run the Docker container (mapping port 8501)
docker run -p 8501:8501 --env-file .env intelliagent:latest
```
Then navigate to `http://localhost:8501` to use the application.

---

## 💡 Example RAG Workflow
1. **Document Loading**: Upload 2 to 5 PDFs or TXT documents (e.g. quarterly financial reports, research papers, or technical guidelines) in the left sidebar.
2. **Ingestion**: Click `Process & Ingest Documents`. The progress indicator will display processing steps.
3. **Conversational RAG**: Enter a question in the chat bar (e.g. *"What is the main contribution of the paper?"* or *"Compare what Document 1 and Document 2 say about AI"*).
4. **Self-Correction & Visual Verification**: The Multi-Agent pipeline will execute. If the Evaluator Agent triggers correction, you will see a `💡 Self-Corrected` badge highlighting the number of internal refinement iterations took place.
5. **Interactive Citations**: Click the expandable `🔍 View Grounded Citations` button underneath any assistant bubble to inspect the exact source file name, page number, and text snippet that grounds the claim.

---

## 🛠️ Project Structure
- `config/settings.py`: Environment variable loading, logging setup, and unified LLM client factory.
- `loaders/doc_loader.py`: Double PDF loaders (PyMuPDF + pdfplumber backup), TXT loader, and recursive character chunk splitter.
- `embeddings/factory.py`: Enforces standard high-fidelity `OpenAIEmbeddings` (`text-embedding-3-small`).
- `vector_store/faiss_store.py`: Wraps FAISS vector database build, load, save, and clear utilities.
- `vector_store/hybrid_retriever.py`: Integrates dense FAISS retrieval with BM25 sparse keyword matching, utilizing Reciprocal Rank Fusion (RRF) and disk serialization.
- `prompts/templates.py`: Strict system instruction templates for response generation and rigorous structured JSON fact-checking.
- `agents/`: Orchestrates the individual LangGraph node agents (Ingestion, Retrieval, Response, Evaluator).
- `graph/`: Declares the typed state dictionary and compiles the `StateGraph` workflows and edges.
- `scratch/verify_graph.py`: Verification testing suites for automated testing.
- `app.py`: Premium Streamlit dashboard frontend interface.
