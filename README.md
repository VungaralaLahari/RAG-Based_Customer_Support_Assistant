
#  RAG-Based Customer Support Assistant (LangGraph + HITL)

##  Overview

This project is a **Retrieval-Augmented Generation (RAG) based Customer Support Assistant** that answers user queries using information extracted from PDF documents.

It combines:

*  Semantic Search (Embeddings + ChromaDB)
*  LLM-based response generation
*  LangGraph workflow orchestration
*  Human-in-the-Loop (HITL) fallback system

The system ensures **accurate, context-aware, and reliable answers** by grounding responses in real document data.

---

##  Problem Statement

Traditional customer support systems face challenges such as:

* Slow manual document search
* Limited chatbot intelligence
* Keyword-based retrieval (low accuracy)
* Lack of contextual understanding

This project solves these issues using **RAG architecture**, enabling intelligent question answering from documents.

---

##  Objective

To build an AI-powered assistant that:

* Reads and processes PDF documents
* Retrieves relevant context using semantic search
* Generates accurate answers using an LLM
* Uses workflow control for structured execution
* Escalates uncertain cases to human support

---

##  System Architecture

```
User Query
   ↓
Embedding Generation
   ↓
Vector Database (ChromaDB)
   ↓
Relevant Chunk Retrieval
   ↓
LLM (Response Generation)
   ↓
Confidence Check
   ↓
LangGraph Workflow
   ↓
HITL (if needed) → Human Agent
   ↓
Final Response
```

---

##  Tech Stack

* **Python** – Core implementation
* **LangChain** – RAG pipeline framework
* **LangGraph** – Workflow orchestration
* **ChromaDB** – Vector database for embeddings
* **Sentence Transformers** – Text embeddings
* **PyPDF** – PDF parsing

---

##  Project Workflow

### 1. Document Processing

* Load PDF file
* Extract text content
* Split into chunks

### 2. Embedding Generation

* Convert text chunks into vector embeddings
* Store in ChromaDB

### 3. Query Processing

* User inputs query
* Query is converted into embeddings

### 4. Retrieval

* Top relevant chunks are fetched using similarity search

### 5. Response Generation

* LLM generates answer using retrieved context

### 6. Workflow Control (LangGraph)

* Manages flow between retrieval, generation, and output

### 7. HITL (Human-in-the-Loop)

* If confidence is low, query is escalated to human agent

---

##  LangGraph Workflow

Nodes:

* `retrieve` → Fetch relevant document chunks
* `generate` → Generate AI response
* `hitl` → Human escalation (if needed)
* `output` → Final response delivery

---

##  Key Features

*  PDF-based knowledge ingestion
*  Semantic search using embeddings
*  Context-aware LLM responses
*  Structured workflow using LangGraph
*  Human fallback system (HITL)
*  Confidence-based decision making

---

##  Limitations

* Depends on quality of input PDF
* Retrieval accuracy varies with chunking strategy
* LLM responses may still have minor hallucinations
* Requires compute for embedding generation

---

##  Future Improvements

*  Web-based UI (Streamlit / React)
*  Multi-document support
*  Memory-based conversational system
*  Feedback loop for improving responses
*  Faster retrieval using hybrid search

---

##  How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebook / script in Colab or local environment

# Upload PDF when prompted
# Start asking questions
```

---

##  Demo

-  Video Demo: *Add Google Drive Link Here*
-  LinkedIn Post: *Add Link Here*

---

##  Author
```
V.LAHARI
IN226023002
Innomatics Internship Project – RAG-Based AI System
```

---

##  Conclusion

This project demonstrates how modern AI systems combine:

* Retrieval systems
* Language models
* Workflow orchestration
* Human fallback mechanisms

to build **real-world intelligent assistants**.


