# =============================================================================
# RAG-BASED CUSTOMER SUPPORT ASSISTANT 
# =============================================================================

# NOTE:
# Run these in terminal (NOT inside Python file):
# pip install langchain-huggingface langchain langchain-core langchain-community
# pip install langchain-text-splitters chromadb pypdf2 sentence-transformers
# pip install langgraph streamlit pillow python-multipart


# =============================================================================
# STEP 1: UPLOAD PDF (Google Colab version)
# =============================================================================

from google.colab import files

print("UPLOAD TechGadget PDF:")
uploaded = files.upload()

# Detect PDF filename
pdf_files = [f for f in uploaded.keys() if f.lower().endswith('.pdf')]

if pdf_files:
    pdf_name = pdf_files[0]
    print(f"Found: {pdf_name}")
else:
    print("No PDF uploaded!")
    raise ValueError("Upload PDF first")


# =============================================================================
# STEP 2: KNOWLEDGE BASE PIPELINE
# =============================================================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load PDF
loader = PyPDFLoader(pdf_name)
docs = loader.load()
print(f"{len(docs)} pages loaded")

# Chunking
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"{len(chunks)} chunks created")

# Embeddings + Vector DB
embedding = HuggingFaceEmbeddings()
db = Chroma.from_documents(chunks, embedding)

print("TechGadget DB ready!")


# =============================================================================
# STEP 3: LANGGRAPH WORKFLOW + HITL
# =============================================================================

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator


# -----------------------------
# STATE STRUCTURE
# -----------------------------
class State(TypedDict):
    query: str
    docs: Annotated[List[str], operator.add]
    answer: str
    confidence: float
    needs_hitl: bool


# -----------------------------
# RETRIEVAL NODE
# -----------------------------
def retrieve(state: State):
    query = state["query"]
    results = db.similarity_search(query, k=2)
    docs = [doc.page_content for doc in results]

    state["docs"] = docs
    print(f"Found {len(docs)} relevant sections")
    return state


# -----------------------------
# GENERATION NODE
# -----------------------------
def generate(state: State):
    docs_text = "\n".join(state["docs"])
    length = len(docs_text)

    if length < 100:
        state["confidence"] = 0.4
        state["answer"] = "Limited info found in documents"
    else:
        state["confidence"] = min(0.9, 0.6 + length / 500)
        state["answer"] = f"TechGadget Policy:\n{docs_text[:250]}..."

    state["needs_hitl"] = state["confidence"] < 0.7

    print(f"Confidence: {state['confidence']:.2f}")
    return state


# -----------------------------
# HUMAN-IN-THE-LOOP NODE
# -----------------------------
def hitl(state: State):
    if state["needs_hitl"]:
        print("\nHUMAN ESCALATION TRIGGERED!")
        print(f"Query: {state['query']}")

        human_response = input("Support Agent Response: ")

        state["answer"] = f"AGENT: {human_response}"
        state["confidence"] = 1.0

    return state


# -----------------------------
# OUTPUT NODE
# -----------------------------
def output(state: State):
    print("\n" + "=" * 70)
    print("TECHGADGET SUPPORT ASSISTANT")
    print(f"Customer Query: {state['query']}")
    print(f"Answer: {state['answer']}")
    print(f"Confidence: {state['confidence']}")
    print("=" * 70)
    return state


# =============================================================================
# STEP 4: LANGGRAPH WORKFLOW SETUP
# =============================================================================

graph = StateGraph(State)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("hitl", hitl)
graph.add_node("output", output)


# Routing logic
def route(state: State):
    return "hitl" if state["needs_hitl"] else "output"


graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_conditional_edges("generate", route, {
    "hitl": "hitl",
    "output": "output"
})
graph.add_edge("hitl", "output")
graph.add_edge("output", END)

app = graph.compile()

print("FULL SYSTEM READY!")


# =============================================================================
# STEP 5: CHAT LOOP
# =============================================================================

print("\nTechGadget Bot Online!")
print("Ask: refund? return policy? damage? quit")

while True:
    q = input("\nUser: ").strip()

    if q.lower() in ["quit", "exit"]:
        print("Exiting...")
        break

    result = app.invoke({
        "query": q,
        "docs": [],
        "answer": "",
        "confidence": 0.0,
        "needs_hitl": False
    })
