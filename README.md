# 🧠 RAGFlow — Internal Knowledge Management System

A VC firm accumulates thousands of documents over time : investment memos,
market research, due diligence reports, portfolio company updates, legal docs.
Finding a specific insight buried in a 3-year-old report takes hours — if it
gets found at all. This project builds an internal knowledge base powered by
RAG (Retrieval-Augmented Generation) that lets any team member ask questions
and get instant, sourced answers from the full document library.

---

## What RAG solves

```
  Without RAG :                       With RAG :
  ─────────────────────               ──────────────────────────────────
  Q: "What was our thesis             Q: (same question)
  on African logistics in 2023?"      │
                                      ▼
  → Search through 200 PDFs           Vector search finds top 5 relevant
  → Hope someone remembers            document chunks
  → 30-60 minutes                     │
                                      ▼
                                      Gemini reads chunks + generates answer
                                      with exact source citations
                                      │
                                      ▼
                                      Answer in 3 seconds with page references
```

---

## RAG Architecture

```
  INDEXING (done once, then updated incrementally)
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  Documents (PDF, Word, web)                          │
  │       │                                              │
  │       ▼                                              │
  │  Document Loader → Text Splitter                     │
  │  (chunks of 500 tokens, 50 overlap)                  │
  │       │                                              │
  │       ▼                                              │
  │  Embedding Model (text-embedding-004)                │
  │  → converts each chunk to a vector [0.23, -0.81...]  │
  │       │                                              │
  │       ▼                                              │
  │  FAISS Vector Store (local) or ChromaDB              │
  │  → stores vectors + original text + metadata        │
  └──────────────────────────────────────────────────────┘

  RETRIEVAL (at query time)
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  User question: "What markets did we enter in 2023?" │
  │       │                                              │
  │       ▼                                              │
  │  Embed the question → query vector                   │
  │       │                                              │
  │       ▼                                              │
  │  FAISS similarity search → top 5 relevant chunks     │
  │       │                                              │
  │       ▼                                              │
  │  Gemini: "Based on these sources, answer the         │
  │  question. Cite your sources."                       │
  │       │                                              │
  │       ▼                                              │
  │  Answer + source references                          │
  └──────────────────────────────────────────────────────┘
```

---

## Core implementation

```python
import os
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Initialize models
llm = ChatGoogleGenerativeAI(
    model       = "gemini-2.0-flash",
    temperature = 0.1
)
embeddings = GoogleGenerativeAIEmbeddings(
    model = "models/text-embedding-004"
)

# ─────────────────────────────────────────
# 1. Index documents
# ─────────────────────────────────────────
class KnowledgeBase:

    def __init__(self, index_path: str = "./knowledge_index"):
        self.index_path = index_path
        self.splitter   = RecursiveCharacterTextSplitter(
            chunk_size    = 500,
            chunk_overlap = 50,
        )
        self.vectorstore = None

    def index_documents(self, docs_folder: str):
        """
        Load all PDFs from a folder and index them.
        Run this once to build the knowledge base.
        """
        print(f"Indexing documents from {docs_folder}...")

        # Load all PDFs
        loader = DirectoryLoader(
            docs_folder,
            glob      = "**/*.pdf",
            loader_cls = PyPDFLoader
        )
        documents = loader.load()
        print(f"  Loaded {len(documents)} pages from {docs_folder}")

        # Add source metadata to each chunk
        for doc in documents:
            doc.metadata["indexed_at"] = str(Path(doc.metadata.get("source", "")).name)

        # Split into chunks
        chunks = self.splitter.split_documents(documents)
        print(f"  Split into {len(chunks)} chunks")

        # Create vector store
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        self.vectorstore.save_local(self.index_path)
        print(f"  ✅ Index saved to {self.index_path}")

    def load_index(self):
        """Load an existing index from disk."""
        self.vectorstore = FAISS.load_local(
            self.index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    def add_document(self, pdf_path: str):
        """Add a single new document to the existing index (incremental update)."""
        loader = PyPDFLoader(pdf_path)
        docs   = loader.load()
        chunks = self.splitter.split_documents(docs)
        self.vectorstore.add_documents(chunks)
        self.vectorstore.save_local(self.index_path)
        print(f"✅ Added {len(chunks)} chunks from {pdf_path}")

# ─────────────────────────────────────────
# 2. Query the knowledge base
# ─────────────────────────────────────────

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a knowledgeable assistant for an investment firm.
Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I don't have this information in the knowledge base."
Always cite which document your answer comes from.

Context:
{context}

Question: {question}

Answer (with source citations):"""
)

class KnowledgeBaseQA:

    def __init__(self, kb: KnowledgeBase):
        self.qa_chain = RetrievalQA.from_chain_type(
            llm           = llm,
            chain_type    = "stuff",
            retriever     = kb.vectorstore.as_retriever(
                search_kwargs={"k": 5}  # retrieve top 5 most relevant chunks
            ),
            chain_type_kwargs = {"prompt": QA_PROMPT},
            return_source_documents = True
        )

    def ask(self, question: str) -> dict:
        result  = self.qa_chain.invoke({"query": question})
        sources = list(set(
            doc.metadata.get("indexed_at", "unknown")
            for doc in result["source_documents"]
        ))
        return {
            "answer":  result["result"],
            "sources": sources,
            "num_sources_consulted": len(result["source_documents"])
        }


# Usage
kb = KnowledgeBase()

# First time : index all documents
kb.index_documents("./investment_docs/")

# Load existing index
kb.load_index()

# Ask questions
qa = KnowledgeBaseQA(kb)

result = qa.ask("What was our investment thesis for East Africa in 2023?")
print(result["answer"])
print(f"\nSources: {', '.join(result['sources'])}")
```

---

## FastAPI interface — REST endpoint for the team

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Renew Capital Knowledge Base API")

kb = KnowledgeBase()
kb.load_index()
qa = KnowledgeBaseQA(kb)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer:  str
    sources: list[str]

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = qa.ask(request.question)
    return AnswerResponse(answer=result["answer"], sources=result["sources"])

@app.post("/add-document")
def add_document(pdf_path: str):
    kb.add_document(pdf_path)
    return {"message": f"Document indexed successfully"}

# Run: uvicorn main:app --reload
# Query: POST /ask {"question": "What markets are we targeting in 2025?"}
```

---

## What I learned

The **chunk size** is the most impactful parameter in RAG. Too large (2000 tokens)
and the retriever returns chunks that mix multiple topics — the LLM gets confused.
Too small (100 tokens) and each chunk lacks context — the LLM can't give a complete
answer. 500 tokens with 50-token overlap works well for investment documents,
which tend to have dense, self-contained paragraphs.

The other key insight : RAG is only as good as the **question quality**. Vague
questions produce vague answers. The best use case is specific, factual queries :
"What was the IRR target for the 2022 Kenya fund?" gets a precise answer.
"Tell me about Africa" does not.

---

*Project built as part of Engineering degree — ENSET Mohammedia*
*By **Abderrahmane Elouafi** · [LinkedIn](https://www.linkedin.com/in/abderrahmane-elouafi-43226736b/) · [Portfolio](https://my-first-porfolio-six.vercel.app/)*
