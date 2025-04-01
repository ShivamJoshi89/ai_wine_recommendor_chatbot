import os
import sys
from typing import List

###############################################
# Add the RAG module folder to PYTHONPATH.
###############################################
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_root = os.path.abspath(current_dir)
if rag_root not in sys.path:
    sys.path.append(rag_root)

# Since this module is in the RAGs folder, we assume the PDF file is also in the same folder.
PDF_PATH = os.path.join(current_dir, "What to Drink with What You Eat.pdf")
# The vectorstore will be saved in the current directory.
VECTORSTORE_PATH = os.path.join(current_dir, "food_wine_pairing_vectorstore")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# ---------------------------
# CONFIGURATION
# ---------------------------
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------
# INITIALIZATION
# ---------------------------
def load_and_split_book(pdf_path: str) -> List[str]:
    """
    Load the book as pages and split into overlapping chunks.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(pages)

def build_and_save_vectorstore(chunks, save_path=VECTORSTORE_PATH):
    """
    Create FAISS vectorstore from text chunks and save it.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(save_path)
    print(f"✅ Vectorstore saved to {save_path}")

def load_vectorstore(load_path=VECTORSTORE_PATH):
    """
    Load FAISS vectorstore from disk with trusted deserialization.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return FAISS.load_local(load_path, embedding_model, allow_dangerous_deserialization=True)

# ---------------------------
# RAG RETRIEVAL FUNCTION
# ---------------------------
def retrieve_food_pairing_passages(query: str, top_k=3) -> List[str]:
    """
    Retrieve top-k relevant passages for a food or wine query.
    """
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]

def build_rag_context_block(query: str, top_k=3) -> str:
    """
    Construct a context string block for prompt injection.
    """
    passages = retrieve_food_pairing_passages(query, top_k=top_k)
    if not passages:
        return ""
    context = "\n\n".join([f"• {p.strip()}" for p in passages])
    return f"Expert Advice from Sommeliers:\n{context}\n"

# ---------------------------
# ON-DEMAND INDEX BUILDING
# ---------------------------
def initialize_rag_pipeline(pdf_path: str = PDF_PATH):
    """
    Build the RAG vectorstore from scratch.
    """
    print("📚 Chunking the PDF and building vector index...")
    chunks = load_and_split_book(pdf_path)
    build_and_save_vectorstore(chunks)

if __name__ == "__main__":
    # CLI initialization or testing
    initialize_rag_pipeline()
    test_query = "What wine should I pair with mushroom risotto?"
    context = build_rag_context_block(test_query)
    print("\n🔍 RAG Retrieved Context:\n")
    print(context)