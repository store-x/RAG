## Just Deprecation warning supression to keep my terminal clean
import warnings

# Langchain warnings
warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*",
    category=DeprecationWarning,
)   
# Hugging Face unauthenticated warnings
warnings.filterwarnings(
    "ignore",
    message=".*unauthenticated requests.*",
)


## Beautifully printing things
import pprint

def bprint(data):
    """Prints any Python object in a beautifully formatted way."""
    pp = pprint.PrettyPrinter(
        indent=4,         
        width=80,      
        depth=None,      
        sort_dicts=False, 
        compact=False    
    )
    pp.pprint(data)
    

## RAG PIPELINES- Data Ingestion to Vector DB Pipeline

import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

## Read all the PDFs inside the directory

def process_all_pdf(pdf_directory):
    """Process all the pdf in a directory"""
    all_documents = []
    path_dir = Path(pdf_directory)

    # Find all pdf files
    pdf_files = list(path_dir.glob("**/*.pdf"))

    print(f"Loaded {len(pdf_files)} PDF Files from directory.")

    for pdf in pdf_files:
        print(f"\nProcessing {pdf.name}")
        try:
            loader = PyPDFLoader(str(pdf))
            documents = loader.load()

            # Add some more informations to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf.name
                doc.metadata['file_type'] = 'pdf'

            all_documents.extend(documents)
            print(f"  Loaded {len(documents)} pages")
        except Exception as e:
            print(f" Failed to load. Error: {e}")

    print(f"\nLoaded documents: {len(all_documents)}")
    return all_documents

all_pdf_document = process_all_pdf('data/pdf')

# bprint(all_pdf_document)

## Text Splitting into chunks

def split_document(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for better RAG Performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents files into {len(split_docs)} chunks")

    # Show example chunk
    if split_docs:
        print(f"\nExample chunk: \nContent: {split_docs[0].page_content[:200]}...\nMetadata: {split_docs[0].metadata}")

    return split_docs


chunks = split_document(all_pdf_document)
# print(len(chunks))


## Embedding and VectorStoreDB

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""

    def __init__(self, model_name: str = "all-MiniLM-L6-V2"):
        """
        Initialize the embedding manager

        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfuly. Embedding dimensions: {self.model.get_embedding_dimension()}") # get_sentence_embedding_dimension has been deprecated
        except Exception as e:
            print(f"Error in loading model {self.model_name}: {e}")
            raise

    def genrate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Genrate embeddings for a list of text

        Args:
            texts: List of text strings to embed

        Return:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")

        print(f"Generating embeddings for {len(texts)}...")    
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

# initialise the embedding manager
embedding_manager = EmbeddingManager()