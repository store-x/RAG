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
import hashlib
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


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "./data/vector_store"):
        """
        Initialise the vector store

        Args
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialise ChromaDB client and collection"""
        try:
            # Create a persistant ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or Create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embedding for RAG"}
            )
            print(f"Vector stor initialised. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.client.count_collections()}")
        except Exception as e:
            print(f"Error initialising vector store: {e}")
            raise

    def add_document(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store

        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents is not equal to embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        # prepare data for ChromaDB
        ids = []
        metadatas = []
        document_text = []
        embedding_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate a deterministic ID based on source file name and chunk index
            source_file = doc.metadata.get('source_file', 'unknown')
            raw_id = f"{source_file}_chunk_{i}"
            doc_id = f"doc_{hashlib.md5(raw_id.encode('utf-8')).hexdigest()[:8]}_{i}"
            ids.append(doc_id)

            # prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # document contents
            document_text.append(doc.page_content)

            # embedding
            embedding_list.append(embedding.tolist())

        # add to collection
        try:
            self.collection.add(
                ids = ids,
                embeddings=embedding_list,
                metadatas=metadatas,
                documents=document_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total ducments in collections: {self.collection.count()}")
        
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")


# initialise the vector store
vector_store = VectorStore()


## Convert the text to embeddings
texts = [doc.page_content for doc in chunks]

## generate the embeddings
embeddings = embedding_manager.genrate_embeddings(texts=texts)

## store into the vector database
vector_store.add_document(chunks, embeddings)


### Retriver Pipeline from VectorStore

class RagRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialise the retriever

        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: The search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of directories containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K = {top_k}, score threshold: {score_threshold }")

        # Generate query embeddings
        query_embeddings = self.embedding_manager.genrate_embeddings([query])[0]

        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embeddings.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1- distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i+1
                        })
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []


# initialise the rag retriever
rag_retriever = RagRetriever(vector_store, embedding_manager)
while True:
    query = input("Enter query: ")
    r = rag_retriever.retrieve(query)
    bprint(r)