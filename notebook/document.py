## Just Deprecation warning supression to keep my terminal clean
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*",
    category=DeprecationWarning,
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

##DOCUMENT INITIALISATION
from langchain_core.documents import Document


doc = Document(
    page_content="This is the main text content I am using to create RAG",
    metadata={
        "source":"example.txt",
        "pages": 1,
        "author": "Sanskar Chauhan",
        "date_created": "2026-10-12"
    }
)
# bprint(doc)

## Create a simple text file
import os
os.makedirs("./data/text_files", exist_ok=True)

sample_texts={
    "./data/text_files/python_intro.txt":"""Python Programming Introduction

Python is a high-level, interpreted programming language known for its simplicity and readability.
Created by Guido van Rossum and first released in 1991, Python has become one of the most popular
programming languages in the world.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support

Python is widely used in web development, data science, artificial intelligence, and automation.""",
    
    "./data/text_files/machine_learning.txt": """Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn and improve
from experience without being explicitly programmed. It focuses on developing computer programs
that can access data and use it to learn for themselves.

Types of Machine Learning:
1. Supervised Learning: Learning with labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through rewards and penalties

Applications include image recognition, speech processing, and recommendation systems
    
    
    """

}

for filepath,content in sample_texts.items():
    if not os.path.exists(filepath):
        with open(filepath,'w',encoding="utf-8") as f:
            f.write(content)
bprint("✅ Sample text files created!")

## TextLoader
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/text_files/python_intro.txt", encoding="utf-8")

document = loader.load()
# bprint(document)

### Directory Loader
from langchain_community.document_loaders import DirectoryLoader

dir_loader = DirectoryLoader(
    path="./data/text_files",
    glob="**/*.txt", # Pattern to match txt files 
    loader_cls=TextLoader, # Loader class to use, use Pdfloader or so and list of loaders as well 
    loader_kwargs={'encoding': 'UTF-8'},
    show_progress=False
)
documents = dir_loader.load()
# bprint(documents)

### PDF Loader
from langchain_community.document_loaders import PyMuPDFLoader

pdf_loader = DirectoryLoader(
    path="./data/pdf",
    glob="**/*.pdf",
    loader_cls = PyMuPDFLoader,
    show_progress=True
)
pdfs = pdf_loader.load()
bprint(pdfs)