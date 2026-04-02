from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("1. Loading documents from 'data/' directory...")
# We specify loader_cls=TextLoader to ensure it reads the .txt files correctly
loader = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

print(f"Loaded {len(documents)} document(s).")

print("\n2. Splitting documents into smaller chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

print("\n3. Initializing embedding model and ChromaDB...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", 
    model_kwargs={"device": "cpu"}
)

# This will create embeddings for all our chunks and store them locally
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="technova_policies"
)

print("\nSuccess! Vectors have been saved to the './chroma_db' directory.")