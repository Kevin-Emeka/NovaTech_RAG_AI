import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

# Constants
DATA_DIR = "./data"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "technova_policies"

def build_vector_store():
    """
    Reads all text files from the data directory, splits them into chunks, 
    embeds them, and stores the vectors in ChromaDB.
    """
    print(f"Loading documents from {DATA_DIR}...")
    loader_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    docs = []
    for path in loader_paths:
        try:
            loader = TextLoader(path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {path}: {e}")
    if not docs:
        print(f"No '.txt' files found in {DATA_DIR}.")
        return None

    print(f"Loaded {len(docs)} document(s). Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", 
        model_kwargs={"device": "cpu"}
    )

    print(f"Storing chunks in ChromaDB at '{PERSIST_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )
    print("Vector store built successfully!")
    return vectorstore

def get_rag_chain():
    """
    Initializes the ChromaDB connection and LLM, then constructs
    and returns a runnable Langchain RAG pipeline.
    """
    # 1. Initialize Embeddings
    print("  [DEBUG] Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", 
        model_kwargs={"device": "cpu"}
    )
    
    # 2. Connect to existing Vectorstore
    print("  [DEBUG] Connecting to ChromaDB...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    # 3. Create Retriever
    print("  [DEBUG] Creating retriever...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 4. Initialize LLM (Using Groq instead of OpenAI)
    print("  [DEBUG] Intializing ChatGroq...")
    from langchain_groq import ChatGroq
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
    
    # 5. Define Prompt Template
    print("  [DEBUG] Defining prompt & chain...")
    rag_template = """You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have that information in our policy documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    prompt = ChatPromptTemplate.from_template(rag_template)
    
    # 6. Build the Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def query_rag(question: str):
    """Convenience function to ask a question to the pipeline."""
    chain = get_rag_chain()
    return chain.invoke(question)

def create_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)
    qa_chain = get_rag_chain()
    return qa_chain, vectorstore

from pydantic import BaseModel, Field
from langchain_core.tools import tool

class DocumentSearchInput(BaseModel):
    query: str = Field(description="The search query to look up in the documents.")

def get_retriever_tool(vectorstore):
    retriever = vectorstore.as_retriever()
    
    @tool("document_search", args_schema=DocumentSearchInput)
    def document_search(query: str) -> str:
        """Search NovaTech company documents for policies, IT security, and HR rules."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
        
    return document_search

from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

def create_agent(vectorstore):
    tool = get_retriever_tool(vectorstore)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    
    system_prompt = (
        "You are a helpful NovaTech assistant. Answer user queries by searching the company documents. "
        "ALWAYS read the content of the retrieved documents and synthesize a detailed answer. "
        "Never just say 'I searched the documents', actually provide the information you found."
    )
    agent = create_react_agent(llm, tools=[tool], prompt=system_prompt)
    return agent

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        # Run: python rag_pipeline.py --build
        build_vector_store()
    else:
        # Run: python rag_pipeline.py
        print("Testing RAG Pipeline Setup...")
        print("To build/update the vector store from data/, run:")
        print("    python rag_pipeline.py --build\n")
        
        test_question = "What is the company's leave policy?"
        print(f"Testing a sample query: '{test_question}'")
        try:
            answer = query_rag(test_question)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error executing query: {e}")
            print("Make sure you build the vector store first and have your OPENAI_API_KEY set.")