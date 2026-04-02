import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def run_rag_app():
    # Load environment variables (API keys)
    load_dotenv()

    print("Initializing NovaTech Assistant...")
    print("Loading embedding model...")
    # 1. Initialize Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )

    print("Connecting to ChromaDB...")
    # 2. Connect to existing Vectorstore
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model,
        collection_name="technova_policies",
    )
    
    # 3. Create Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    print("Loading Language Model...")
    # 4. Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # 5. Define Prompt Template
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

    print("\n" + "="*60)
    print("📋 NovaTech Policy Assistant is Ready!")
    print("Type your questions about HR, IT, Finance, or Compliance.")
    print("Type 'exit' or 'quit' to close the assistant.")
    print("="*60)

    # 7. Interactive Chat Loop
    while True:
        try:
            question = input("\n🧑 You: ")
            
            if question.strip().lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
            
            if not question.strip():
                continue

            print("🤖 Assistant: Thinking...")
            # Retrieve documents for context (just to display sources)
            docs = retriever.invoke(question)
            
            # Generate Answer
            answer = chain.invoke(question)
            print(f"\n{answer}")
            
            # Print sources for debugging/verification
            print("\nSources used:")
            sources = set(doc.metadata.get("source", "Unknown").split("\\")[-1].split("/")[-1] for doc in docs)
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    run_rag_app()
