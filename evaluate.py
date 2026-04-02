import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Attempt new wrapper import, fallback safely
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError:
    LangchainLLMWrapper, LangchainEmbeddingsWrapper = None, None

def run_evaluation():
    print("Loading RAG pipeline...")
    from rag_pipeline import get_rag_chain # We need access to just the retriever to grab context
    
    chain = get_rag_chain()
    
    # We also recreate the retriever to easily track context explicitly
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    from langchain_chroma import Chroma
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="technova_policies")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    questions = [
        "What are our core working hours?",
        "How long is maternity leave?",
        "What happens if I lose my laptop?",
        "What is NovaTech's flagship product?"
    ]

    ground_truths = [
        "Our core working hours are 9 AM to 5 PM.",
        "Maternity leave is 6 months fully paid.",
        "You must report it to the IT helpdesk within 2 hours.",
        "NovaTech's flagship product is the Enterprise Assistant Platform."
    ]

    answers = []
    contexts = []

    print("Generating answers for evaluation...")
    for q in questions:
        print(f"  Q: {q}")
        # Get raw LLM answer
        ans = chain.invoke(q)
        answers.append(ans)
        
        # Get raw retrieved docs for context metrics
        docs = retriever.invoke(q)
        ctx = [d.page_content for d in docs]
        contexts.append(ctx)
        print(f"  A: {ans}")

    # Build HuggingFace Dataset required by Ragas
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)

    # Initialize Evaluator Models (Using Groq instead of OpenAI)
    print("\nInitializing RAGAS Evaluator LLM (Groq Llama-3.1)...")
    eval_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    eval_embeddings = embeddings
    
    if LangchainLLMWrapper:
        eval_llm = LangchainLLMWrapper(eval_llm)
        eval_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]

    print("\nRunning RAGAS Evaluation... (This may take a minute)")
    try:
        results = evaluate(
            dataset = dataset,
            metrics = metrics,
            llm = eval_llm,
            embeddings = eval_embeddings
        )
        print("\n" + "="*50)
        print("RAGAS EVALUATION SCORECARD:")
        print("="*50)
        print(results)
    except Exception as e:
        print(f"RAGAS evaluation failed (Groq models sometimes struggle with the strict formatting required): {e}")

if __name__ == "__main__":
    run_evaluation()
