import sys
import traceback

print("Starting test...")
try:
    print("Importing get_rag_chain...")
    from rag_pipeline import get_rag_chain
    print("Calling get_rag_chain()...")
    chain = get_rag_chain()
    print("Chain loaded! Invoking...")
    answer = chain.invoke("What is our leave policy?")
    print("Invocation finished!")
    with open("test_out.txt", "w") as f:
        f.write(answer)
except Exception as e:
    print(f"Exception caught: {e}")
    with open("test_err.txt", "w") as f:
        f.write(traceback.format_exc())
