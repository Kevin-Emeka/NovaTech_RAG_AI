from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

try:
    from rag_pipeline import get_rag_chain
except Exception as e:
    logging.error(f"Failed to import rag_pipeline: {e}")
    # Will fail securely later if queried when setup is broken

app = FastAPI(title="NovaTech Assistant API")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

# Lazy-load the chain to wait until API is queried
chain = None

@app.on_event("startup")
def load_model():
    global chain
    try:
        logging.info("Initializing RAG chain...")
        chain = get_rag_chain()
        logging.info("RAG chain initialized successfully.")
    except Exception as e:
        logging.error(f"Error loading RAG chain at startup: {e}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global chain
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if not chain:
        try:
           chain = get_rag_chain()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG System is offline: {e}")
            
    try:
        # We just get the answer text from the simple StrOutputParser
        answer = chain.invoke(request.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        print(f"Error during chain invocation: {e}")
        raise HTTPException(status_code=500, detail="Error generating answer.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
