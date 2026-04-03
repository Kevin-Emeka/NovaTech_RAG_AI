from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

try:
    from rag_pipeline import get_rag_chain
except Exception as e:
    logging.error(f"Failed to import rag_pipeline: {e}")
    # Will fail securely later if queried when setup is broken

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="NovaTech Assistant API")

# Disable index serving here so it doesn't block API routes
# Serve the index.html on the root URL
@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")

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
    import os
    from rag_pipeline import build_vector_store
    try:
        if not os.path.exists("./chroma_db") or not os.listdir("./chroma_db"):
            logging.info("Vector store not found. Building from docs...")
            build_vector_store()
            
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

# Important: Mount static files LAST so it doesn't block /api/ routes!
app.mount("/", StaticFiles(directory="frontend"), name="frontend")
