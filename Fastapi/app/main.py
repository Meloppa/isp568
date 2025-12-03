from fastapi import FastAPI
from api_router import router
from llm_service import initialize_llm

app = FastAPI(
    title="Modular Fuzzy Student Evaluation System",
    description="A modular API for student evaluation using Fuzzy Logic and a locally hosted Ollama LLM.",
    version="1.0.0"
)

# Include the main application router
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Initializes the LLM on application startup."""
    # Ensure Ollama is running before this is called!
    print("--- Starting up application and initializing LLM client... ---")
    initialize_llm()
    print("--- Application ready. ---")


# To run this API, ensure your required libraries are installed and run:
# uvicorn main:app --reload