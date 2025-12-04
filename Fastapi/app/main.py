from fastapi import FastAPI
from api_router import router
from llm_service import initialize_llm

app = FastAPI(
    title="Fuzzy Student Evaluation System",
    description="An API that evaluates student performance using fuzzy logic and provides AI-generated academic suggestions.",
    version="1.1.1"
)

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Initializes the LLM on application startup."""
    print("--- Starting up application and initializing LLM client... ---")
    initialize_llm()
    print("--- Application ready. ---")


# uvicorn main:app --reload