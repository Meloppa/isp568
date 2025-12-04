from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import io
import matplotlib.pyplot as plt
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
import time

# --- Module Imports ---
try:
    from fuzzy_system import evaluate_score, get_performance_level 
    from llm_service import initialize_llm, get_ai_suggestion, get_lecturer_chat_response
    from reports import generate_student_report_pdf
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: Could not find supporting modules. Error: {e}")
    raise e

router = APIRouter(prefix="/api/v1")

# --- Pydantic Models ---

class EvaluationInput(BaseModel):
    """Input model for student scores."""
    attendance: float = Field(..., ge=0, le=100, description="Attendance percentage (0-100).")
    test_score: float = Field(..., ge=0, le=100, description="Test score percentage (0-100).")
    assignment_score: float = Field(..., ge=0, le=100, description="Assignment score percentage (0-100).")

class Message(BaseModel):
    """Single message in the conversation history."""
    role: str = Field(..., description="The role of the message sender ('user' or 'assistant').")
    content: str = Field(..., description="The content of the message.")

class ChatQuery(BaseModel):
    """Input model for the chat lecturer endpoint, now including history."""
    student_performance_level: str = Field(..., description="The calculated performance level (e.g., 'Weak', 'Average', 'Excellent').")
    # History must be passed by the client
    history: List[Message] = Field(default=[], description="The previous messages in the conversation (user and assistant).")
    # FIX: Changed field name from 'current_question' to 'question' to match client's payload
    question: str = Field(..., description="The student's new question for the lecturer.")


# Dependency to check LLM status
def check_llm_status():
    if not initialize_llm():
        raise HTTPException(status_code=503, detail="LLM service is unavailable. Ensure Ollama is running and the model is loaded.")

# --- Endpoint Implementations ---

@router.post("/evaluate", response_model=Dict[str, Any], tags=["Evaluation"])
async def evaluate_student(input_data: EvaluationInput):
    """
    Calculates the student's overall performance using the Fuzzy Logic System.
    """
    try:
        inputs = input_data.model_dump()
        
        # 1. FUZZIFICATION: The crisp input scores (attendance, test, assignment) 
        #    are mapped to the degree of membership in the fuzzy sets (Low, Medium, High).
        
        # 2. INFERENCE & AGGREGATION: The fuzzy rules are applied using AND/OR operators
        #    to determine the output fuzzy shape (Weak, Average, Good, Excellent).
        
        # The evaluate_score function runs the fuzzy computation:
        result = evaluate_score(inputs['attendance'], inputs['test_score'], inputs['assignment_score'])

        # 3. DEFUZZIFICATION: The resulting output fuzzy shape is converted back 
        #    into a single, crisp score (result['fuzzy_score']) using the Centroid 
        #    method (default in skfuzzy).
        
        return {
            "status": "success",
            "inputs": inputs,
            "fuzzy_score": result['fuzzy_score'],
            "performance_level": result['performance_level'],
            "explanation": f"The student's performance has been evaluated as {result['performance_level']} with a fuzzy score of {result['fuzzy_score']} out of 100."
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@router.post("/suggestion", response_model=Dict[str, Any], dependencies=[Depends(check_llm_status)], tags=["LLM Advisor"])
async def get_ai_suggestion_endpoint(input_data: EvaluationInput):
    """
    Uses the local LLM (Ollama) to generate a personalized suggestion based on calculated fuzzy score.
    """
    # 1. Calculate fuzzy result
    evaluation_result = await evaluate_student(input_data)
    level = evaluation_result['performance_level']
    score = evaluation_result['fuzzy_score']
    
    # 2. Run synchronous LLM call in a threadpool to prevent blocking
    suggestion = await run_in_threadpool(
        get_ai_suggestion, 
        evaluation_result['inputs'], 
        level, 
        score
    )
    
    if suggestion.startswith("LLM service is not initialized") or suggestion.startswith("An error occurred"):
        raise HTTPException(status_code=500, detail=suggestion)

    return {
        "status": "success",
        "performance_level": level,
        "suggestion": suggestion
    }

@router.post("/chat", response_model=Dict[str, Any], dependencies=[Depends(check_llm_status)], tags=["LLM Lecturer"])
async def chat_with_lecturer_endpoint(chat_data: ChatQuery):
    """
    Acts as a lecturer (using Ollama) who answers questions related to student evaluation and suggestions, 
    maintaining conversation history.
    """
    # Prepare history for the service function
    history_messages = [msg.model_dump() for msg in chat_data.history]
    
    # FIX: Use chat_data.question (the new field name from the Pydantic model)
    current_question = chat_data.question 

    # Run synchronous LLM chat call in a threadpool
    answer = await run_in_threadpool(
        get_lecturer_chat_response,
        chat_data.student_performance_level,
        current_question, # Pass the question 
        history_messages # Pass the history list
    )

    if answer.startswith("LLM service is not initialized") or answer.startswith("An error occurred"):
        raise HTTPException(status_code=500, detail=answer)
    
    return {
        "status": "success",
        "answer": answer
    }


@router.get("/report/download", tags=["Reports"])
async def download_pdf_report(
    attendance: float, 
    test_score: float, 
    assignment_score: float
):
    """
    Generates a PDF report for a student's evaluation, including the AI suggestion, and returns it for download.
    """
    # Create input model for validation and internal use
    input_data = EvaluationInput(
        attendance=attendance, 
        test_score=test_score, 
        assignment_score=assignment_score
    )
    
    # 1. Get Evaluation Data
    evaluation = await evaluate_student(input_data)
    
    # 2. Get Suggestion Data (runs LLM call in a threadpool)
    suggestion_response = await get_ai_suggestion_endpoint(input_data)
    suggestion_text = suggestion_response['suggestion']
    
    # 3. Generate PDF (runs synchronous ReportLab logic in a threadpool)
    pdf_buffer = await run_in_threadpool(
        generate_student_report_pdf,
        input_data.model_dump(),
        evaluation,
        suggestion_text
    )

    # 4. Return the PDF file response
    report_filename = f"student_report_{evaluation['performance_level']}_{int(time.time())}.pdf"

    return Response(
        content=pdf_buffer.read(),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={report_filename}"}
    )