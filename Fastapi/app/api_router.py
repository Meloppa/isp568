from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import io
import matplotlib.pyplot as plt
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
import time

# Import logic from other modules using simple file names
from fuzzy_system import evaluate_score, performance_variable
from llm_service import initialize_llm, get_ai_suggestion, get_lecturer_chat_response
from reports import generate_student_report_pdf

router = APIRouter(prefix="/api/v1")

# --- Pydantic Models ---

class EvaluationInput(BaseModel):
    """Input model for student scores."""
    attendance: float = Field(..., ge=0, le=100, description="Attendance percentage (0-100).")
    test_score: float = Field(..., ge=0, le=100, description="Test score percentage (0-100).")
    assignment_score: float = Field(..., ge=0, le=100, description="Assignment score percentage (0-100).")

class ChatQuery(BaseModel):
    """Input model for the chat lecturer endpoint."""
    student_performance_level: str = Field(..., description="The calculated performance level (e.g., 'Weak', 'Average', 'Excellent').")
    question: str = Field(..., description="The student's question for the lecturer.")

# Dependency to check LLM status
def check_llm_status():
    if not initialize_llm():
        # This will trigger if the Ollama server is not running or the model is not loaded.
        raise HTTPException(status_code=503, detail="LLM service is unavailable. Ensure Ollama is running and the model is loaded.")

# --- Endpoint Implementations ---

@router.post("/evaluate", response_model=Dict[str, Any], tags=["Evaluation"])
async def evaluate_student(input_data: EvaluationInput):
    """
    Calculates the student's overall performance using the Fuzzy Logic System.
    """
    try:
        inputs = input_data.model_dump()
        result = evaluate_score(inputs['attendance'], inputs['test_score'], inputs['assignment_score'])

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

@router.get("/graph", tags=["Visualization"])
async def get_fuzzy_graph():
    """
    Generates a PNG image of the fuzzy membership functions for the 'Performance' output.
    """
    try:
        # Plotting the Performance output variable MFs
        fig, ax = plt.subplots(figsize=(8, 4))
        performance_variable.view(ax=ax)
        ax.set_title("Performance Evaluation Membership Functions")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Save the figure to an in-memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        buf.seek(0)
        
        # Return the image as a streaming response
        return Response(content=buf.read(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating graph: {e}")

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
    Acts as a lecturer (using Ollama) who only answers questions related to student evaluation and suggestions.
    """
    # Run synchronous LLM chat call in a threadpool
    answer = await run_in_threadpool(
        get_lecturer_chat_response,
        chat_data.student_performance_level,
        chat_data.question
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