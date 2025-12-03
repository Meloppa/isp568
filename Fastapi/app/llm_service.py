from openai import OpenAI
from typing import Optional, Dict

# --- IMPORTANT: Ollama Configuration ---
# Ollama provides an OpenAI-compatible API endpoint.
# FIX: Added the necessary '/v1' suffix to the base URL.
OLLAMA_BASE_URL = "http://localhost:11434/v1" 
OLLAMA_MODEL_NAME = "qwen3:4b"                 # Specify the model you have running in Ollama

LLM_CLIENT: Optional[OpenAI] = None

def initialize_llm():
    """Initializes the OpenAI client pointing to the local Ollama server."""
    global LLM_CLIENT
    if LLM_CLIENT is None:
        try:
            print(f"Attempting to connect to Ollama server at: {OLLAMA_BASE_URL}")
            # Initialize the OpenAI client
            LLM_CLIENT = OpenAI(
                base_url=OLLAMA_BASE_URL,
                # FIX: Reverted the API key to a standard placeholder for Ollama.
                api_key="ollama" 
            )
            # A simple call to verify connection and model availability (non-blocking)
            print(f"LLM client initialized. Target Model: {OLLAMA_MODEL_NAME}")
        except Exception as e:
            print(f"ERROR: Failed to initialize OpenAI client for Ollama. Error: {e}")
            LLM_CLIENT = None
            
    return LLM_CLIENT

def generate_llm_response(system_prompt: str, user_prompt: str) -> str:
    """
    Generates a response from the LLM using the chat completion interface.
    NOTE: This function is synchronous and should be run in a thread pool (e.g., via run_in_threadpool).
    """
    client = initialize_llm()
    if client is None:
        return "LLM service is not initialized. Please ensure Ollama is running and the model is loaded."

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            # INCREASED MAX_TOKENS: Give the LLM more room to generate a complete suggestion.
            max_tokens=1024,
        )
        
        generated_content = response.choices[0].message.content.strip()
        
        # --- FIX: Check for empty response ---
        if not generated_content:
            return "The language model failed to generate a suggestion. This can happen if the model's output is blank or too short."
        
        return generated_content
        
    except Exception as e:
        print(f"LLM generation error: {e}")
        # The detailed error message will now show up in your Uvicorn logs when a 500 happens.
        return f"An error occurred while communicating with the local Ollama server. Check if the model '{OLLAMA_MODEL_NAME}' is running: {e}"

def get_ai_suggestion(inputs: Dict, level: str, score: float) -> str:
    """Generates an academic suggestion based on the student's performance."""
    context = (
        f"The student received the following evaluation scores: "
        f"Attendance: {inputs['attendance']}%, Test Score: {inputs['test_score']}%, "
        f"Assignment Score: {inputs['assignment_score']}%. "
        f"The calculated overall performance level is **{level}** (Score: {score}/100)."
    )
    
    system_prompt = (
        "You are a supportive and professional Academic Advisor focused on student success. "
        "Your task is to analyze the provided student evaluation scores and performance level, and offer **a single, plain-text paragraph of** "
        "concise, actionable, and encouraging suggestions for improvement or continued excellence. "
        "Focus on how the student can adjust their attendance, test preparation, or assignment effort. **Do not use markdown formatting (like bolding, lists, or headers) in your final response.**"
    )
    
    user_prompt = f"{context}\n\nBased on these details, please provide a constructive suggestion and an encouraging closing statement."
    
    return generate_llm_response(system_prompt, user_prompt)

def get_lecturer_chat_response(performance_level: str, question: str) -> str:
    """Generates a chat response from the lecturer persona."""
    # MODIFICATION: Added instructions for brevity and plain text output.
    system_prompt = (
        f"You are Professor Aniq, a knowledgeable and strict lecturer. "
        f"Your sole focus is on providing advice and answering questions related to **student evaluation, performance, and academic improvement**. "
        f"The student's current performance level is **{performance_level}**. "
        f"Answer the user's question directly and concisely, using a maximum of 3-4 short sentences. "
        f"**Do not use any markdown formatting (like bolding, lists, or headers) in your final response.** "
        f"If the user asks a question unrelated to academics, evaluation, or study habits, you must firmly but politely refuse, stating: "
        f"'I am Professor Academicus, and I can only assist with questions regarding your academic evaluation and performance improvement.'"
    )
    
    return generate_llm_response(system_prompt, question)