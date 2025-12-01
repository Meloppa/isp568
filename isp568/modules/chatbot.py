def chatbot_response(message):
    message = message.lower()

    if "tips" in message:
        return "Study consistently and revise weekly!"
    if "attendance" in message:
        return "Attendance helps you understand lessons better."
    if "improve" in message:
        return "Focus on weak areas and ask teachers for help."

    return "I'm here to help with your performance questions!"
