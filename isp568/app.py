from modules.fuzzy_logic import compute_fuzzy_performance
from modules.feedback import generate_feedback
from modules.report import generate_pdf
from modules.chatbot import chatbot_response

# Get inputs
attendance = float(input("Enter your attendance percentage (0-100): "))
test = float(input("Enter your test score (0-100): "))
assignment = float(input("Enter your assignment score (0-100): "))

# Fuzzy logic evaluation
score, category = compute_fuzzy_performance(attendance, test, assignment)

feedback = generate_feedback(attendance, test, assignment)

print("\n=== Student Performance (Fuzzy Logic) ===")
print("Fuzzy Score:", round(score, 2))
print("Category:", category)
print("Feedback:", feedback)

# Generate PDF
generate_pdf("Student", score, category, feedback)
print("\nPDF report created successfully!")

# Chatbot mode
print("\nChatbot Mode (type 'exit' to stop)")
while True:
    msg = input("You: ")
    if msg == "exit":
        break
    print("Bot:", chatbot_response(msg))
