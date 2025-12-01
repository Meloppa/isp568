def generate_feedback(attendance, test, assignment):
    tips = []

    if test >= 80:
        tips.append("Great job in test performance!")

    if attendance < 70:
        tips.append("Try to attend more classes.")

    if test < 50:
        tips.append("You need to improve your test understanding.")

    if assignment < 15:
        tips.append("Assignments need more effort.")

    tips.append("Keep practicing and don't give up!")

    return tips
