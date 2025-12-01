def calculate_total(attendance, test, assignment):
    return (attendance * 0.2) + (test * 0.5) + (assignment * 0.3)

def get_category(score):
    if score >= 80:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Poor"
