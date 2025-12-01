import numpy as np
import skfuzzy as fuzz

def compute_fuzzy_performance(attendance, test, assignment):

    # Define fuzzy ranges 0–100 scale
    x = np.arange(0, 101, 1)

    # Membership functions
    attendance_low = fuzz.trimf(x, [0, 0, 50])
    attendance_med = fuzz.trimf(x, [40, 60, 80])
    attendance_high = fuzz.trimf(x, [70, 100, 100])

    test_low = fuzz.trimf(x, [0, 0, 50])
    test_med = fuzz.trimf(x, [40, 60, 80])
    test_high = fuzz.trimf(x, [70, 100, 100])

    assignment_low = fuzz.trimf(x, [0, 0, 60])
    assignment_med = fuzz.trimf(x, [50, 70, 85])
    assignment_high = fuzz.trimf(x, [75, 100, 100])

    # Fuzzy membership strength
    att_low = fuzz.interp_membership(x, attendance_low, attendance)
    att_med = fuzz.interp_membership(x, attendance_med, attendance)
    att_high = fuzz.interp_membership(x, attendance_high, attendance)

    test_low_m = fuzz.interp_membership(x, test_low, test)
    test_med_m = fuzz.interp_membership(x, test_med, test)
    test_high_m = fuzz.interp_membership(x, test_high, test)

    ass_low = fuzz.interp_membership(x, assignment_low, assignment)
    ass_med = fuzz.interp_membership(x, assignment_med, assignment)
    ass_high = fuzz.interp_membership(x, assignment_high, assignment)

    # Rule-based evaluation (simple fuzzy rules)
    performance_score = (
        att_high * 0.3 + test_high_m * 0.4 + ass_high * 0.3 +
        att_med * 0.2 + test_med_m * 0.3 + ass_med * 0.2 +
        att_low * 0.1 + test_low_m * 0.1 + ass_low * 0.1
    ) * 100

    # Final fuzzy result
    if performance_score > 75:
        category = "Excellent"
    elif performance_score > 60:
        category = "Good"
    elif performance_score > 45:
        category = "Average"
    else:
        category = "Poor"

    return performance_score, category
