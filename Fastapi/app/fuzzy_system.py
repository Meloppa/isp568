import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- 1. Define Antecedent (Input) and Consequent (Output) Variables ---
# Range for all inputs and output is 0 to 100
input_range = np.arange(0, 101, 1)

# Antecedents (Inputs)
attendance = ctrl.Antecedent(input_range, 'attendance')
test_score = ctrl.Antecedent(input_range, 'test_score')
assignment_score = ctrl.Antecedent(input_range, 'assignment_score')
# Consequent (Output)
performance = ctrl.Consequent(input_range, 'performance')

# --- 2. Define Membership Functions (MFs) ---
# Inputs MFs: Low, Medium, High
attendance['low'] = fuzz.trimf(input_range, [0, 0, 50])
attendance['medium'] = fuzz.trimf(input_range, [25, 50, 75])
attendance['high'] = fuzz.trimf(input_range, [50, 100, 100])

test_score['low'] = fuzz.trimf(input_range, [0, 0, 50])
test_score['medium'] = fuzz.trimf(input_range, [30, 60, 80])
test_score['high'] = fuzz.trimf(input_range, [70, 100, 100])

assignment_score['low'] = fuzz.trimf(input_range, [0, 0, 50])
assignment_score['medium'] = fuzz.trimf(input_range, [30, 60, 80])
assignment_score['high'] = fuzz.trimf(input_range, [70, 100, 100])

# Output MFs: Weak, Average, Good, Excellent
performance['weak'] = fuzz.trimf(input_range, [0, 0, 30])
performance['average'] = fuzz.trimf(input_range, [20, 45, 70])
performance['good'] = fuzz.trimf(input_range, [50, 75, 90])
performance['excellent'] = fuzz.trimf(input_range, [80, 100, 100])

# --- 3. Define Fuzzy Rules ---
rule1 = ctrl.Rule(attendance['low'] | test_score['low'] | assignment_score['low'], performance['weak'])
rule2 = ctrl.Rule(attendance['medium'] & test_score['medium'], performance['average'])
rule3 = ctrl.Rule(test_score['high'] & assignment_score['high'], performance['excellent'])
rule4 = ctrl.Rule(attendance['high'] & (test_score['medium'] | assignment_score['medium']), performance['good'])
rule5 = ctrl.Rule(attendance['medium'] & test_score['low'] & assignment_score['low'], performance['weak'])
rule6 = ctrl.Rule(attendance['high'] & test_score['high'] & assignment_score['high'], performance['excellent'])
rule7 = ctrl.Rule(attendance['low'] & test_score['medium'] & assignment_score['medium'], performance['average'])
rule8 = ctrl.Rule(test_score['high'] & assignment_score['medium'], performance['good'])

# --- 4. Create Control System and Simulation ---
PERFORMANCE_CTRL = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
PERFORMANCE_SIMULATION = ctrl.ControlSystemSimulation(PERFORMANCE_CTRL)

def get_performance_level(score: float) -> str:
    """Maps the defuzzified score to a descriptive level."""
    if score < 30:
        return "Weak"
    elif 30 <= score < 60:
        return "Average"
    elif 60 <= score < 85:
        return "Good"
    else:
        return "Excellent"

def evaluate_score(attendance_score: float, test_score_val: float, assignment_score_val: float) -> dict:
    """Runs the fuzzy logic evaluation."""
    try:
        # Fuzzification happens implicitly here when setting inputs
        PERFORMANCE_SIMULATION.input['attendance'] = attendance_score
        PERFORMANCE_SIMULATION.input['test_score'] = test_score_val
        PERFORMANCE_SIMULATION.input['assignment_score'] = assignment_score_val

        # Inference and Aggregation occur during compute()
        PERFORMANCE_SIMULATION.compute()
        
        # Defuzzification extracts the crisp score from the output fuzzy set
        final_score = PERFORMANCE_SIMULATION.output['performance']
        level = get_performance_level(final_score)

        return {
            "fuzzy_score": round(final_score, 2),
            "performance_level": level,
        }
    except ValueError as e:
        raise ValueError(f"Fuzzy computation failed: {e}. Check if input scores are valid and rules cover the input space.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during fuzzy evaluation: {e}")