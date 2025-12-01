import streamlit as st
from modules.fuzzy_logic import compute_fuzzy_performance
from modules.feedback import generate_feedback

st.title("Student Performance Evaluation (Fuzzy Logic)")

attendance = st.slider("Attendance (%)", 0, 100, 70)
test = st.slider("Test Score", 0, 100, 60)
assignment = st.slider("Assignment Score", 0, 100, 75)

if st.button("Evaluate"):
    score, category = compute_fuzzy_performance(attendance, test, assignment)
    feedback = generate_feedback(attendance, test, assignment)

    st.subheader("Results")
    st.write("Fuzzy Score:", round(score, 2))
    st.write("Category:", category)

    st.subheader("Feedback")
    for f in feedback:
        st.write("- " + f)
