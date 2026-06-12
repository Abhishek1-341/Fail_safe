import streamlit as st

st.set_page_config(
    page_title="Failsafe",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 FAILSAFE")

st.markdown("""
## Predict Student Final Exam Performance (G3)

This app provides:

### 📌 Single Prediction
Predict G3 marks for one student using manual inputs.

### 📂 Batch Prediction
Upload a CSV file and predict G3 marks for multiple students at once.

### 🔍 SHAP Explainability
Understand why the model made a prediction using SHAP values.
            
### using this you can predict G3(final exam) at differnrt time :-
            1. Pre-G1 (Early) - no need to provide G1 and G2 marks, But error is high
            2. Pre-G2 (Mid-term) - no need to provide G1 marks but G2 mark is required, reduced error compared to first one
            3. Pre-Final (End of Year) - G1 and G2 marks are required, least error in prediction
so this is an trade off between time school/college have to take action and quality of prediction

---

### 👈 Use the sidebar to navigate between pages.
""")