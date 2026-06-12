import streamlit as st
import requests
import pandas as pd
import shap
import plotly.express as px

# make app wide
st.set_page_config(layout="wide")

st.title("🎓 Failsafe : predict G3 marks in advance")

# 1. Select Stage
stage = st.radio(
    "When are you predicting?",
    ["Pre-G1 (Early)", "Pre-G2 (Mid-term)", "Pre-Final (End of Year)"]
)

# 2. Dynamic Input Form
with st.form("input_form"):

    col1, col2 = st.columns(2)
    with col1:

        # numerical variables
        age = st.number_input("age", min_value=15, value=18, max_value=22)
        Medu = st.slider("Medu", 0,4,2)
        Fedu = st.slider("Fedu", 0,4,2)
        traveltime = st.slider("traveltime", 1,4,2)
        studytime = st.slider("studytime", 1,4,2)
        failures = st.slider("failures", 0,3,2)
        famrel = st.slider("famrel", 1, 5, 3)
        freetime = st.slider("freetime", 1, 5, 3)
        goout = st.slider("goout", 1, 5, 3)
        Dalc = st.slider("Dalc", 1, 5, 1)
        Walc = st.slider("Walc", 1, 5, 1)
        health = st.slider("health", 1, 5, 3)
        absences = st.number_input("absences", min_value=0, value=20, max_value=100)

        if stage == "Pre-G2 (Mid-term)" :
            G1 = st.number_input("G1", min_value=0, max_value=20, value=10)

        if stage == "Pre-Final (End of Year)" :
            G1 = st.number_input("G1", min_value=0, max_value=20, value=10)
            G2 = st.number_input("G2", min_value=0, max_value=20, value=10)

    with col2:

        # categorical variables
        school = st.selectbox("school", ['GP', 'MS'])
        sex = st.selectbox("sex", ['F', 'M'])
        address = st.selectbox("address", ['U', 'R'])
        famsize = st.selectbox("famsize", ['GT3', 'LE3'])
        Pstatus = st.selectbox("Pstatus", ['A', 'T'])
        Mjob = st.selectbox("Mjob",['at_home', 'health', 'other', 'services', 'teacher'])
        Fjob = st.selectbox("Fjob",['teacher', 'other', 'services', 'health', 'at_home'])
        reason = st.selectbox("reason",['course', 'other', 'home', 'reputation'])
        guardian = st.selectbox("guardian",['mother', 'father', 'other'])
        schoolsup = st.selectbox("schoolsup", ['yes', 'no'])
        famsup = st.selectbox("famsup", ['no', 'yes'])
        paid = st.selectbox("paid", ['no', 'yes'])
        activities = st.selectbox("activities", ['no', 'yes'])
        nursery = st.selectbox("nursery", ['yes', 'no'])
        higher = st.selectbox("higher", ['yes', 'no'])
        internet = st.selectbox("internet", ['no', 'yes'])
        romantic = st.selectbox("romantic", ['no', 'yes'])
        subject = st.selectbox("subject",['Maths', 'Portuguese'])

    predict_btn = st.form_submit_button("Predict G3 Result")


# 3. Handle Prediction Result
if predict_btn:

    payload = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "subject": subject
    }

    # add conditionally
    if stage == "Pre-G2 (Mid-term)":
        payload["G1"] = G1

    if stage == "Pre-Final (End of Year)":
        payload["G1"] = G1
        payload["G2"] = G2

    #st.write(payload)

    # url
    if stage == "Pre-G1 (Early)" :
        url = "http://127.0.0.1:8000/predict_1"
    elif stage == "Pre-G2 (Mid-term)":
        url = "http://127.0.0.1:8000/predict_2"
    elif stage == "Pre-Final (End of Year)" :
        url = "http://127.0.0.1:8000/predict_3"

    # hiting fast API endpoint
    response = requests.post(url = url, json=payload)
    #st.write(response.json())
    score = response.json()['my_prediction']

    # Logic for Comment
    if score < 8:  # as 25th percentile = 8
        status = "🔴 Low Performance"
    elif score < 14:  # as 75th percentile = 14
        status = "🟡 Average Performance"
    else:
        status = "🟢 High Performance"
    
    st.subheader(f'Predicted G3 Marks : {score:.2f} \n ({status})')


    # 4. Show SHAP Analysis
    st.subheader("Why this score? - SHAP Analysis")

    shap_df = pd.DataFrame({
        "feature": response.json()['feature_names'],
        "shap_value": response.json()['shap_values']
    })

    shap_df = shap_df.sort_values(
        by="shap_value",
        key=abs,
        ascending=False
    )

    fig = px.bar(shap_df,y="shap_value",x="feature",orientation = "v",
                title="Feature Contribution to Prediction", template="presentation", height=400, width=400)

    st.plotly_chart(fig)


