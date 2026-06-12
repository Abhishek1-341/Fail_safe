import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(layout="wide")

st.title("📂 Batch Prediction")

# select prediction stage
stage = st.radio(
    "Prediction Stage",
    ["Pre-G1 (Early)", "Pre-G2 (Mid-term)", "Pre-Final (End of Year)"]
)

# upload csv
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # endpoint selection
    if stage == "Pre-G1 (Early)":
        url = "http://127.0.0.1:8000/predict_1"

    elif stage == "Pre-G2 (Mid-term)":
        url = "http://127.0.0.1:8000/predict_2"

    else:
        url = "http://127.0.0.1:8000/predict_3"

    if st.button("Run Batch Prediction"):

        predictions = []
        shap_all = []

        progress_bar = st.progress(0)

        for i in range(len(df)):

            payload = df.iloc[i].to_dict()

            response = requests.post(
                url=url,
                json=payload
            )

            result = response.json()

            predictions.append(result["my_prediction"])

            shap_all.append({
                "row": i,
                "feature_names": result["feature_names"],
                "shap_values": result["shap_values"]
            })

            progress_bar.progress((i + 1) / len(df))

        # add predictions
        df["Predicted_G3"] = predictions

        st.subheader("Prediction Results")

        st.dataframe(df)

        # download csv
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        st.divider()

        st.subheader("Individual Prediction Analysis")

        for i in range(len(df)):

            with st.container(border=True):

                st.markdown(f"## 👤 Student {i}")

                # predicted score
                score = predictions[i]

                if score < 8:
                    status = "🔴 Low Performance"

                elif score < 14:
                    status = "🟡 Average Performance"

                else:
                    status = "🟢 High Performance"

                st.metric(
                    "Predicted G3 Marks",
                    f"{score:.2f}",
                    status
                )

                # shap analysis
                shap_info = shap_all[i]

                shap_df = pd.DataFrame({
                    "feature": shap_info["feature_names"],
                    "shap_value": shap_info["shap_values"]
                })

                shap_df = shap_df.sort_values(
                    by="shap_value",
                    key=abs,
                    ascending=False
                )

                fig = px.bar(
                    shap_df,
                    x="feature",
                    y="shap_value",
                    title=f"SHAP Analysis - Student {i}",
                    template="presentation"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )