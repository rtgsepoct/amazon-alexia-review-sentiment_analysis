import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import base64  # ✅ ADDED

prediction_endpoint = "http://127.0.0.1:5000/predict"  # ✅ CHANGED (was 5500)

st.title("Text Sentiment Predictor")

uploaded_file = st.file_uploader("Choose a CSV file for bulk prediction", type="csv")
user_input = st.text_input("Enter text for prediction", "")

if st.button("Predict"):
    if uploaded_file is not None:
        files = {"file": uploaded_file}
        res = requests.post(prediction_endpoint, files=files)

        if res.status_code != 200:
            st.error(f"Error {res.status_code}: {res.text}")
        else:
            df = pd.read_csv(BytesIO(res.content))
            st.dataframe(df.head(20))

            st.download_button(
                label="Download Predictions",
                data=res.content,
                file_name="Predictions.csv",
                mime="text/csv",
            )

            # ✅ ADDED: show pie chart if provided
            if res.headers.get("X-Graph-Exists") == "true":
                graph_b64 = res.headers.get("X-Graph-Data")
                if graph_b64:
                    st.image(BytesIO(base64.b64decode(graph_b64)))

    else:
        # ✅ CHANGED: send JSON (your Flask expects JSON)
        res = requests.post(prediction_endpoint, json={"text": user_input})

        if res.status_code != 200:
            st.error(f"Error {res.status_code}: {res.text}")
        else:
            data = res.json()
            st.write(f"Predicted sentiment: {data.get('prediction')}")