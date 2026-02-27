import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# Flask is on 5000 (NOT 5500)
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Text Sentiment Predictor")

uploaded_file = st.file_uploader(
    "Choose a CSV file for bulk prediction",
    type="csv",
)

user_input = st.text_input("Enter text for prediction", "")

if st.button("Predict"):
    if uploaded_file is not None:
        files = {"file": uploaded_file}
        response = requests.post(prediction_endpoint, files=files)

        if response.status_code != 200:
            st.error(f"Error {response.status_code}: {response.text}")
        else:
            response_bytes = BytesIO(response.content)
            df = pd.read_csv(response_bytes)
            st.dataframe(df.head(20))

            st.download_button(
                label="Download Predictions",
                data=response.content,
                file_name="Predictions.csv",
                mime="text/csv",
            )

            # graph headers (optional)
            if response.headers.get("X-Graph-Exists") == "true":
                graph_b64 = response.headers.get("X-Graph-Data")
                if graph_b64:
                    st.image(BytesIO(__import__("base64").b64decode(graph_b64)))

    else:
        payload = {"text": user_input}
        response = requests.post(prediction_endpoint, json=payload)

        if response.status_code != 200:
            st.error(f"Error {response.status_code}: {response.text}")
        else:
            data = response.json()
            st.write(f"Predicted sentiment: {data.get('prediction')}")