from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO

# nltk.cropus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as ply
import pandas as pd
import pickle 
import base64

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

@app.route("./test", method=["GET"])
def test():
    return "Teset request received successfully. Service is running"


@app.route("/", methods = ["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    # select the predictor.to be loaded from Models: 
    predictor  = pickle.load(open(r"./model_dt.pkl", "rb"))
    scaler = pickle.load(open(r"./scaler.pkl", 'rb'))
    cv = pickle.load(open(r"./countVectorizer.pkl", "rb"))
    try:
        #check if the request contains a file (for bulk predictions) or text input
        if "file" in request.files:
            #Bulk prediction from CSV file
            file = request.files['file']
            data = pd.read_csv(file)

            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions, 
                mimetype="text/csv"
                as_attachment=True
                download_name = "Predictions.csv"
            )

            response.header["X-Graph-Exists"] = "true"

            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")

            return response
        elif "text" in request.json:
            # Single string predictoin 
            text_input  = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)

            return jsonify({"prediction":prediction_sentiment})
        
        