from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO

# nltk.cropus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
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
    except Exception as e: 
        return jsonify({"error": str(e)})
    

# Single Prediction: 
def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word for word in review if not word in STOPWORDS)]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.tranform(X_prediction)
    y_prediction = predictor.predict_proba(X_prediction_scl)
    y_prediction = y_predictions.argmax(axis=1)[0] 


    return "Positive" if y_predictions ==1 else  "Negative"

# BULK Prediction 
def bulk_prediction(predictor, scaler, cv, data):
    def bulk_prediction(predictor, scaler, cv, data):
        corpus = []
        stemmer = PorterStemmer ()
        for i in range(0, data. shape[0]):
            review = re.sub("[^a-zA-Z]".
            , "", data. iloc[i][ "Sentence"])
            review = review. lower(). split()
            review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
            review = " ".join(review)
            corpus. append (review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_sc1)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data[ "Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()

    data. to_csv(predictions_csv, index=False) 
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)
    return predictions_csv, graph


def get_distribution_graph(data) :
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"Linewidth": 1, "edgecolor": "black"}
    tags = data[ "Predicted sentiment"]. value_counts ()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True, 
        colors=colors, 
        startangle=90, 
        wedgeprops=wp,
        explode=explode,
        title= "Sentiment Distribution", 
        xlabel=""
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph

def sentiment_mapping(x): 
    if x==1:
        return "Positive"
    else: 
        return "Negative"

if __name__ == "__main__":
    app.run(port=5000, debug = True)
