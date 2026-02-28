from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import re
from io import BytesIO

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # allow requests from http://localhost:5500 etc.

STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Load artifacts once (not every request)
predictor = pickle.load(open("./model_xgb.pkl", "rb"))
scaler = pickle.load(open("./scaler.pkl", "rb"))
cv = pickle.load(open("./countVectorizer.pkl", "rb"))


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running"


@app.route("/", methods=["GET"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Bulk prediction from CSV file
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            data = pd.read_csv(file)

            predictions_csv, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions_csv,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            # Custom headers for graph
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")

            return response

        # Single text prediction (JSON)
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            text_input = payload.get("text", "")
            if not text_input.strip():
                return jsonify({"error": "Missing 'text' field"}), 400

            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            return jsonify({"prediction": predicted_sentiment})

        return jsonify({"error": "Send either a CSV file (multipart/form-data) or JSON {'text': '...'}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def clean_text(text: str) -> str:
    review = re.sub(r"[^a-zA-Z]", " ", str(text))
    words = review.lower().split()
    words = [stemmer.stem(word) for word in words if word not in STOPWORDS]
    return " ".join(words)


def single_prediction(predictor, scaler, cv, text_input: str) -> str:
    corpus = [clean_text(text_input)]
    X = cv.transform(corpus).toarray()
    X_scl = scaler.transform(X)

    print("DEBUG text:", text_input)
    print("DEBUG cleaned:", corpus[0])
    print("DEBUG nonzero features:", int((X != 0).sum()), "of", X.size)

    X_scl = scaler.transform(X)

    proba = predictor.predict_proba(X_scl)
    #pred_class = int(proba.argmax(axis=1)[0])
    pred_class = int(predictor.predict(X_scl)[0])
    
    #return "Positive Sentiment" if pred_class == 1 else "Negative Sentiment"
    print("classes_:", getattr(predictor, "classes_", None))
    print("Nonzero features:", (X != 0).sum(), "out of", X.size)

    return "Positive Sentiment" if pred_class == 1 else "Negative Sentiment"




def bulk_prediction(predictor, scaler, cv, data: pd.DataFrame):
    if "Sentence" not in data.columns:
        raise ValueError("CSV must contain a 'Sentence' column.")

    corpus = [clean_text(s) for s in data["Sentence"].astype(str).tolist()]
    X = cv.transform(corpus).toarray()
    X_scl = scaler.transform(X)

    proba = predictor.predict_proba(X_scl)
    y = proba.argmax(axis=1)
    y = [sentiment_mapping(int(v)) for v in y]

    data = data.copy()
    data["Predicted sentiment"] = y

    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)
    return predictions_csv, graph


def get_distribution_graph(data: pd.DataFrame) -> BytesIO:
    fig = plt.figure(figsize=(5, 5))
    tags = data["Predicted sentiment"].value_counts()

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png", bbox_inches="tight")
    plt.close(fig)
    graph.seek(0)
    return graph


def sentiment_mapping(x: int) -> str:
    return "Positive" if x == 1 else "Negative"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)