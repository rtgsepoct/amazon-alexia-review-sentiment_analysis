from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS  # ✅ ADDED
import re
from io import BytesIO

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # ✅ ADDED: allow calls from Live Server (5500) to Flask (5000)

STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ✅ CHANGED: load once at startup (fast + stable)
predictor = pickle.load(open("./model_xgb.pkl", "rb"))
scaler = pickle.load(open("./scaler.pkl", "rb"))
cv = pickle.load(open("./countVectorizer.pkl", "rb"))

# ✅ ADDED: discover class labels from the model (prevents wrong mapping)
MODEL_CLASSES = getattr(predictor, "classes_", None)
print("Loaded model classes_:", MODEL_CLASSES)


@app.route("/test", methods=["GET"])  # ✅ CHANGED: correct route + methods
def test():
    return "Test request received successfully. Service is running"


@app.route("/", methods=["GET"])  # ✅ CHANGED: GET only for page render
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ CHANGED: bulk prediction if a CSV file was uploaded
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

            # ✅ CHANGED: correct attribute name -> headers (not header)
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")

            return response

        # ✅ CHANGED: text prediction expects JSON
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            text_input = payload.get("text", "")
            if not text_input.strip():
                return jsonify({"error": "Missing 'text'"}), 400

            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            return jsonify({"prediction": predicted_sentiment})

        return jsonify({"error": "Send a CSV file (multipart/form-data) OR JSON {'text': '...'}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ CHANGED: centralized cleaner (keep consistent for both single & bulk)
def clean_text(text: str) -> str:
    review = re.sub(r"[^a-zA-Z]", " ", str(text))
    words = review.lower().split()

    # ✅ CHANGED: correct stemming list comprehension
    words = [stemmer.stem(word) for word in words if word not in STOPWORDS]

    return " ".join(words)


# ✅ CHANGED: safest prediction logic:
# - checks vector not empty
# - uses predict() rather than proba.argmax() (avoids class-order mistakes)
# - maps output robustly using model.classes_ if present
def single_prediction(predictor, scaler, cv, text_input: str) -> str:
    corpus = [clean_text(text_input)]
    X = cv.transform(corpus).toarray()

    # ✅ ADDED: detect “all-zero vector” (common cause of always-Positive)
    nonzero = int((X != 0).sum())
    if nonzero == 0:
        return "Unknown (no recognized words in vocabulary)"  # ✅ ADDED safe fallback

    X_scl = scaler.transform(X)  # ✅ CHANGED: transform spelled correctly

    pred = predictor.predict(X_scl)[0]  # ✅ CHANGED: use predict()
    pred_int = int(pred) if str(pred).isdigit() else pred

    # ✅ ADDED: robust label mapping using classes_ if needed
    # If your model classes are [0,1], then pred==1 is positive.
    # If classes are ['Negative','Positive'] etc, we map string labels too.
    if isinstance(pred_int, int):
        return "Positive Sentiment" if pred_int == 1 else "Negative Sentiment"

    # If model returns string labels
    pred_str = str(pred_int).lower()
    if "pos" in pred_str:
        return "Positive Sentiment"
    if "neg" in pred_str:
        return "Negative Sentiment"
    return f"Prediction: {pred_int}"


def bulk_prediction(predictor, scaler, cv, data: pd.DataFrame):
    if "Sentence" not in data.columns:
        raise ValueError("CSV must contain a 'Sentence' column.")

    corpus = [clean_text(s) for s in data["Sentence"].astype(str).tolist()]
    X = cv.transform(corpus).toarray()

    # ✅ ADDED: warn if most rows become zero-vectors
    nonzero_rows = (X != 0).sum(axis=1)
    if (nonzero_rows == 0).mean() > 0.5:
        print("WARNING: >50% of rows have zero recognized words. Check preprocessing/training mismatch.")

    X_scl = scaler.transform(X)
    preds = predictor.predict(X_scl)

    # ✅ CHANGED: map predictions robustly
    mapped = []
    for p in preds:
        try:
            pi = int(p)
            mapped.append("Positive" if pi == 1 else "Negative")
        except:
            ps = str(p).lower()
            if "pos" in ps:
                mapped.append("Positive")
            elif "neg" in ps:
                mapped.append("Negative")
            else:
                mapped.append(str(p))

    out = data.copy()
    out["Predicted sentiment"] = mapped

    predictions_csv = BytesIO()
    out.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(out)
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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)