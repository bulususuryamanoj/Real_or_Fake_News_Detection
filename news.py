# app.py
from flask import Flask, render_template, request
import pickle
import re

# Load pipeline
with open("fake_news_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news_text = request.form["news_text"]

        # Clean text
        cleaned_text = re.sub(r"[^a-zA-Z ]", "", news_text.lower())

        # Predict
        prediction = pipeline.predict([cleaned_text])[0]
        probability = pipeline.predict_proba([cleaned_text]).max() * 100

        if prediction == 1:
            result = f"✅ This looks REAL with {probability:.2f}% confidence."
        else:
            result = f"❌ This looks FAKE with {probability:.2f}% confidence."

        return render_template("index.html", prediction=result, news=news_text)

if __name__ == "__main__":
    app.run(debug=True)

