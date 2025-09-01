from flask import Flask, request, render_template
import pickle
import os
from custom_transformers import NumericFeatures  # 👈 import your transformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained pipeline
with open(os.path.join(BASE_DIR, "/Users/stephanieijere/Documents/data_science/AI assignemts/Fake News Detector/notebooks/news_pipeline.pkl"), "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["news_text"]
    prediction = model.predict([text])[0]
    
    if prediction == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return render_template("index.html", prediction=result, input_text=text)

if __name__ == "__main__":
    app.run(debug=True)
