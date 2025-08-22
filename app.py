from flask import Flask, request, render_template
import pickle
import os

# Load the saved model and vectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["news_text"]
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return render_template("index.html", prediction=result, input_text=text)

if __name__ == "__main__":
    app.run(debug=True)
