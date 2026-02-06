from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    category = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        previous = float(request.form["previous"])
        sleep = float(request.form["sleep"])
        papers = float(request.form["papers"])

        features = np.array([[hours, previous, sleep, papers]])
        prediction = round(model.predict(features)[0], 2)

        # Clamp score
        prediction = max(0, min(100, prediction))

        if prediction < 40:
            category = "Low 📉"
        elif prediction < 70:
            category = "Medium 📊"
        else:
            category = "High 🚀"

    return render_template(
        "index.html",
        prediction=prediction,
        category=category
    )

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
