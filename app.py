from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        previous = float(request.form["previous"])
        sleep = float(request.form["sleep"])
        papers = float(request.form["papers"])

        # IMPORTANT: order must match training
        features = np.array([[hours, previous, sleep, papers]])
        prediction = round(model.predict(features)[0], 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
