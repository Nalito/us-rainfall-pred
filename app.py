# Run me using "flask run" while within the app directory in cli.
# send post requests to "http://127.0.0.1:5000/usa-rain-predictor/"

# URLs to useful resources...
# RESTful APIs for models -> https://medium.com/@einsteinmunachiso/rest-api-implementation-in-python-for-model-deployment-flask-and-fastapi-e80a6cedff86
# Serializing/Deserializing models -> https://medium.com/@einsteinmunachiso/saving-your-machine-learning-model-in-python-pickle-dump-b01ae60a791c

import numpy as np
import pickle # for deserialization of saved model
from flask import Flask, request, jsonify, render_template

import warnings
warnings.filterwarnings("ignore")

# Creating a flask instance.
app = Flask(__name__)

# Model and encoder paths.
model_file_path = "model.pkl"
location_encoder_path = "location_encoder.pkl"

# deserializing saved model and encoder.
with open(model_file_path, "rb") as file:
    model = pickle.load(file)

with open(location_encoder_path, "rb") as file:
    loc_enc = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        location = request.form.get("location")
        temperature = float(request.form.get("temperature"))
        humidity = float(request.form.get("humidity"))
        wind_speed = float(request.form.get("wind_speed"))
        precipitation = float(request.form.get("precipitation"))
        cloud_cover = float(request.form.get("cloud_cover"))
        pressure = float(request.form.get("pressure"))

        location = loc_enc.transform([location])[0]

        pred = model.predict([[location, temperature, humidity, wind_speed, precipitation, cloud_cover, pressure]])[0]

        if pred == 1:
            prediction = "It is going to rain tomorrow."
        else:
            prediction = "It won't rain tomorrow."
        return render_template("index.html", pred=prediction)



if __name__ == "__main__":
    app.run(debug=True)


