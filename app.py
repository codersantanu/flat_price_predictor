from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)


# -----------------------------
# Load ML model
# -----------------------------

try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    logging.info("Model loaded successfully.")

except Exception as e:
    model = None
    logging.exception("Failed to load model: %s", e)


# -----------------------------
# Home page
# -----------------------------

@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# Prediction API
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model not loaded on server."
        }), 500

    try:

        # Get values from HTML form
        area = float(request.form.get("area"))
        facing = request.form.get("facing")
        floor = float(request.form.get("floor"))
        bedrooms = float(request.form.get("bedrooms"))
        car_parking = float(request.form.get("car_parking"))

        # -----------------------------
        # Encode categorical variable
        # -----------------------------

        facing_map = {
            "East": 1,
            "West": 2,
            "North": 3,
            "South": 4
        }

        facing_value = facing_map[facing]

        # -----------------------------
        # Create model input
        # -----------------------------

        input_values = [
            area,
            facing_value,
            floor,
            bedrooms,
            car_parking
        ]

        input_array = np.array(input_values).reshape(1, -1)

        logging.info("Input: %s", input_array)

        # -----------------------------
        # Prediction
        # -----------------------------

        prediction = model.predict(input_array)

        predicted_price = float(prediction[0])

        logging.info("Prediction: %s", predicted_price)

        # -----------------------------
        # Return JSON
        # -----------------------------

        return jsonify({
            "prediction": predicted_price
        })

    except Exception as e:

        logging.exception("Prediction failed: %s", e)

        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


# -----------------------------
# Run Flask
# -----------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )