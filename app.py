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

    # Check model
    if model is None:
        return jsonify({
            "error": "Model not loaded on server."
        }), 500

    try:

        # -----------------------------
        # Get data from HTML
        # -----------------------------

        area = float(request.form.get("area"))
        facing = request.form.get("facing")
        floor = float(request.form.get("floor"))
        bedrooms = float(request.form.get("bedrooms"))
        car_parking = float(request.form.get("car_parking"))


        # -----------------------------
        # Validate input
        # -----------------------------

        if area <= 0:
            return jsonify({
                "error": "Area must be greater than 0"
            }), 400

        if floor <= 0:
            return jsonify({
                "error": "Floor must be greater than 0"
            }), 400

        if bedrooms <= 0:
            return jsonify({
                "error": "Bedrooms must be greater than 0"
            }), 400

        if car_parking < 0:
            return jsonify({
                "error": "Car parking cannot be negative"
            }), 400


        # -----------------------------
        # Facing encoding
        # -----------------------------

        facing_map = {
            "East": 1,
            "West": 2,
            "North": 3,
            "South": 4
        }

        facing_value = facing_map.get(facing)

        if facing_value is None:
            return jsonify({
                "error": "Invalid facing value"
            }), 400


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

        input_array = np.array(
            input_values
        ).reshape(1, -1)

        logging.info(
            "Model input: %s",
            input_array
        )


        # -----------------------------
        # Make prediction
        # -----------------------------

        prediction = model.predict(input_array)

        predicted_price = float(prediction[0])


        logging.info(
            "Predicted price: %s",
            predicted_price
        )


        # -----------------------------
        # Send result to JavaScript
        # -----------------------------

        return jsonify({
            "prediction": predicted_price
        })


    except ValueError as e:

        logging.exception(
            "Invalid input: %s",
            e
        )

        return jsonify({
            "error": "Please enter valid numeric values."
        }), 400


    except Exception as e:

        logging.exception(
            "Prediction failed: %s",
            e
        )

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