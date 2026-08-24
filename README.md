# Flat Price Predictor

## 🏠 Overview

**Flat Price Predictor** is a Machine Learning project that predicts the estimated price of a flat based on different property-related features.

The project demonstrates an end-to-end ML workflow:

**Data → Preprocessing → Model Training → Evaluation → Prediction → Deployment**

## 🎯 Features

* 📊 Data preprocessing and cleaning
* 🤖 Machine Learning model training
* 🏠 Flat price prediction
* 📈 Model evaluation
* 💾 Trained model saved using Pickle
* 🌐 Flask backend for serving predictions
* 🖥️ Web-based prediction interface

## 🛠️ Technologies Used

* **Python**
* **Pandas** — Data processing
* **NumPy** — Numerical operations
* **Scikit-learn** — Machine Learning
* **Flask** — Backend/API
* **HTML & CSS & js** — Frontend
* **Pickle** — Model serialization
* **Git & GitHub** — Version control

## 📁 Project Structure

```text
flat_price_predictor/
│
├── data/
│   └── dataset.csv
│
├── model/
│   └── model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## ⚙️ How It Works

### 1. Collect Data

A dataset containing flat/property information is used for training.

Example features may include:

```text
Area
Bedrooms
Bathrooms
Location
Parking
Age of Property
```

### 2. Preprocess Data

The dataset is cleaned and converted into a format suitable for Machine Learning.

### 3. Train the Model

A regression algorithm is trained using the property features.

```python
model.fit(X_train, y_train)
```

### 4. Save the Model

The trained model is saved as a `.pkl` file.

```python
import pickle

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)
```

### 5. Flask Prediction

The Flask application loads the trained model and receives property information from the user.

```python
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

prediction = model.predict([features])
```

### 6. Display Result

The predicted flat price is returned to the frontend and displayed to the user.

## 🚀 Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd flat_price_predictor
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

First train the model:

```bash
python train_model.py
```

Then start Flask:

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

## 🧠 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Save Model (.pkl)
   ↓
Flask Backend
   ↓
User Input
   ↓
Prediction
   ↓
Predicted Flat Price
```

## 📊 Model Evaluation

The model can be evaluated using regression metrics such as:

* **MAE** — Mean Absolute Error
* **MSE** — Mean Squared Error
* **RMSE** — Root Mean Squared Error
* **R² Score** — Coefficient of Determination

Example:

```python
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R² Score:", r2)
```

## 🔮 Future Improvements

* Improve prediction accuracy
* Add more property features
* Try Random Forest, Gradient Boosting, and XGBoost
* Add location-based features
* Add data visualization
* Deploy the application online
* Add a database for storing predictions
* Add model monitoring

## 📚 Learning Outcome

Through this project, you can learn how to:

* Build a regression Machine Learning model
* Prepare real-world datasets
* Train and evaluate a model
* Save and load models using Pickle
* Connect an ML model with Flask
* Send user input from a frontend to a backend
* Build an end-to-end ML application

## 👨‍💻 Author

**Santanu and Team**

B.Tech — Artificial Intelligence & Machine Learning

---

⭐ If you found this project useful, consider giving the repository a star!
