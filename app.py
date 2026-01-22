# Import Flask framework components for web app development
from flask import Flask, render_template, request
# Import joblib for loading saved machine learning models
import joblib
# Import numpy for numerical operations and array handling
import numpy as np

# Create a Flask application instance
app = Flask(__name__)

# Load the pre-trained Titanic survival prediction model from file
model = joblib.load("model/titanic_survival_model.pkl")
# Load the scaler used for feature normalization during model training
scaler = joblib.load("model/scaler.pkl")

# Define the route for the home page, accepting both GET and POST requests
@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize prediction variable to None
    prediction = None

    # Check if the request method is POST (form submission)
    if request.method == "POST":
        # Extract and convert passenger class from form data to integer
        pclass = int(request.form["pclass"])
        # Extract and convert sex from form data to integer (0=Female, 1=Male)
        sex = int(request.form["sex"])
        # Extract and convert age from form data to float
        age = float(request.form["age"])
        # Extract and convert fare from form data to float
        fare = float(request.form["fare"])
        # Extract and convert embarked port from form data to integer (0=C, 1=Q, 2=S)
        embarked = int(request.form["embarked"])

        # Create a numpy array with the input features
        data = np.array([[pclass, sex, age, fare, embarked]])
        # Apply the same scaling transformation used during training
        data = scaler.transform(data)

        # Make prediction using the loaded model and get the first (only) result
        result = model.predict(data)[0]
        # Convert numerical prediction to human-readable string
        prediction = "Survived" if result == 1 else "Did Not Survive"

    # Render the HTML template with the prediction result
    return render_template("index.html", prediction=prediction)

# Check if this script is being run directly (not imported)
if __name__ == "__main__":
    # Start the Flask development server with debug mode enabled
    app.run(debug=True)
