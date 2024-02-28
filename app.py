from flask import Flask, request, jsonify, render_template
from predict import predict_diabetes
from joblib import load
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Load the trained Random Forest model
model = load('models\\random_forest_model.joblib')
# Load the StandardScaler used for normalization
scaler = joblib.load('models\\standard_scaler.joblib')

# Create the Flask app
app = Flask(__name__)

# Define a route to render the index.html file
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()

    try:
        # Extract the features (glucose, BMI, age) from the input data
        glucose = float(data['glucose'])
        bmi = float(data['bmi'])
        age = float(data['age'])

        # Normalize the input features
        input_features = np.array([[glucose, bmi, age]])
        input_features_norm = scaler.transform(input_features)
        print(f'Input: {input_features}')
        print(f'Input Norm: {input_features_norm}')

        # Make predictions
        prediction = model.predict(input_features_norm)[0]
        prediction_probability = model.predict_proba(input_features_norm)[0] * 100
        print(f'Prediction: {prediction}')
        print(f'Prpbability: {prediction_probability}')

        # Return the prediction as a JSON response
        response = {
            'prediction': int(prediction),
            'probability_diabetic': round(prediction_probability[1], 2),
            'probability_non_diabetic': round(prediction_probability[0], 2)
        }
    except Exception as e:
        response = {
            'error': str(e)
        }

    return jsonify(response)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
