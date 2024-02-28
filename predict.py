import joblib
from sklearn.preprocessing import StandardScaler

def predict_diabetes(glucose, bmi, age):
    # Load the trained Random Forest model
    model = joblib.load('models\\random_forest_model1.joblib')
    # Load the StandardScaler used for normalization
    scaler = joblib.load('models\\standard_scaler.joblib')

    # Normalize the input features
    input_features = [[glucose, bmi, age]]
    input_features_norm = scaler.fit_transform(input_features)

    # Make predictions
    prediction = model.predict(input_features_norm)[0]
    prediction_probability = model.predict_proba(input_features_norm)[0]

    if prediction == 0:
        result = "Non-Diabetic"
    else:
        result = "Diabetic"

    return result, prediction_probability
