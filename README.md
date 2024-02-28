# Diabetes Prediction
--
## OVERVIEW
This project is a Flask web application that classifies if the person is diabetic or non-diabetic on given some input features. It uses a machine learning model trained on features like glucose, BMI and age to make predictions. The user inputs these features into forms on the web app, and the app returns if the person is diabetic or not.

## SPECIFICATIONS
- The data used for training is taken from Kaggle. It has 8 different features out of which 3 are used.
- The preprocessing done on this data is sampling and removal of outliers.
- The features are normalized using StandardScaler from scikit learn library.
- The machine learning algorithm used is Random Forest Classifier as it was giving the best accuracy out of all.
- The metrics used for evaluation is accuracy, 78.67% of which is achieved.
- Trying deep learning on a complex architecture having 7 layers and 25k params gave about 80% accuracy.
- I used Pytorch as it feels interesting to use.
- The project uses Flask, a lightweight web framework for Python, to create the web application.
- The input features are normalized before being fed into the model for prediction.
  
## USAGE
```python
def predict_diabetes(glucose, bmi, age, model, scaler):


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
predict_diabetes(120, 30, 45, model, scaler)
