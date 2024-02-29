# Diabetes Prediction Web App

### Overview
This project aims to create a web application that predicts the likelihood of an individual being diabetic based on their glucose level, BMI, and age. The application uses a machine learning model trained on the Random Forest algorithm.

### Steps
- Data Preprocessing: The dataset is preprocessed to remove outliers and features not relevant to the prediction (Pregnancies, BloodPressure, SkinThickness, Insulin, DiabetesPedigreeFunction).
- Feature Engineering: Features are extracted and split into input features (glucose, BMI, age) and target feature (Outcome).
- Data Balancing: The dataset is balanced using the Synthetic Minority Over-sampling Technique (`SMOTE`) to address class imbalance.
- Normalization: Input features are normalized using the `StandardScaler` to ensure consistent scaling across features.
- Model Training: The `RandomForestClassifier` is trained on the preprocessed and normalized data to predict the likelihood of an individual being diabetic.
- Web App Development: The Flask framework is used to develop a web application that takes input from users (glucose, BMI, age), uses the trained model to make predictions, and displays the results.

### Techniques Used
- Machine Learning:  `Random Forest Classifier`
- Data Preprocessing: Outlier removal, feature selection, data balancing
- Normalization: `StandardScaler`
- Web Development: `Flask`

### Prediction
`predict_diabetes(glucose, bmi, age)`: Function that takes glucose, BMI, and age as input, normalizes the input features, and returns the prediction and probability of being diabetic.

### Usage
- Clone the repository.
- Install the required libraries `pip install -r requirements.txt`.
- Run `app.py`.
- Access the web application in a browser.

### Web App
![Screenshot (26)](https://github.com/HassaanIK/Diabetes-Prediction/assets/139614780/580c6192-fba5-494d-a321-691ac61e2ecc)
