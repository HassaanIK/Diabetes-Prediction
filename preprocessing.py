import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print('Data Loaded..')
    df = df.drop(columns=['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction'])
    print('Columns Dropped..')
    Q1 = df[['Glucose', 'BMI', 'Age']].quantile(0.25)
    Q3 = df[['Glucose', 'BMI', 'Age']].quantile(0.75)
    IQR = Q3 - Q1

    outliers = ((df[['Glucose', 'BMI', 'Age']] < (Q1 - 1.5 * IQR)) | (df[['Glucose', 'BMI', 'Age']] > (Q3 + 1.5 * IQR))).any(axis=1)
    df = df[~outliers]
    print('Outliers Removed..')

    features_neq = df.drop('Outcome', axis=1)
    targets_neq = df['Outcome']

    features_np, targets_np = features_neq.to_numpy(), targets_neq.to_numpy()

    os = SMOTE(sampling_strategy='minority', random_state=2007)
    features, targets = os.fit_resample(features_np, targets_np)

    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=2007)
    print('Splitted..')
    # Normalize the data
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    print('Normalized..')
    
    # Assuming scaler is your StandardScaler object
    dump(scaler, 'models\\standard_scaler.joblib')
    print('Scaler Saved...')


    return X_train_norm, X_val_norm, y_train, y_val

