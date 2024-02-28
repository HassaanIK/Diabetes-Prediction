from preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

# Preprocess the data
X_train_norm, X_val_norm, y_train, y_val = preprocess_data('data\\diabetes.csv')

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=2007)

# Train the model
model.fit(X_train_norm, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val_norm)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)

print("Accuracy:", accuracy)

# Save the model to a file
file_path = 'models\\random_forest_modelf.joblib'
dump(model, file_path)
print('Saved at ', file_path)
