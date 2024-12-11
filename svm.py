#esha alvi sp23-bai-015
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Define the path for the CSV file
csv_file_path = r"/content/creditcard.csv"

# Load the dataset
try:
    data = pd.read_csv(csv_file_path)
    print("\nDataset loaded successfully!")
except FileNotFoundError:
    print("CSV file not found. Ensure the file name matches.")
    exit()

# Display dataset information
print("\nDataset Information:")
print(data.info())


X = data.drop(columns=['Class']).iloc[:, :5]  
y = data['Class']

#  StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Check for missing or invalid values in the features and target
if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
    print("Missing values found in the training data!")
    X_train = np.nan_to_num(X_train)  # Replace NaNs with 0 or a specific value
    y_train = np.nan_to_num(y_train)  # Same for the target variable

# Ensure the target variable is binary and numeric
y_train = y_train.astype(int)

# Initialize the SVM classifier
svm_model = SVC(random_state=42, kernel='linear', class_weight='balanced')

random_params = {
    'C': np.logspace(-3, 3, 4), 
    'gamma': ['scale', 'auto'],
    'kernel': ['linear']
}

random_search = RandomizedSearchCV(
    estimator=svm_model,
    param_distributions=random_params,
    scoring='f1',
    n_iter=2, 
    cv=3,
    random_state=42,
    n_jobs=-1
)


random_search.fit(X_train, y_train)

# Get the best model from RandomizedSearchCV
best_random_model = random_search.best_estimator_
print("\nBest Parameters (Random Search):", random_search.best_params_)

# Predict using the best model
y_pred_test = best_random_model.predict(X_test)


def evaluate_model(y_test, y_pred, model_name):
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
    plt.yticks([0, 1], ["Non-Fraud", "Fraud"])
    plt.show()

evaluate_model(y_test, y_pred_test, "SVM with Random Search") 