import pandas as pd
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --- Data Collection ---

zip_file_path = "C:\\Users\\guffs\\OneDrive - santero limited\\Documents\\Projects\\creditcardfraud\\archive (1).zip"
extract_path = "C:\\Users\\guffs\\OneDrive - santero limited\\Documents\\Projects\\creditcardfraud\\extracted"

# Create the extraction directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract the CSV file
try:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        csv_file_to_extract = 'creditcard.csv'
        zip_ref.extract(csv_file_to_extract, extract_path)
        print(f"Successfully extracted '{csv_file_to_extract}' to '{extract_path}'")
except FileNotFoundError:
    print(f"Error: Zip file not found at '{zip_file_path}'.")
    exit() # Exit if the zip file is not found
except KeyError:
    print(f"Error: '{csv_file_to_extract}' not found in the zip archive.")
    exit() # Exit if the CSV file is not in the zip

# Load the extracted CSV file into a pandas DataFrame
csv_file_path = os.path.join(extract_path, 'creditcard.csv')

try:
    df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded data from '{csv_file_path}'")
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found after extraction.")
    df = None # Set df to None if the file is not found
    exit() # Exit if the CSV file is not found after extraction

# --- Data Preprocessing ---

if df is not None:
    print("\nData Overview:")
    print(df.head())
    print("\nData Info:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDistribution of the target variable ('Class'):")
    print(df['Class'].value_counts())
    fraud_percentage = (df['Class'].value_counts()[1] / len(df)) * 100
    print(f"\nPercentage of fraudulent transactions: {fraud_percentage:.4f}%")

    # No explicit missing value handling needed as df.isnull().sum() shows 0 for all columns.
    # For outliers and imbalanced data, further techniques would be needed in a real project,
    # but for this consolidated script, we'll proceed with the basic dataset.

    # --- Feature Engineering ---

    # Convert 'Time' to hours of the day (assuming a 24-hour cycle)
    df['Hour'] = (df['Time'] / 3600) % 24
    print("\nDataFrame with new 'Hour' feature:")
    print(df.head())

    # --- Model Selection ---

    # Define features (X) and target (y)
    X = df.drop(['Time', 'Class'], axis=1) # Exclude 'Time' and 'Class' from features
    y = df['Class']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nTraining set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    # Initialize and train a Logistic Regression model
    model = LogisticRegression(solver='liblinear', random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    print("\nLogistic Regression model trained.")

    # --- Model Evaluation ---

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Get probabilities for the positive class (fraudulent)

    # Evaluate the model
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nAUC-ROC Score:")
    print(roc_auc_score(y_test, y_pred_proba))

    # --- Model Deployment (Placeholder) ---
    print("\n--- Model Deployment ---")
    print("Deployment strategies would be implemented here (e.g., saving the model, building an API).")

    # --- Model Monitoring and Maintenance (Placeholder) ---
    print("\n--- Model Monitoring and Maintenance ---")
    print("Monitoring and maintenance procedures would be defined and implemented here.")

    # --- Finish Task (Summary) ---
    print("\n--- Project Summary ---")
    print("This script performed the following steps:")
    print("- Loaded credit card transaction data.")
    print("- Performed basic data exploration.")
    print("- Engineered a simple 'Hour' feature.")
    print("- Split data into training and testing sets.")
    print("- Trained a Logistic Regression model.")
    print("- Evaluated the model using confusion matrix, classification report, and AUC-ROC score.")
    print("Further steps for a complete project would include advanced preprocessing, feature engineering, model tuning, deployment, and monitoring.")