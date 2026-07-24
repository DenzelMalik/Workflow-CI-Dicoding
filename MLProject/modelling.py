import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Enable autolog
mlflow.sklearn.autolog()

dataset_path = "namadataset_preprocessing/EV_Adoption_and_Range_Anxiety_Dataset_clean.csv"

if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} not found.")
    exit(1)

df = pd.read_csv(dataset_path)

X = df.drop('Will_Buy_EV', axis=1)
y = df['Will_Buy_EV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="CI_Pipeline_Run"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"CI Run Accuracy: {accuracy:.4f}")
