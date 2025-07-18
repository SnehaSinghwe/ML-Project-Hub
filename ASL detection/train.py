import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import glob

# Load all CSV files
files = glob.glob("Unified_mentor/new_ASL/asl_landmark_data/landmarks_*.csv")
data = pd.concat([pd.read_csv(f, header=None) for f in files], axis=0)

# Separate features and labels
X = data.iloc[:, 1:].values  # 63 features
y = data.iloc[:, 0].values

# Encode labels (A-Z → 0–25)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Train MLP model
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and encoder
joblib.dump(model, "Unified_mentor/new_ASL/asl_mlp_model.pkl")
joblib.dump(le, "Unified_mentor/new_ASL/label_encoder.pkl")
print("[INFO] Model and label encoder saved.")
