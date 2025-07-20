import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.utils.multiclass import unique_labels

# Load and prepare dataset
df = pd.read_csv("emotion_landmarks_dataset.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(" Accuracy:", round(acc * 100, 2), "%")

# Safe classification report
actual_classes = unique_labels(y_test, y_pred)
print("\n Report:\n", classification_report(
    y_test, y_pred, target_names=le.inverse_transform(actual_classes)
))

# Save model and encoder
joblib.dump(model, "emotion_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print(" Model + Label Encoder saved.")
