import pandas as pd
import zipfile
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

archive_path = "C:/Users/Sathyam/Downloads/bank+marketing.zip"
output_dir = "C:/Users/Sathyam/Desktop/bank+marketing"

with zipfile.ZipFile(archive_path, 'r') as zf:
    zf.extractall(output_dir)

csv_file = None
for directory, subdirs, files in os.walk(output_dir):
    for filename in files:
        if filename.endswith(".csv"):
            csv_file = os.path.join(directory, filename)
            break
    if csv_file:
        break

if not csv_file:
    raise FileNotFoundError("CSV file not found inside the ZIP archive.")

data = pd.read_csv(csv_file, sep=';')
print("✅ Data successfully loaded:\n")
print(data.head())

encoders = {}
for feature in data.select_dtypes(include=['object']).columns:
    encoder = LabelEncoder()
    data[feature] = encoder.fit_transform(data[feature])
    encoders[feature] = encoder

features = data.drop(columns=['y'])
labels = data['y']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print("\n✅ Preprocessing complete and data split into training and test sets.")

classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
model_accuracy = accuracy_score(y_test, predictions)
print(f"\n📊 Decision Tree Accuracy: {model_accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print("\n📄 Classification Report:\n", classification_report(y_test, predictions))

importance_scores = classifier.feature_importances_
feature_labels = features.columns
plt.figure(figsize=(12, 6))
sns.barplot(x=importance_scores, y=feature_labels, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

new_data = np.array([[35, 2, 1, 2, 0, 2000, 1, 0, 1, 5, 30, 999, 0, 1, 0]])
new_prediction = classifier.predict(new_data)
result = "Yes" if new_prediction[0] == 1 else "No"
print(f"\n🧾 Prediction for the new customer: {result}")
