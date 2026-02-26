import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Loading dataset
file = 'database.csv'
data = pd.read_csv(file)

# Showing counts of each class
print("Number of benign samples:", data[data['classification'] == 'benign'].shape[0])
print("Number of malware samples:", data[data['classification'] == 'malware'].shape[0])

data = data.drop(['usage_counter', 'normal_prio', 'policy', 'vm_pgoff', 'task_size',
                  'cached_hole_size', 'hiwater_rss', 'nr_ptes', 'cgtime', 'signal_nvcsw'], axis=1)

# Convert classification to numeric
data['classification'] = data['classification'].map({'benign': 0, 'malware': 1})

# Separate features and target
target = data['classification']
features = data.drop(['classification', 'hash'], axis=1)

# Standardize the data
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Deep Learning model
model_dl = Sequential()
model_dl.add(Dense(16, input_dim=23, activation="relu"))
model_dl.add(Dense(8, activation="relu"))
model_dl.add(Dense(4, activation="relu"))
model_dl.add(Dense(1, activation="sigmoid"))
model_dl.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Train the model
model_dl.fit(X_train_scaled, y_train, epochs=5, batch_size=32)

# Evaluate on training data
train_pred_dl = model_dl.predict(X_train_scaled)
train_pred_dl = [1 if y >= 0.5 else 0 for y in train_pred_dl]
train_accuracy_dl = accuracy_score(y_train, train_pred_dl)
print("Deep Learning - Training data accuracy:", train_accuracy_dl)

# Evaluate on test data
test_pred_dl = model_dl.predict(X_test_scaled)
test_pred_dl = [1 if y >= 0.5 else 0 for y in test_pred_dl]
test_accuracy_dl = accuracy_score(y_test, test_pred_dl)
print("Deep Learning - Test data accuracy:", test_accuracy_dl)

# Calculate F1 score
f1 = f1_score(y_test, test_pred_dl)
print("F1 score:", f1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, test_pred_dl)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()