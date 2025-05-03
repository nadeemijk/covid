import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load and preprocess data
df = pd.read_csv("preprocessed_data.csv")
df = pd.get_dummies(df, dtype=int)

# Split features and target
x = df.drop(['DIED'], axis=1)
y = df['DIED']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

# Train XGBoost model
classifier = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
classifier.fit(x_train, y_train)

# Save the model
pickle.dump(classifier, open("model.pkl", "wb"))


print('The XGBoost model is successfully created and pickled. 😊')
