# Develop a linear regression model, decision tree, and random forest classifiers to predict whether a given email is spam or not. Use the SpamAssassin datasetLinks to an external site. and implement these classifiers in Python using scikit-learn. (10 points)
#
# Load the SpamAssassin dataset (`spam_dataset.csv`) using pandas.
# Preprocess the data (handle missing values, categorical encoding, etc.).
# Split the data into training and testing sets.
# Implement a decision tree classifier using scikit-learn.
# Train the model on the training set.
# Evaluate the model's performance using appropriate metrics.
# Include comments to explain each step of your code.

#Script-Assignment-part-1
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mlxtend.plotting import plot_decision_regions
from sklearn import tree
# Load the SpamAssassin dataset 
#  'spam_assassin.csv' with the actual path to your dataset
data = pd.read_csv('spam_assassin.csv')

# Assuming your dataset has a 'text' column containing email text and a 'target' column indicating spam or not
X = data['text']
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Build and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_vectorized)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")

#plot for decision tree
print("\n plot for decision tree-")
fig, ax = plt.subplots(figsize=(15, 15))
tree.plot_tree(clf, fontsize=10)
plt.show()