import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Generate a random linearly separable dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Support Vector Classifier (SVC)
clf = LinearSVC()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing data
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Check if the data is linearly separable
if accuracy == 1.0:
    print("The data is linearly separable.")
else:
    print("The data is not linearly separable.")
