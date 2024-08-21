import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# Load the Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest classifier
model = RandomForestClassifier(random_state=42)

# Initialize RFE with the classifier and the number of features to select
selector = RFE(estimator=model, n_features_to_select=2, step=1)

# Fit RFE on the training data
selector = selector.fit(X_train, y_train)

# Get the selected features
selected_features = X_train.columns[selector.support_]

print("Selected features:", selected_features)

# Train the model using the selected features
model.fit(X_train[selected_features], y_train)

# Evaluate the model on the test set
accuracy = model.score(X_test[selected_features], y_test)
print(f"Model accuracy with selected features: {accuracy:.2f}")




#output:
#Selected features: Index(['petal length (cm)', 'petal width (cm)'], dtype='object')
#Model accuracy with selected features: 1.00

