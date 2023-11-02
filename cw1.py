# import libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# initializations ----------------------------------------------------------------------------
svm_model = SVC(kernel='linear') # SVM
gb_model = GradientBoostingClassifier()
knn_model = KNeighborsClassifier(n_neighbors=5) # number of neighbors can be changed

# dataset loading ---------------------------------------------------------------------------
data = pd.read_csv(r'C:\LHU\AI\Files\titanic3.csv') # using r to locate the file

# data preprocessing ------------------------------------------------------------------------
data['age'].fillna(data['age'].mean(), inplace=True)
data['fare'].fillna(data['age'].median(), inplace=True)
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)
data.drop(['name', 'ticket', 'cabin', 'home.dest', 'boat', 'pclass', 'body'], axis=1, inplace=True)
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['embarked'] = le.fit_transform(data['embarked'])

x = data.drop('survived', axis=1)
y = data['survived']

data = data.dropna(subset=['survived']) # remove NaN rows in the column

scaler = StandardScaler()
x = scaler.fit_transform(x)
# create a new column in the dataset
data['Title'] = data['sex'].map({'male': 'Mr',  'female': 'Miss/Mrs'})
# converting string to numbers male=1 and female=0
title_mapping_numeric  = {"Mr": 1, "Miss/Mrs": 2}
data['Title'] = data['Title'].map(title_mapping_numeric)
# make a familySize feature  parch=parents/children of a  passenger
data['FamilySize'] = data['sibsp'] + data['parch'] + 1

# splitting data ----------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Get the indices of rows where y_train is not NaN
valid_indices = ~np.isnan(y_train)

# Filter x_train and y_train using these indices
x_train = x_train[valid_indices]
y_train = y_train[valid_indices]

# normalize data ----------------------------------------------------------------------------
scaler = StandardScaler()
# Fit on training set only
X_train = scaler.fit_transform(x_train)

# Apply same transformation to test set
X_test = scaler.transform(x_test)

# training ----------------------------------------------------------------------------------
# Decision tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Random forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Logistic regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Support vector machine
svm_model.fit(x_train, y_train)

# Gradient boosting
gb_model.fit(x_train, y_train)

# K-Nearest Neighbors
knn_model.fit(x_train, y_train)

# Evaluation --------------------------------------------------------------------------------
# Decision tre
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred) * 100:.2f}%")

# Random forest
print(f"Random forest Accuracy: {accuracy_score(y_test, rf_pred) * 100:.2f}%") 

# Logistic regression
print(f"Logistic regression Accuracy: {accuracy_score(y_test, lr_pred) * 100:.2f}%") 

# Support vector machine
print(f"SVM Accuracy: {svm_model.score(x_test, y_test) * 100:.2f}%") 

# Gradient boosting
print(f"Gradient boosting Accuracy: {gb_model.score(x_test, y_test) * 100:.2f}%") 

# K-Nearest Neighbors
print(f"KNN Accuracy: {knn_model.score(x_test, y_test) * 100:.2f}%")