# import libraries
import pandas as pd
import numpy as np
from numpy import hstack
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import warnings

# Hyperparameter ---
# Support vector machine
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': [1, 0.1, 0.01, 0.001]
}

# Gradient boosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'max_depth': [3, 4, 5]
}

# Initializations ---
svm_model = SVC(kernel='linear') # support vector machine
gb_model = GradientBoostingClassifier() # gradient boosting classifier
knn_model = KNeighborsClassifier(n_neighbors=5) # number of neighbors can be changed

grid_search_svm = GridSearchCV(SVC(), param_grid_svm, verbose=2, n_jobs=-1, cv=5)
grid_search_gb = GridSearchCV(GradientBoostingClassifier(), param_grid_gb, verbose=2, n_jobs=-1, cv=5)

reg = LinearRegression() # linear regression

base_estimator = DecisionTreeClassifier(max_depth=1) # ensemble method
ada_clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42) # ensemble method

model = AdaBoostClassifier(n_estimators=50) # cross validation

# Dataset Load ---
data = pd.read_csv(r'C:\LHU\AI\Files\titanic3.csv') # using r to locate the file

# Data pre-process ---
data['age'].fillna(data['age'].mean(), inplace=True) # fill empty with mean value
data['fare'].fillna(data['age'].median(), inplace=True) # fill empty with median value
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True) # fill empty with mode value

data['Title'] = data['name'].str.extract(r'([A-Za-z]+)\.', expand=False) # extract titles

# fill missing age values with median age
median_ages = data.groupby('Title')['age'].median()
for title, median_age in median_ages.items():
    condition = (data['age'].isnull()) & (data['Title'] == title)
    data.loc[condition, 'age'] = median_age

if data['age'].isnull().sum() > 0:
    data['age'].fillna(data['age'].median(), inplace=True)

# drop columns from the dataset
data.drop(['ticket', 'cabin', 'home.dest', 'boat', 'pclass', 'body', 'name'], axis=1, inplace=True)

le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['embarked'] = le.fit_transform(data['embarked'])
data['Title'] = le.fit_transform(data['Title'])

x = data.drop('survived', axis=1)
y = data['survived']

data = data.dropna(subset=['survived']) # remove NaN rows in the column
x = data.drop('survived', axis=1)
y = data['survived']

scaler = StandardScaler()
x = scaler.fit_transform(x)

scores = cross_val_score(model, x, y, cv=10) # for cross-validation

# converting string to numbers male=1 and female=0
title_mapping_numeric  = {"Mr": 1, "Miss/Mrs": 2}
data['Title'] = data['Title'].map(title_mapping_numeric)
# make a familySize feature  parch=parents/children of a  passenger
data['FamilySize'] = data['sibsp'] + data['parch'] + 1

# Split Data ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Get the indices of rows where y_train is not NaN
valid_indices = ~np.isnan(y_train)

# Filter x_train and y_train using these indices
x_train = x_train[valid_indices]
y_train = y_train[valid_indices]

# Elbow Method ---
# Suppress the specific FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)
kmeans = KMeans(n_clusters=2, n_init=10)

# check the best number of clusters for KMeans
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(x_train)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), inertia)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Normalize Data ---
scaler = StandardScaler()
# Fit on training set only
X_train = scaler.fit_transform(x_train)

# Apply same transformation to test set
X_test = scaler.transform(x_test)

# Train ---
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

# Support vector machine (hyperparameter tuning)
grid_search_svm.fit(x_train, y_train)

# Gradient boosting (hyperparameter tuning)
grid_search_gb.fit(x_train, y_train)

# Random forest with clustering
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)

# Regression
reg.fit(x_train, y_train)
y_train_pred = reg.predict(x_train)
y_test_pred = reg.predict(x_test)

# Ensemble method
ada_clf.fit(x_train, y_train)

# Threshold ---
# Regression
y_train_pred_class = [1 if prob > 0.5 else 0 for prob in y_train_pred]
y_test_pred_class = [1 if prob > 0.5 else 0 for prob in y_test_pred]
train_accuracy = accuracy_score(y_train, y_train_pred_class)
test_accuracy = accuracy_score(y_test, y_test_pred_class)

# Evaluation ---
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

# Support vector machine (hyperparameter tuning)
print(f"Optimized SVM Accuracy: {grid_search_svm.best_estimator_.score(x_test, y_test) * 100:.2f}%")

# Gradient boosting (hyperparameter tuning)
print(f"Optimized Gradient Boosting Accuracy: {grid_search_gb.best_estimator_.score(x_test, y_test) * 100:.2f}")

# Random forest with clustering
print(f"Random Forest Accuracy (with clustering): {rf.score(x_test, y_test) * 100:.2}%")

# Regression
print(f"Training Accuracy (regression): {train_accuracy:.2%}")
print(f"Test Accuracy (regression): {test_accuracy:.2%}")

# Ensemble method
print(f"Training Accuracy (ensemble method): {accuracy_score(y_train, ada_clf.predict(x_train))*100:.2f}%")
print(f"Test Accuracy (ensemble method): {accuracy_score(y_test, ada_clf.predict(x_test))*100:.2f}%")

# Cross-validation
print(f"Cross-Validation Scores: {scores}")
print(f"Mean Accuracy: {scores.mean()}")
print(f"Standard Deviation: {scores.std()}")