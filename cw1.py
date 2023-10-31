# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# dataset loading
data = pd.read_csv(r'C:\LHU\AI\Files\titanic3.csv') # using r to locate the file

# data preprocessing
data['age'].fillna(data['age'].mean(), inplace=True)
data['fare'].fillna(data['age'].median(), inplace=True)
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)
data.drop(['name', 'ticket', 'cabin', 'home.dest', 'boat', 'pclass', 'body', 'sibsp', 'parch'], axis=1, inplace=True)
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['embarked'] = le.fit_transform(data['embarked'])

x = data.drop('survived', axis=1)
y = data['survived']

# check columns datatypes
#print(data.dtypes) 
#print(data['survived'].isnull().sum())
data = data.dropna(subset=['survived'])
#data = data.dropna(subset=['sex'])
#data = data.dropna(subset=['age'])
#data = data.dropna(subset=['fare'])
#data = data.dropna(subset=['embarked'])

scaler = StandardScaler()
x = scaler.fit_transform(x)

# splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Get the indices of rows where y_train is not NaN
valid_indices = ~np.isnan(y_train)

# Filter x_train and y_train using these indices
x_train = x_train[valid_indices]
y_train = y_train[valid_indices]
#print(y_train.value_counts(dropna=False))

# training and evaluation
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression()
}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")