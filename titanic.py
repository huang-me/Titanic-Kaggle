import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

train_data = pd.read_csv('train.csv')

# set feature and survived data
train_data["Sex"] = train_data["Sex"].str.replace("female", "0")
train_data["Sex"] = train_data["Sex"].str.replace("male", "1")
feature = np.array(train_data.Sex).reshape(-1,1)
survived = np.array(train_data.Survived).reshape(-1,1)

# initialize the model
model = DecisionTreeClassifier()
model = model.fit(feature, survived)

# get test file
test_data = pd.read_csv('test.csv')

test_data["Sex"] = test_data["Sex"].str.replace("female", "0")
test_data["Sex"] = test_data["Sex"].str.replace("male", "1")
feature = np.array(test_data.Sex).reshape(-1,1)
print(model.predict(feature))
