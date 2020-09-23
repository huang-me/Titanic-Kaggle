import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv('train.csv')

# creating a dict file
sexdic = {'male': 1,'female': 2}
train_data.Sex = [sexdic[item] for item in train_data.Sex]

# sex
sex = np.array(train_data.Sex).reshape(-1, 1)

# pclass
pclass = np.array(train_data.Pclass).reshape(-1, 1)

# family class
train_data['family_size'] = train_data.SibSp + train_data.Parch
train_data.loc[train_data['family_size'] == 0, 'family_cut'] = 0
train_data.loc[(train_data['family_size'] >= 1) & (train_data['family_size'] < 5), 'family_cut'] = 1
train_data.loc[(train_data['family_size'] >= 4) & (train_data['family_size'] < 7), 'family_cut'] = 2
train_data.loc[train_data['family_size'] >= 7, 'family_cut'] = 3
family_cut = np.array(train_data.family_cut).reshape(-1, 1)

# age class
label = LabelEncoder()
train_data.Age[np.isnan(train_data.Age)] = 25
train_data['age_class'] = pd.qcut(train_data.Age, 5)
train_data['age_class_code'] = label.fit_transform(train_data.age_class)
age_class = np.array(train_data.age_class_code).reshape(-1, 1)

feature = np.concatenate((sex, pclass, family_cut, age_class), axis=1)

survived = np.array(train_data.Survived).reshape(-1,1)

# initialize the model
model = DecisionTreeClassifier()
model = model.fit(feature, survived)

# get test file
test_data = pd.read_csv('test.csv')

# predict test file
test_data.Sex = [sexdic[item] for item in test_data.Sex]
sex = np.array(test_data.Sex).reshape(-1, 1)
pclass = np.array(test_data.Pclass).reshape(-1, 1)
parch = np.array(test_data.Parch).reshape(-1, 1)

test_data['family_size'] = test_data.SibSp + test_data.Parch
test_data.loc[test_data['family_size'] == 0, 'family_cut'] = 0
test_data.loc[(test_data['family_size'] >= 1) & (test_data['family_size'] < 5), 'family_cut'] = 1
test_data.loc[(test_data['family_size'] >= 4) & (test_data['family_size'] < 7), 'family_cut'] = 2
test_data.loc[test_data['family_size'] >= 7, 'family_cut'] = 3
family_cut = np.array(test_data.family_cut).reshape(-1, 1)

test_data.Age[np.isnan(test_data.Age)] = 25
test_data['age_class'] = pd.qcut(test_data.Age, 5)
test_data['age_class_code'] = label.fit_transform(test_data.age_class)
age_class = np.array(test_data.age_class_code).reshape(-1, 1)

feature = np.concatenate((sex, pclass, family_cut, age_class), axis=1)
print(feature.shape)
predict = model.predict(feature)

# output the result
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':predict})
output.to_csv('submission.csv', index=False)
print('submission save successfully')
