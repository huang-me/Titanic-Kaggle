import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# get file
test_data = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
train_data = train.append(test_data)

# creating a dict file
sexdic = {'male': 1,'female': 2}
train_data.Sex = [sexdic[item] for item in train_data.Sex]

# sex
sex = np.array(train_data.Sex).reshape(-1, 1)

# pclass
train_data['pclass'] = (train_data['Pclass'] == 0) * 1
pclass = np.array(train_data.pclass).reshape(-1, 1)

# family class
train_data['family_size'] = train_data.SibSp + train_data.Parch
train_data.loc[train_data['family_size'] == 0, 'family_cut'] = 0
train_data.loc[(train_data['family_size'] >= 1) & (train_data['family_size'] < 5), 'family_cut'] = 1
train_data.loc[(train_data['family_size'] >= 4) & (train_data['family_size'] < 7), 'family_cut'] = 2
train_data.loc[train_data['family_size'] >= 7, 'family_cut'] = 3
family_cut = np.array(train_data.family_cut).reshape(-1, 1)

# find age with nan and replace with median of name
train_data['name'] = train_data['Name'].str.extract("([A-Za-z]+)\.", expand=False)
train_data['name'] = train_data['name'].replace(['Mlle', 'Ms', 'Mme'], 'Miss')
train_data['name'] = train_data['name'].replace(['Lady'], 'Mrs')
train_data['name'] = train_data['name'].replace(['Dr', 'Capt', 'Col', 'Countess', 'Don', 'Dona', 'Jonkheer', 'Major', 'Rev', 'Sir'], 'Rare')
train_data['name'] = train_data['name'].map({'Miss':0, 'Mrs':1, 'Master':2, 'Mr':3, 'Rare':4})
age_median = train_data.groupby('name')['Age'].median()

train_data['age_p'] = train_data['Age']
for i in range(0,5):
    train_data.loc[( (train_data.age_p.isnull()) & (train_data.name == i) ), 'age_p'] = age_median[i]
train_data['age_class'] = (train_data['age_p'] <= 15 )*1

age_class = np.array(train_data['age_class']).reshape(-1, 1)

# fare class
train_data.loc[np.isnan(train_data['Fare']), 'my_fare'] = 0
train_data.loc[(train_data['Fare'] > 55), 'my_fare'] = 1
train_data.loc[ (train_data['Fare'] <= 55), 'my_fare'] = 0
fare_class = np.array(train_data['my_fare']).reshape(-1, 1)

# feature = np.concatenate((sex, pclass, family_cut, age_class, fare_class), axis=1)
feature = np.concatenate((sex, family_cut, age_class, pclass), axis=1)
feature_train = feature[:len(train)]

survived = np.array(train_data[:len(train)].Survived).reshape(-1,1)

# initialize the model
model = RandomForestClassifier(oob_score=True)
model = model.fit(feature_train, survived.ravel())

feature_test = feature[len(train):]

predict = model.predict(feature_test).astype(int)

# output the result
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':predict})
output.to_csv('submission.csv', index=False)
print('submission save successfully')
print("%.4f" %(model.oob_score_))
