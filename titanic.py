import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from statistics import stdev

train_data = pd.read_csv('train.csv')

# set feature and survived data
# creating a dict file
sexdic = {'male': 1,'female': 2}
train_data.Sex = [sexdic[item] for item in train_data.Sex]
#print(train_data)

sex = np.array(train_data.Sex).reshape(-1, 1)
pclass = np.array(train_data.Pclass).reshape(-1, 1)
parch = np.array(train_data.Parch).reshape(-1, 1)
train_data['family_size'] = train_data.SibSp + train_data.Parch
train_data.loc[train_data['family_size'] == 0, 'family_cut'] = 0
train_data.loc[(train_data['family_size'] >= 1) & (train_data['family_size'] < 5), 'family_cut'] = 2
print(train_data.family_cut)
feature = np.concatenate((sex, pclass, parch), axis=1)
# print(feature)

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
feature = np.concatenate((sex, pclass, parch), axis=1)
print(feature.shape)
predict = model.predict(feature)

# output the result
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':predict})
output.to_csv('submission.csv', index=False)
print('submission save successfully')
