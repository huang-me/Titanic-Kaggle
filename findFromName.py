import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

train_data = pd.read_csv('train.csv')

#train['family'] = train['SibSp'] + train['Parch']

train_data['name'] = train_data['Name'].str.extract("([A-Za-z]+)\.", expand=False)
train_data['name'] = train_data['name'].replace(['Mlle', 'Ms', 'Mme'], 'Miss')
train_data['name'] = train_data['name'].replace(['Lady'], 'Mrs')
train_data['name'] = train_data['name'].replace(['Dr', 'Capt', 'Col', 'Countess', 'Don', 'Dona', 'Jonkheer', 'Major', 'Rev', 'Sir'], 'Rare')
train_data['name'] = train_data['name'].map({'Miss':0, 'Mrs':1, 'Master':2, 'Mr':3, 'Rare':4})
age_median = train_data.groupby('name')['Age'].median()

train_data['age_p'] = train_data['Age']
for i in range(0,5):
    train_data.loc[( (train_data.age_p.isnull()) & (train_data.name == i) ), 'age_p'] = age_median[i]

train_data['age_p'] = pd.qcut(train_data['age_p'], 5)

#sns.factorplot(x='Survived', y='Fare', data=train)
sns.factorplot(y='Survived', x='Parch', data=train_data)
plt.show()
