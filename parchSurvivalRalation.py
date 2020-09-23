from statistics import stdev
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

train_data = pd.read_csv('train.csv')

parch = train_data.Parch

survive = train_data.Survived
train_data['age_class4'] = pd.qcut(train_data.Age, 4)
train_data['age_class5'] = pd.qcut(train_data.Age, 5)
train_data['age_class6'] = pd.qcut(train_data.Age, 6)

#sns.countplot(x='age_class', hue='Survived', data=train_data)
#pd.crosstab(train_data.Parch, train_data.Survived).plot(kind='bar', stacked=True)

#train_data.loc[train_data.Age <]

sns.factorplot(x='age_class4', y='Survived', data=train_data)
sns.factorplot(x='age_class5', y='Survived', data=train_data)
sns.factorplot(x='age_class6', y='Survived', data=train_data)
plt.show()
