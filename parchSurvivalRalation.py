from statistics import stdev
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

train_data = pd.read_csv('train.csv')

parch = train_data.Parch

survive = train_data.Survived
train_data['family'] = train_data.Parch + train_data.SibSp

sns.countplot(x='Parch', hue='Survived', data=train_data)
#pd.crosstab(train_data.Parch, train_data.Survived).plot(kind='bar', stacked=True)
sns.factorplot(x='family', y='Survived', data=train_data)
plt.show()
