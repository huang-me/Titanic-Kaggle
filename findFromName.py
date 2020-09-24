import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv('train.csv')

train['family'] = train['SibSp'] + train['Parch']

#sns.factorplot(x='Survived', y='Fare', data=train)
sns.factorplot(y='Survived', x='family', data=train)
plt.show()
