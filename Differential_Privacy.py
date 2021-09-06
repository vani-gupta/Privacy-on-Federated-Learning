#dependancies
!pip install diffprivlib

import diffprivlib.models as dp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv", header=None)
df.columns=['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
       'marital-status', 'occupation', 'relationship', 'race', 'gender',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country','income']
num_features=['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_features=[ 'workclass',  'education', 'marital-status', 'occupation', 'relationship', 'race',
              'gender', 'native-country']

#applying normalization to numerical attributes
for col in num_features:
    df[col] = df[col].apply(lambda x: np.log1p(x))
    
#label encoding categorical features
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    
#label encoding target variable
df['income'] = le.fit_transform(df['income'])

X = df.drop('income' , axis = 1).copy()
y = df['income']
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# calculating accuracy for epsilon=1
dp_clf = dp.GaussianNB()
dp_clf.fit(X_train, y_train)

print("Differentially private test accuracy (epsilon=%.2f): %.2f%%" % 
      (dp_clf.epsilon, dp_clf.score(X_test, y_test) * 100))

# calculating and plotting accuracy for epsilon[]= np.logspace(-2, 2, 50)

epsilons = np.logspace(-2, 2, 50)
accuracy = list()

for epsilon in epsilons:
  dp_clf2 = dp.GaussianNB(epsilon=epsilon)
  dp_clf2.fit(X_train, y_train)
  accuracy.append(dp_clf2.score(X_test, y_test))

plt.semilogx(epsilons, accuracy)
plt.title("Differentially private Naive Bayes accuracy")
plt.xlabel("epsilon")
plt.ylabel("Accuracy")
plt.show()
