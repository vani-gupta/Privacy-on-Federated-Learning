#dependancies
!pip install diffprivlib

import diffprivlib.models as dp
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

X_train = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                        usecols=(0, 4, 10, 11, 12), delimiter=", ")

y_train = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                        usecols=14, dtype=str, delimiter=", ")

X_test = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                        usecols=(0, 4, 10, 11, 12), delimiter=", ", skiprows=1)

y_test = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                        usecols=14, dtype=str, delimiter=", ", skiprows=1)

# Must trim trailing period "." from label
y_test = np.array([a[:-1] for a in y_test])

# calculating accuracy for epsilon=1
dp_clf = dp.GaussianNB()
dp_clf.fit(X_train, y_train)

print("Differentially private test accuracy (epsilon=%.2f): %.2f%%" % 
      (dp_clf.epsilon, dp_clf.score(X_test, y_test) * 100))


# calculating and plotting accuracy for epsilon[]= np.logspace(-2, 2, 50)

epsilons = np.logspace(-2, 2, 50)
bounds = ([17, 1, 0, 0, 1], [100, 16, 100000, 4500, 100])
accuracy = list()



for epsilon in epsilons:
  dp_clf2 = dp.GaussianNB(epsilon=epsilon, bounds=bounds)
  dp_clf2.fit(X_train, y_train)
  accuracy.append(dp_clf2.score(X_test, y_test))


plt.semilogx(epsilons, accuracy)
plt.title("Differentially private Naive Bayes accuracy")
plt.xlabel("epsilon")
plt.ylabel("Accuracy")
plt.show()

