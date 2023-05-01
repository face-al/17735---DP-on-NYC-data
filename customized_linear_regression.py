from mechanism import *
from preprocessing import *
import numpy as np
from sklearn.linear_model import LinearRegression as sk_LinearRegression
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = preprocessing('example1.csv')

# Non-private baseline
regr = sk_LinearRegression()
regr.fit(X_train, y_train)
baseline = regr.score(X_test, y_test)
print("Non-private baseline R2 score: %.2f" % baseline)

epsilons = np.logspace(-3, 0, 100)
accuracy = []

for epsilon in epsilons:
    regr = sk_LinearRegression()
    # Specify mechanism to use
    temp = geometric(X_train, epsilon)
    regr.fit(temp, y_train)
    accuracy.append(regr.score(X_test, y_test))

plt.semilogx(epsilons, accuracy, label="Differentially private linear regression", zorder=10)
plt.semilogx(epsilons, baseline * np.ones_like(epsilons), dashes=[2,2], label="Non-private baseline", zorder=5)
plt.xlabel("epsilon")
plt.ylabel("r-squared score")
plt.ylim(-0.2, 1)
plt.xlim(epsilons[0], epsilons[-1])
plt.legend(loc=2)

plt.show()
