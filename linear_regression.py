import numpy as np
from sklearn.linear_model import LinearRegression as sk_LinearRegression
from sklearn import metrics
from diffprivlib.models import LinearRegression
from preprocessing import *
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = preprocessing('example1.csv')

# Non-private baseline
regr = sk_LinearRegression()
regr.fit(X_train, y_train)
pred = regr.predict(X_test)
baseline = metrics.mean_absolute_error(y_test, pred)
print("Non-private baseline mean_absolute_error: %.2f" % baseline)

epsilons = np.logspace(0.5, 2, 100)
accuracy = []

for epsilon in epsilons:
    regr = LinearRegression(epsilon=epsilon, bounds_X=(X_train.min(), X_train.max()), bounds_y=(y_train.min(), y_train.max()))
    regr.fit(X_train, y_train)
    pred = regr.predict(X_test)
    score = metrics.mean_absolute_error(y_test, pred)

    accuracy.append(score)

plt.loglog(epsilons, accuracy, label="Differentially private linear regression", zorder=10)
plt.loglog(epsilons, baseline * np.ones_like(epsilons), dashes=[2,2], label="Non-private baseline", zorder=5)
plt.xlabel("epsilon")
plt.ylabel("Mean Absolute Error")
plt.ylim(100, 10 ** 7)
plt.xlim(epsilons[0], epsilons[-1])
plt.legend(loc=2) 

plt.show()

