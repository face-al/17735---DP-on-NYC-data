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
baseline = regr.predict(X_test)

epsilon = 100

regr = LinearRegression(epsilon=epsilon, bounds_X=(X_train.min(), X_train.max()), bounds_y=(y_train.min(), y_train.max()))
regr.fit(X_train, y_train)
predict = regr.predict(X_test)
print(predict)

index = range(len(X_test))
plt.plot(index, predict, label="Differentially private with epsilon ***", zorder=10)
plt.plot(index, baseline, label="baseline", zorder=10)
plt.xlabel("Location Data")
plt.ylabel("Target")
plt.ylim(-1000, 4000)
plt.xlim(0, len(X_test))
plt.legend(loc=2)

plt.show()
