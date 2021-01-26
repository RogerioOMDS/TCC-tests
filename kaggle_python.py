import pandas as pd 
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import timeit
import time

df = pd.read_csv("new_biddings.csv", dtype='float32');

X_train = df.iloc[0:300000, 0:87]
y_train = df.iloc[0:300000, 88]
X_test = df.iloc[300001:499999, 0:87]
y_test = df.iloc[300001:499999, 88]
logi = SGDClassifier(loss='log')

logi.fit(X_train, y_train)

# timer = timeit.timeit(lambda: logi.fit(X_train, y_train), number=1)

# print(timer)
y_pred = logi.predict(X_test)
print(accuracy_score(y_test, y_pred))