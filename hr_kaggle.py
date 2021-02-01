import pandas as pd 
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import timeit
import time
import statistics

df = pd.read_csv("HR_data.csv");
df

X_train = df.iloc[0:11000, 1:9]
X_test = df.iloc[11001:14999, 1:9]

y_train = df.iloc[0:11000, 0]
y_test = df.iloc[11001:14999, 0]

logi = SGDClassifier(loss='log')

logi.fit(X_train, y_train)

timer = timeit.repeat(lambda: logi.fit(X_train, y_train), number=1, repeat=1)
print('timer = ', timer)
# print(statistics.mean(timer))

y_pred = logi.predict(X_test)

print(classification_report(y_test, y_pred))

CM = confusion_matrix(y_test, y_pred)
print(CM)

tn = CM[0,0]
tp = CM[1,1]
fp = CM[0,1]
fn = CM[1,0]

precisionScore = tp / (tp + fp)

recallScore = tp / (tp + fn)

accuracyScore = (tp + tn) / (fp + fn + tp + tn)

f1Score = (2 * precisionScore * recallScore ) / (precisionScore + recallScore)

print(f'precisionScore = {precisionScore} | recallScore = {recallScore} |  accuracyScore = {accuracyScore} | f1Score = {f1Score}' )