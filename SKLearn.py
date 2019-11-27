from sys import argv

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

fileName = argv[1]

dataSet = pd.read_csv(fileName, index_col=0)

y = dataSet['class']
spectro = dataSet['spectrometric_redshift'].values

for column in ('Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'galex_objid', 'sdss_objid', 'class', 'spectrometric_redshift', 'pred'):
	try: dataSet.drop(columns=column, inplace=True)
	except: pass

X = MinMaxScaler().fit_transform(dataSet)

X_train, X_test, y_train, y_test, spectro_train, spectro_test = train_test_split(dataSet, y, spectro, test_size=0.3, random_state=0)

knn = KNeighborsClassifier(5)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
spectro_pred = [0 if z <= 0.0033 else (1 if z >= 0.004 else 2) for z in spectro_test]

print(round(accuracy_score(y_test, y_pred) * 100, 2),
      round(accuracy_score(spectro_pred, y_pred) * 100, 2))
