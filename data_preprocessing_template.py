#importing dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
#takecare of missingdata
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3]=missingvalues.transform(X[:, 1:3])


#Enconding categrical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder().fit_transform(Y)
#spliltting dataset 
from sklearn.model_selection import train_test_split 
X_train , X_test ,Y_train , Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)






                