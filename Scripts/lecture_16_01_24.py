import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Data\50_Startups.csv")

X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , 4].values

# Modificamos datos categóricos
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = make_column_transformer((OneHotEncoder(), [3]),
                                         remainder="passthrough")
X = onehotencoder.fit_transform(X)

# Eliminamos una variable ficticia
X = X[:, 1:]

# Dividir el dataset en test y train
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, 
                                                    random_state=0)

# Ajustar el model de regresion lineal multiple con los datos train
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en datos test
y_pred = regression.predict(X_test)

# Construir el modelo óptimo de RLM usando la Eliminación hacia atras
import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values = X, axis = 1)
SL = 0.05

# Resultados de los modelos
# MODELO 1
X_opt = X[: , [0,1,2,3,4,5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt.tolist()).fit()
regression_OLS.summary()

# MODELO 2
X_opt = X[: , [0,1,3,4,5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt.tolist()).fit()
regression_OLS.summary()

# MODELO 3
X_opt = X[: , [0,3,4,5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt.tolist()).fit()
regression_OLS.summary()

# MODELO 4
X_opt = X[: , [0,3]]
regression_OLS = sm.OLS(endog=y, exog=X_opt.tolist()).fit()
regression_OLS.summary()