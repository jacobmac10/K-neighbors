#siempre al inicio del codigo importamos las galerias que vayamos a usar. 
#----------librerias---------
from sklearn import datasets 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor  #ctrl + espacio y te recomienda las librerias.
#----------librerias----------



X, y = datasets.load_diabetes(return_X_y=True)


#regression 

#print(X)



#me interesan todos los registros en y y me interesa solo la 3er columna 
#el newaxis agrega una dimension extra.
X = X[:, np.newaxis, 2]


#shape --> dimensiones del arreglo. 

regr = KNeighborsRegressor()

#valores de x que van a servir para entrenar
X_train = X[:-20, :] #todos los registros menos los ultimos 20
X_test = X[-20:, :] #los ultimos 20 registros

#valores que van a servir para entrenar
y_train = y[:-20] #todos los registros menos los ultimos 20
y_test = y[-20:] # los ultimos 20 

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

print("mean squared error: %2f" %mean_squared_error(y_test, y_pred))
print("coefiicient of determination: %2f" %r2_score(y_test, y_pred))

plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred, c='orange')
plt.show()

