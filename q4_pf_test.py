import numpy as np
from preprocessing.polynomial_features import PolynomialFeatures



X = np.array([1,2])
X=X.reshape((1,2))
poly = PolynomialFeatures(2,include_bias=True)
X_=poly.transform(X)
print(X_)