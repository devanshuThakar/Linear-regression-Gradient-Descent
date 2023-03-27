import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *
from sklearn.metrics import mean_absolute_error,mean_squared_error 

N = 30
P = 3
C=2#Collinear rows
X = np.random.randn(N, P)
#Generating collinear columns
for i in range(C):
    size=np.random.randint(1,P)
    cols=np.random.randint(0,P,size)#Column indices
    coeffs=np.random.randint(1,10,size)#respective coefficients for making new column,
    new_col=np.zeros(N)
    for j in range(size):
        new_col=new_col+coeffs[j]*X[:,j]
    X=np.c_[X,new_col]
X=pd.DataFrame(X)
y = pd.Series(np.random.randn(N))



for fit_intercept in [True, False]:
    if fit_intercept==True:
        print("\nWith Fit intercept")
    else:
        print("\nWithout fit intercept")
    
    LR = LinearRegression(fit_intercept=fit_intercept)

    LR.fit_vectorised(X, y,batch_size=5) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X,isvectorized=True)
    
    print('Vectorized \nMSE: ', mean_squared_error(y,y_hat))
    print('MAE: ', mean_absolute_error(y,y_hat))

  

