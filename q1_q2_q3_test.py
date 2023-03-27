
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *
from sklearn.metrics import mean_absolute_error,mean_squared_error 

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))


for fit_intercept in [True, False]:
    if fit_intercept==True:
        print("\nWith Fit intercept")
    else:
        print("\nWithout fit intercept")
    
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_non_vectorised(X, y,batch_size=10) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X,isvectorized=False)
    
    print('Non Vectorized \nMSE: ', mean_squared_error(y,y_hat))
    print('MAE: ', mean_absolute_error(y,y_hat))
    
    LR.fit_vectorised(X, y,batch_size=10) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X,isvectorized=True)
    
    print('Vectorized \nMSE: ', mean_squared_error(y,y_hat))
    print('MAE: ', mean_absolute_error(y,y_hat))


    LR.fit_autograd(X, y,batch_size=10) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X,isvectorized=True)
    
    print('Autograd \nMSE: ', mean_squared_error(y,y_hat))
    print('MAE: ', mean_absolute_error(y,y_hat))
