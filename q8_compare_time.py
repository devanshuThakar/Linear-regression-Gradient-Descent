import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time

np.random.seed(42)

N = 3000
P = 1000
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))


for fit_intercept in [True, False]:
    if fit_intercept==True:
        print("\nWith Fit intercept")
    else:
        print("\nWithout fit intercept")
    
    LR = LinearRegression(fit_intercept=fit_intercept)
    
    start=time.process_time()
    LR.fit_vectorised(X, y,n_iter=25,batch_size=1) # here you can use fit_non_vectorised / fit_autograd methods
    end=time.process_time()

    grad_descent_time=end-start

    start=time.process_time()
    LR.fit_normal(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    end=time.process_time()

    normal_time=end-start

    print("Gradient Descent time: ",grad_descent_time)
    print("Normal method time: ",normal_time)
    
    
