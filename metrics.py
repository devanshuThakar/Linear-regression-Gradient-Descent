import numpy as np

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    total=y.size
    count=0
    for i in range(total):
        if y_hat[i]==y[i]:
            count=count+1

    return count/total

def precision(y_hat, y, cls):
    """
    Function to calculate the precision
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    total=0
    count=0
    for i in range(y_hat.size):
        if y_hat[i]==cls:
            total=total+1
            if y_hat[i] == y[i]:
                count=count+1
    if total==0:
        return 1
    return count/total


def recall(y_hat, y, cls):
    """
    Function to calculate the recall
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    total=0
    count=0
    for i in range(y_hat.size):
        if y[i]==cls:
            total=total+1
            if y_hat[i] == y[i]:
                count=count+1
    if total==0:
        return 1

    return count/total


def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """

    rmse=0
    Size=y.size
    for i in range(Size):
        rmse=((y_hat[i]-y[i])**2)

    rmse=np.sqrt(rmse/Size)
    
    return rmse

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
     """
    mae=0
    Size=y.size
    for i in range(Size):
        mae=(abs(y_hat[i]-y[i]))/Size

    return mae
