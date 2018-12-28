import numpy as np
from scipy.io import loadmat


#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name):
    if name == 'flow_cylinder':
        return flow_cylinder() 
    
    else:
        raise ValueError('dataset {} not recognized'.format(name))


def rescale(Xsmall, Xsmall_test):
    #******************************************************************************
    # Rescale data
    #******************************************************************************
    Xmin = Xsmall.min()
    Xmax = Xsmall.max()
    
    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin)) 
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin)) 

    return Xsmall, Xsmall_test



def flow_cylinder():
    X = np.load('data/flow_cylinder.npy')
    print(X.shape)
    
    
    # Split into train and test set
    Xsmall = X[0:100, :, :]
    t, m, n = Xsmall.shape
    
    Xsmall = Xsmall.reshape(100, -1)

    Xsmall_test = X[100:151, :, :].reshape(51, -1)

    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return Xsmall, Xsmall_test, m, n
