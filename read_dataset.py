import numpy as np
from scipy.io import loadmat


#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name):
    if name == 'flow_cylinder':
        return flow_cylinder()

    elif name == 'flow_isotropic':
        return flow_isotropic()  
    
    elif name == 'sst':
        return sst()   
    
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
    Xsmall = X[0:100, 65::, :]
    t, m, n = Xsmall.shape
    
    Xsmall = Xsmall.reshape(100, -1)

    Xsmall_test = X[100:151, 65::, :].reshape(51, -1)
    
    
    # Compute engergy
    #====================
    #_, s, _ = np.linalg.svd(Xsmall, 0)
    #cs = np.cumsum(s**2 / np.linalg.norm(Xsmall)**2)
    #print('Number of required modes:', np.where(cs > 0.999)[0][0])

    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return Xsmall, Xsmall_test, m, n





def flow_isotropic():
    X = np.load('data/flow_isotropic.npy')
    
    np.random.seed(1234567899)
    # Split into train and test set
    #X = X[:, 0:400, 0:400]
    X = X[:, 0:350, 0:350]
    t, m, n = X.shape
    
    indices = np.random.permutation(1000)
    indices = range(1000)
    training_idx, test_idx = indices[:800], indices[800:] 
    
    
    Xsmall = X[training_idx, :, :].reshape(800, -1)

    Xsmall_test = X[test_idx, :, :].reshape(200, -1)
    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return Xsmall, Xsmall_test, m, n



def sst():
    from netCDF4 import Dataset, date2index, num2date
    #from mpl_toolkits.basemap import Basemap, cm, interp
    #from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter
    #from matplotlib import ticker
    
    
    data_land_mask = 'data/lsmask.nc'
    data = 'data/sst.wkmean.1990-present.nc'
    
    nc = Dataset(data, mode='r')
    ncLAND = Dataset(data_land_mask, mode='r')
    
    
    # read sst.  Will automatically create a masked array using
    # missing_value variable attribute. 'squeeze out' singleton dimensions.
    sst = nc.variables['sst'][:].squeeze()
    sst_land = ncLAND.variables['mask'][:].squeeze()
    
    # read lats and lons (representing centers of grid boxes).
    lats_global = nc.variables['lat'][:]
    lons_global = nc.variables['lon'][:]
      
    nc.close()
    
    lons_global, lats_global = np.meshgrid(lons_global,lats_global)
    
    
    #******************************************************************************
    # Preprocess data
    #******************************************************************************
    t, m, n = sst.shape
    ssts = sst.reshape(t, m * n)
    ssts = ssts[:, np.where(sst_land.flatten() == 1)]
    ssts = ssts.reshape(t, -1)
    
    
    
    #******************************************************************************
    # Slect train data
    #******************************************************************************
    indices = np.random.permutation(1400)
    training_idx, test_idx = indices[:1100], indices[300:] 
    
    Xsmall = ssts[training_idx, :]
    Xsmall_test = ssts[test_idx, :]
 
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return Xsmall, Xsmall_test, lons_global, lats_global, sst_land, m, n



    
    