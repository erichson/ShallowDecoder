import numpy as np
import scipy as sci
from scipy import linalg
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.available
mpl.style.use('seaborn-paper') 
import matplotlib.patches as mpatches



def sensor_from_name(name, Xsmall, Xsmall_test, sensor_num=64, fix_sensor=10, random_seed=12345, m=None, n=None):
    if name == 'leverage_score':
        return _set_sensor_leverage_score(Xsmall, Xsmall_test, sensor_num=sensor_num, random_seed=random_seed)

    elif name == 'wall':
        return _set_sensor_wall(Xsmall, Xsmall_test, sensor_num=sensor_num, random_seed=random_seed)

    raise ValueError('sensor strategy {} not recognized'.format(name))


# Leverage score
def _set_sensor_leverage_score(Xsmall, Xsmall_test, sensor_num=64, random_seed=12345):
    np.random.seed(random_seed)
    n_snapshots_train, n_pix = Xsmall.shape
    n_snapshots_test, _ = Xsmall_test.shape


    U, s, Q = np.linalg.svd(scale(Xsmall, axis=1, with_mean=False, with_std=False, copy=True), full_matrices=False)
    lev_score = np.sum(Q**2, axis=0)
   
    pivots = np.random.choice(range(n_pix), sensor_num, replace=False, p = lev_score / (n_snapshots_train))

    sensors = Xsmall[:, pivots].reshape(n_snapshots_train, sensor_num)
    sensors_test = Xsmall_test[:, pivots].reshape(n_snapshots_test, sensor_num)

    return sensors, sensors_test, pivots



# Wall
def _set_sensor_wall(Xsmall, Xsmall_test, sensor_num=64, random_seed=12345): 
    np.random.seed(random_seed)
    n_snapshots_train, n_pix = Xsmall.shape
    n_snapshots_test, _ = Xsmall_test.shape
    m, n = 384, 199


    mask = np.zeros((m,n))
    theata = np.linspace(0, 2*np.pi, 300)
    x_cord = np.round(36 * np.cos(theata)) + 0
    y_cord = np.round(36 * np.sin(theata)) + 99
    cords = np.vstack((x_cord,y_cord)).T
    x_cord = np.unique(cords, axis=0)
    idx = x_cord[:,0] > 0
    x_cord = x_cord[idx,:]
    
    idx = np.random.choice(range(x_cord.shape[0]), sensor_num, False)
    x_cord = np.int64(x_cord[idx,:])
    
    mask[x_cord[:,0], x_cord[:,1]] = 1
    pivots = np.where(mask.reshape(-1) == 1)
    pivots = np.asarray(pivots).ravel()
 

    sensors = Xsmall[:, pivots].reshape(n_snapshots_train, sensor_num)
    sensors_test = Xsmall_test[:, pivots].reshape(n_snapshots_test, sensor_num)

    return sensors, sensors_test, pivots




def add_channels(X):
    if len(X.shape) == 2:
        return X.reshape(X.shape[0], 1, -1)
    
    elif len(X.shape) == 3:
        return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    
    else:
        return "dimenional error"
    
    
    
def error_summary(sensors, Xsmall, n_snapshots, model, Xmean, train_or_test='training'):   
        from torch.autograd import Variable, Function
        from torch.utils.data import DataLoader, Dataset
        
        # ===================compute relative error train========================
        dataloader_temp = iter(DataLoader(sensors, batch_size = n_snapshots))
        output_temp = model(Variable(dataloader_temp.next()).float().cuda())
        tt, _, mt = output_temp.shape
        
        redata = output_temp.cpu().data.numpy()
        error = np.linalg.norm(Xsmall.data.numpy() - redata) / np.linalg.norm(Xsmall.data.numpy())
        string_output = 'Relative deviation error ' + train_or_test + ' :'
        print(string_output, error)        
        
        
        # error for original data
        redata = redata + Xmean
        Xsmall_temp = Xsmall.data.numpy() + Xmean 
        error = np.linalg.norm(Xsmall_temp - redata) / np.linalg.norm(Xsmall_temp)
        string_output = 'Relative error ' + train_or_test + ' :'
        print(string_output, error)   
 
        return error
    
    
    
    
def final_summary(sensors, Xsmall, n_snapshots, model, Xmean, sensor_locations):
    from torch.autograd import Variable, Function
    from torch.utils.data import DataLoader, Dataset
    
    dataloader_temp = iter(DataLoader(sensors, batch_size = n_snapshots))
    output_temp = model(Variable(dataloader_temp.next()).float().cuda())
    tt, _, mt = output_temp.shape
    #        
    redata = output_temp.cpu().data.numpy().reshape(tt, mt)
    Xsmall_temp = Xsmall.data.numpy().reshape(tt, mt)    
    
    error_dev_DD = np.linalg.norm(redata - Xsmall_temp) / np.linalg.norm(Xsmall_temp)
    error_DD = np.linalg.norm((redata + Xmean) - (Xsmall_temp + Xmean)) / np.linalg.norm(Xsmall_temp + Xmean)
    
    print('Relative error using Shallow Decoder: ', error_DD)
    print('Relative error (deviation) using Shallow Decoder: ', error_dev_DD) 

    return  error_DD, error_dev_DD    
    
    
    
    
def summary_pod(Xsmall, Xsmall_test, sensors, sensors_test, Xmean, sensor_locations, alpha=0, case='naive_pod'):
    from sklearn import linear_model 
    
    #============================
    # Train data
    #============================
    
    # Get dimensions
    n_sensors = len(sensor_locations)       
    tt, _, mt = Xsmall.shape
    
    # Get Data from GPU to CPU
    Xsmall_temp = Xsmall.data.numpy().reshape(tt, mt) 
    sensors_temp = sensors.cpu().data.numpy().reshape(tt, n_sensors).T

    # Compute SVD
    u, s, v = np.linalg.svd(Xsmall_temp.T, 0)
    
    # Compute P as P = S pinv(X) U[:,J] 
    if case == 'naive_pod':
        P = u[sensor_locations, 0:n_sensors]
    elif case == 'plus_pod':
        P = sensors_temp.dot(np.linalg.pinv(Xsmall_temp.T)).dot(u[:, 0:n_sensors])


    # Create linear regression object
    if alpha == 0:
        reg = linear_model.LinearRegression(fit_intercept=False, normalize=False)
    
    else:
        reg = linear_model.Ridge(alpha=alpha, fit_intercept=False, normalize=False)

    
    
    # Train the model using the training sets
    reg.fit(P, sensors_temp)
    
    # Approximate data
    redata_linear = (u[:,0:n_sensors].dot(reg.coef_.T)).T
            
    # Compute error
    error_dev_POD_train = np.linalg.norm(redata_linear - Xsmall_temp) / np.linalg.norm(Xsmall_temp)
    error_POD_train = np.linalg.norm((redata_linear + Xmean) - (Xsmall_temp + Xmean)) / np.linalg.norm(Xsmall_temp + Xmean)   
 
    # Print
    print('Relative error using POD: ', error_POD_train)
    print('Relative error (deviation) using POD: ', error_dev_POD_train)
   

    #============================
    # Test data
    #============================
    
    # Get dimensions
    n_sensors = len(sensor_locations)       
    tt, _, mt = Xsmall_test.shape
    
    # Get Data from GPU to CPU
    Xsmall_temp = Xsmall_test.data.numpy().reshape(tt, mt) 
    sensors_temp = sensors_test.cpu().data.numpy().reshape(tt, n_sensors).T
    
    
    # Compute P as P = S pinv(X) U[:,J] 
    if case == 'naive_pod':
        P = u[sensor_locations, 0:n_sensors]
    elif case == 'plus_pod':
        P = sensors_temp.dot(np.linalg.pinv(Xsmall_temp.T)).dot(u[:, 0:n_sensors])
    elif case == 'plus2_pod':
        P = sensors_temp.dot(np.linalg.pinv(Xsmall_temp.T[sensor_locations,:])).dot(u[sensor_locations, 0:n_sensors])
    
    # Create linear regression object
    if alpha == 0:
        reg = linear_model.LinearRegression(fit_intercept=False, normalize=False)
    
    else:
        reg = linear_model.Ridge(alpha=alpha, fit_intercept=False, normalize=False)

    # Train the model using the training sets
    reg.fit(P, sensors_temp)
    
    # Approximate data
    redata_linear = (u[:,0:n_sensors].dot(reg.coef_.T)).T
            
    # Compute error
    error_dev_POD_test = np.linalg.norm(redata_linear - Xsmall_temp) / np.linalg.norm(Xsmall_temp)
    error_POD_test = np.linalg.norm((redata_linear + Xmean) - (Xsmall_temp + Xmean)) / np.linalg.norm(Xsmall_temp + Xmean)   

    # Print 
    print('Relative error using POD: ', error_POD_test)
    print('Relative error (deviation) using POD: ', error_dev_POD_test)
       

    return  error_POD_train, error_dev_POD_train, error_POD_test, error_dev_POD_test  
        
    


    
def plot_spectrum(sensors, Xsmall, sensors_test, Xsmall_test,
                  n_snapshots, n_snapshots_test,
                  model, Xmean, sensor_locations, plotting, train_or_test='training'):
    from torch.autograd import Variable, Function
    from torch.utils.data import DataLoader, Dataset
    
    dataloader_temp = iter(DataLoader(sensors, batch_size = n_snapshots))
    output_temp = model(Variable(dataloader_temp.next()).float().cuda())
    tt, _, mt = output_temp.shape
    #        
    redata = output_temp.cpu().data.numpy().reshape(tt, mt)
    Xsmall_temp = Xsmall.data.numpy().reshape(tt, mt)    
    
    u, s, v = np.linalg.svd(Xsmall_temp.T, 0)
    _, s2, _ = np.linalg.svd(redata.T, 0)
       
       
    plt.figure(facecolor="white",  figsize=(10.0, 6.5),  edgecolor='k')
    plt.loglog(np.arange(len(s))+1, s, label='True spectrum', 
               marker= "o", lw=6, c='k', markersize=15)
        
    plt.loglog(np.arange(len(s2))+1, s2, label='Shallow Decoder', 
               marker= "o", c='#de2d26', lw=4, markersize=6)
    
    
    # Linear reconstruction using PCA
    n_sensors = len(sensor_locations)
    linear_coef = (np.linalg.pinv(u[sensor_locations, 0:n_sensors])).dot(sensors.cpu().data.numpy().reshape(tt, n_sensors).T)    
    redata_linear = (u[:,0:n_sensors].dot(linear_coef)).T
        
    
    _, s3, _ = np.linalg.svd(redata_linear, 0)
     
    plt.loglog(np.arange(len(s3))+1, s3, '--', label='POD', marker= "s",
               c='#3182bd', lw=4, markersize=6)
    
    
    plt.tick_params(axis='x', labelsize=22) 
    plt.tick_params(axis='y', labelsize=22) 
    #plt.locator_params(axis='y', nbins=4)
    #plt.locator_params(axis='x', nbins=4)
    
    plt.ylabel('Magnitude', fontsize=28)
    plt.xlabel('Number of singular value', fontsize=28)
    plt.grid(False)
    #plt.yscale("log")
    #ax[0].set_ylim([0.01,1])
    plt.legend(fontsize=22)
    plt.tight_layout()  
    #plt.show()
    plt.savefig('results/flow_spectrum_train.png', dpi=300)          
    plt.close()  
        


    dataloader_temp = iter(DataLoader(sensors_test, batch_size = n_snapshots_test))
    output_temp = model(Variable(dataloader_temp.next()).float().cuda())
    tt, _, mt = output_temp.shape
    #        
    redata = output_temp.cpu().data.numpy().reshape(tt, mt)
    Xsmall_temp = Xsmall_test.data.numpy().reshape(tt, mt)    
    
    _, s, _ = np.linalg.svd(Xsmall_temp.T, 0)
    _, s2, _ = np.linalg.svd(redata.T, 0)
     
    
    plt.figure(facecolor="white",  figsize=(10.0, 6.5),  edgecolor='k')
    plt.loglog(np.arange(len(s))+1, s, 
               marker= "o",lw=6, c='k', label='True spectrum', markersize=15)
    #plt.scatter(np.log10(np.arange(len(s))+1), np.log10(s), label='True spectrum', marker= "D", s=80 )
        
    plt.loglog(np.arange(len(s))+1, s2, 
               marker= "o", label='Shallow Decoder', c='#de2d26', lw=4, markersize=6)
    #plt.scatter(np.log10(np.arange(len(s2))+1), np.log10(s2), label='Shallow Decoder', c='#de2d26', lw=3 )
    
    
    # Linear reconstruction using PCA
    n_sensors = len(sensor_locations)
    linear_coef = (np.linalg.pinv(u[sensor_locations, 0:n_sensors])).dot(sensors_test.cpu().data.numpy().reshape(tt, n_sensors).T)    
    redata_linear = (u[:,0:n_sensors].dot(linear_coef)).T
        
    
    _, s3, _ = np.linalg.svd(redata_linear, 0)
     
    plt.loglog(np.arange(len(s))+1, s3, '--',
               label='POD', marker= "s", c='#3182bd', lw=4, markersize=6)
    #plt.scatter(np.log10(np.arange(len(s3))+1), np.log10(s3))
    
    
    plt.tick_params(axis='x', labelsize=22) 
    plt.tick_params(axis='y', labelsize=22) 
    #plt.locator_params(axis='y', nbins=4)
    #plt.locator_params(axis='x', nbins=4)
    
    plt.ylabel('Magnitude', fontsize=28)
    plt.xlabel('Number of singular value', fontsize=28)
    plt.grid(False)
    #plt.yscale("log")
    #ax[0].set_ylim([0.01,1])
    plt.legend(fontsize=22)
    plt.tight_layout()  
    #plt.show()
    plt.savefig('results/flow_spectrum_test.png', dpi=300)          
    plt.close()
   
    
    
    


def plot_flow_cyliner(Xsmall, sensor_locations, m, n, Xmean):
    import cmocean

    x2 = np.arange(0, 384, 1)
    y2 = np.arange(0, 199, 1)
    mX, mY = np.meshgrid(x2, y2)

    img = Xsmall[0,:]  + Xmean
    img = img.reshape(384,199)

    minmax = np.max(np.abs(img)) * 0.65
    plt.figure(facecolor="white",  edgecolor='k', figsize=(7.9,4.7))
    plt.contourf(mX, mY, img.T, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
    plt.contour(mX, mY, img.T, 80, colors='black', alpha=0.5, vmin=-minmax, vmax=minmax)
    im = plt.imshow(img.T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
    
    wedge = mpatches.Wedge((0,99), 33, 270, 90, ec="#636363", color='#636363',lw = 5, zorder=200)
    im.axes.add_patch(p=wedge)    
    
    plt.tight_layout()
    plt.axis('off')
    
    ygrid = range(n)
    xgrid = range(m)
    yv, xv = np.meshgrid(ygrid, xgrid)
    
    x_sensors = xv.reshape(1, m*n)[:, sensor_locations]
    y_sensors = yv.reshape(1, m*n)[:, sensor_locations]

    plt.scatter(x_sensors, y_sensors, marker='.', color='#ff7f00', s=500, zorder=5)
    plt.title('Truth with sensor locations')      
    #plt.show()
    plt.savefig('results/flow_truth_with_sensors.png', dpi=300)          
    plt.close()


    img = Xsmall[0,:]  + Xmean
    img = img.reshape(384,199)

    plt.figure(facecolor="white",  edgecolor='k', figsize=(7.9,4.7))
    plt.contourf(mX, mY, img.T, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
    plt.contour(mX, mY, img.T, 80, colors='black', alpha=0.5, vmin=-minmax, vmax=minmax)
    im = plt.imshow(img.T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)    
    wedge = mpatches.Wedge((0,99), 33, 270, 90, ec="#636363", color='#636363',lw = 5, zorder=200)
    im.axes.add_patch(p=wedge)    
        
    plt.title('Truth')      
    plt.tight_layout()
    plt.axis('off')
    #plt.show()
    plt.savefig('results/flow_truth.png', dpi=300)          
    plt.close()






def plot_flow_cyliner_2(img_epoch, m, n, epoch, Xmean):
    import cmocean

    x2 = np.arange(0, 384, 1)
    y2 = np.arange(0, 199, 1)
    mX, mY = np.meshgrid(x2, y2)

    
    img_epoch += Xmean.reshape(m,n)

    minmax = np.max(np.abs(img_epoch)) * 0.65
    plt.figure(facecolor="white",  edgecolor='k', figsize=(7.9,4.7))
    im = plt.imshow(img_epoch.T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
    plt.contourf(mX, mY, img_epoch.T, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
    
    
    wedge = mpatches.Wedge((0,99), 33, 270, 90, ec="#636363", color='#636363',lw = 5)
    im.axes.add_patch(p=wedge)    
        
    #plt.title('Epoch number ' + str(epoch), fontsize = 16 )
    plt.axis('off')
    plt.title('Reconstructed flow filed using the shallow decoder')          
    plt.tight_layout()
    #plt.show()
    plt.savefig('results/reconstruction_via_shallow_decoder.png', dpi=300)          
    plt.close()



def plot_flow_cyliner_pod(Xsmall, sensors, Xmean, sensor_locations, m, n):
    import cmocean

    x2 = np.arange(0, 384, 1)
    y2 = np.arange(0, 199, 1)
    mX, mY = np.meshgrid(x2, y2)
    
    tt, _, mt = Xsmall.shape
    Xsmall_temp = Xsmall.data.numpy().reshape(tt, mt)    
    u, s, v = np.linalg.svd(Xsmall_temp.T, 0)
       
    
    # Linear reconstruction using PCA - Train
    n_sensors = len(sensor_locations)
    linear_coef = (np.linalg.pinv(u[sensor_locations, 0:n_sensors])).dot(sensors.cpu().data.numpy().reshape(tt, n_sensors).T)    
    redata_linear = (u[:,0:n_sensors].dot(linear_coef)).T

    img_epoch = redata_linear[0,:].reshape(m,n)
    img_epoch += Xmean.reshape(m,n)
     
    minmax = np.max(np.abs(img_epoch)) * 0.65
    plt.figure(facecolor="white",  edgecolor='k', figsize=(7.9,4.7))
    im = plt.imshow(img_epoch.T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
    plt.contourf(mX, mY, img_epoch.T, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
    
    
    wedge = mpatches.Wedge((0,99), 33, 270, 90, ec="#636363", color='#636363',lw = 5)
    im.axes.add_patch(p=wedge)    
        
    #plt.title('Epoch number ' + str(epoch), fontsize = 16 )
    plt.axis('off')
    plt.title('Reconstructed flow filed using POD')      
    plt.tight_layout()
    #plt.show()
    plt.savefig('results/reconstruction_via_pod.png', dpi=300)          
    plt.close() 

        

def plot_flow_cyliner_regularized_pod(Xsmall, sensors, Xmean, sensor_locations, m, n, alpha):
    import cmocean
    from sklearn import linear_model

    x2 = np.arange(0, 384, 1)
    y2 = np.arange(0, 199, 1)
    mX, mY = np.meshgrid(x2, y2)
    
    tt, _, mt = Xsmall.shape
    Xsmall_temp = Xsmall.data.numpy().reshape(tt, mt)    
    u, s, v = np.linalg.svd(Xsmall_temp.T, 0)
       
    
    # Linear reconstruction using PCA - Train
    n_sensors = len(sensor_locations)
    reg = linear_model.Ridge(alpha=alpha, fit_intercept=False, normalize=False)
    reg.fit(u[sensor_locations, 0:n_sensors], sensors.cpu().data.numpy().reshape(tt, n_sensors).T)
    redata_linear = (u[:,0:n_sensors].dot(reg.coef_.T)).T

    img_epoch = redata_linear[0,:].reshape(m,n)
    img_epoch += Xmean.reshape(m,n)
     

    minmax = np.max(np.abs(img_epoch)) * 0.65
    plt.figure(facecolor="white",  edgecolor='k', figsize=(7.9,4.7))
    im = plt.imshow(img_epoch.T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)
    plt.contourf(mX, mY, img_epoch.T, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
    wedge = mpatches.Wedge((0,99), 33, 270, 90, ec="#636363", color='#636363',lw = 5)
    im.axes.add_patch(p=wedge)    
        
    #plt.title('Epoch number ' + str(epoch), fontsize = 16 )
    plt.axis('off')
    plt.title('Reconstructed flow filed using POD Plus')          
    plt.tight_layout()
    #plt.show()
    plt.savefig('results/reconstruction_via_pod_plus.png', dpi=300)          
    plt.close()   
