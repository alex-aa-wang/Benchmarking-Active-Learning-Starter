import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def plot_data(data, XY, plot_name:str,three='n'):
    Xi = XY[:,0]
    Xj = XY[:,1]
    plt.title(plot_name)
    plt.scatter(Xi,Xj,20, c = flatten(data))
    plt.title(plot_name)
    if three !='n':
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(Xi,Xj,flatten(data),cmap=cm.coolwarm)
        fig.colorbar(surf, shrink=0., aspect = 5)
        plt.show()

def flatten(a):
    return np.ndarray.flatten(a)

def calc_min_regret(Y_samples, Y_max):
    min_regret = Y_max - np.maximum.accumulate(Y_samples)
    return min_regret

def calc_rmse(Y_estimates, Y_grid):
    residuals_squared = np.square( Y_estimates - np.tile(Y_grid,(1,Y_estimates.shape[1])) )
    rmse = np.sqrt( np.mean( residuals_squared, axis = 0) )
    return rmse
