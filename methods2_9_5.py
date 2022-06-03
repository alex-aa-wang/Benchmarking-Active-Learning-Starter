# Support for math
import numpy as np

# Plotting tools
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# we use the following for plotting figures in jupyter
import scipy
from google.colab import files

import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

# GPy: Gaussian processes library

import contextlib
import sys

import gpflow

from scipy.special import ndtr
from scipy.stats import norm

import math
import helpers as h

import plotly.express as px
import plotly.graph_objects as go

FIGSIZE = (8, 5)
# define a list of methods and a list of colors for future reference
methods = ["exploit","explore","ucb","samy","thompson","random","EI","cameo"]#"AEI","cameo"
colors = ['b','g','r','c','m','y','darkorange','lightskyblue','k']

# takes a number of measurements, reps, iterations and a list of data e.g.
# [mag,kerr], their associated Y values, along with the names for plots    
def total_optimize(num_meas,num_reps,num_iter,data_list,data_XY,data_names):
    query = dict()
    for curr_data in range(0,len(data_list)):
    	for curr_method in range(0,len(methods)):
            # assign all the results to a dictionary so they can be saved as a .pkl and accessed easily
            query[(data_names[curr_data],methods[curr_method])] = optimize_g(num_meas,num_reps,num_iter,data_list[curr_data],data_XY[curr_data], methods[curr_method], colors[curr_method], data_names[curr_data],odd = "y")
    
    return query
        	
# performs optimization for one data set with all 7ish methods
def optimize_deep_data(num_meas,num_reps, num_iter, data,XY , data_name):
    query_data = []
    for i in range(0, len(methods)):
        query_data.append(optimize_g(num_meas, num_reps, num_iter, data, XY, methods[i],i,data_name,odd="y"))
    return query_data

# performs optimization for multiple datasets with one specific method, e.g. exploitation
def optimize_more_data(num_meas, num_reps, num_iter,method_num, data , data_name_list):
    num_data = len(data)
    query = np.zeros(num_data)
    for i in range(0, num_data):
        query[i] = optimize_g(num_meas, num_reps, num_iter, data[i][0], data[i][1], methods[method_num],i,data_name_list[i])
    return query
    
# performs a repetition of optimizations on one dataset with one method, backbone of the three previous methods
def optimize_g(num_meas, num_reps, num_iter, Y_grid, X_grid, selection_type:str,color:int,name,odd = "n"):
    # adjust for total
    num_iter = num_iter-num_meas# +1 for pretty graphs at sizes less than max
    
    # get max for minregret	
    true_max = np.argmax(Y_grid)
    
    N = len(X_grid)
    rmse_ = np.empty((num_reps,num_iter))
    sampled_ = []

    #performs optimization for the number of reps specified
    for j in range(0, num_reps):

        # Get and set random sample locations
        already_sampled = np.random.permutation(N)[np.arange(0,num_meas)]
        # XY at randomly generated locations
        #X_samples = X_grid[already_sampled] 

        #Y_samples = np.array([Y_grid[x1] for x1 in already_sampled]) 
        # Use this matrix to store the GP mean at every iteration.
        #Y_estimates = np.full((len(X_grid), num_iter),np.nan)
        # use the specific method to get the next sample
        # different for random to avoid using GP
        if selection_type == "random":
            already_sampled = rando(X_grid, Y_grid, already_sampled, num_iter, num_meas,selection_type) 
        else:
            already_sampled = gp(X_grid, Y_grid, already_sampled, num_iter, num_meas, selection_type)
            #rmse_[j,:] = h.calc_rmse(Y_estimates, Y_grid)
        sampled_.append(already_sampled.tolist())
        
    # returns indices of sampled
    return sampled_

# performs the actual individual optimization for each dataset, with x iterations
def gp(X_grid, Y_grid, already_sampled,num_iter,num_meas,scheme:str):
    for i in range(0,num_iter):
        if already_sampled[-1] == np.argmax(Y_grid):
            #print(already_sampled[-1])
            already_sampled = np.append(already_sampled,already_sampled[-1])
        else:
        	# call predict to separate GP usage
        	mean,Cov,var = predict(X_grid[already_sampled],Y_grid[already_sampled],X_grid,scheme)
        	#mean,var = predict(X_grid[already_sampled],Y_grid[already_sampled],X_grid)
        	# called scheme becomes the function with the same name of the method
        	called_scheme = globals()[scheme]

        	# uses the new function to call the method to get the next index
        	next_sample_index = called_scheme(mean,Cov,var,already_sampled,X_grid,i+1,np.max(Y_grid[already_sampled]))#,model)

        	# add sampled data to sampled array
        	already_sampled = np.append(already_sampled,next_sample_index)
        	#Y_estimates[:,i] = mean.ravel()
        
    return already_sampled
        
# use one method to do GP and predict
def predict(X_samples,Y_samples,X_grid,scheme):
    
    k = gpflow.kernels.SquaredExponential(lengthscales = [.1, .1])

    # Use GP regression to fit the data
    m = gpflow.models.GPR(data=(X_samples,Y_samples), kernel=k, mean_function=None) # set GPR model

    m.likelihood.variance.assign(0.01) # prior for variance
    opt = tf.optimizers.Adam()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables)
    # get the GP
    if not scheme=='thompson':
        prop_mean, prop_var = m.predict_f(X_grid) # using prop for N random points, predict the prop for the rest.
        Cov = tf.convert_to_tensor(np.array([]))
    else:
        prop_mean, [Cov] = m.predict_f(X_grid, full_cov=True)
        prop_var = tf.convert_to_tensor(np.array([]))
    
    #temp is equal to prop_mean
    #return prop_mean.numpy(),prop_var.numpy()# Cov.numpy()[0], prop_var.numpy()#,model
    return prop_mean.numpy(), Cov.numpy(), prop_var.numpy()

# adds a sample to a list
def add_sample(X_samples,X_grid,Y_samples,Y_grid,index):
    X_samples = np.vstack((X_samples,X_grid[index,:]))
    Y_samples = np.vstack((Y_samples,Y_grid[index,0]))
    return X_samples,Y_samples

# prevents double counting of samples
def mask(target, source):
    target_mask = np.zeros(target.size, dtype=bool)
    target_mask[source] = True
    return np.ma.array(target,mask = target_mask)

# returns a sample based on pure exploitation
def exploit(mean,c,v,sampled,x,i,m):
    #print(v)
    #print(type(sampled))
    masked = mask(mean,sampled)
    return np.argmax(masked)

#returns a sample based on pure exploration
def explore(mean,c,var,sampled,x,iteration,m):#,m):
    masked = mask(var,sampled)
    return np.argmax(masked)

# returns a sample based on upper confidence bound
def ucb(mean,c,var,sampled,X_grid,iteration,m):#,m):
    BO_number_of_iterations = iteration
    BO_lambda = .1
    size = np.prod(X_grid[:,0].shape)
    BO_beta = 2 * math.log(size * math.pow(BO_number_of_iterations,2) * math.pow(np.pi,2) / (6 * BO_lambda) )
    #print(var)
    
    BO_alpha = mean + math.sqrt(BO_beta) * np.sqrt(var)

    masked = mask(BO_alpha, sampled)
    return np.argmax(masked)

# returns a BO sample based on the kandesamy method
def samy(mean,c,var,sampled,x,iteration,m):
    BO_beta = math.sqrt(.2*2*math.log(2*iteration))
    BO_alpha = mean + math.sqrt(BO_beta)*np.sqrt(var)

    masked = mask(BO_alpha,sampled)
    return np.argmax(masked)

# returns a sample based on thompson method 
def thompson(mean,Cov,var,sampled,X_grid,i,m):
    Z = np.random.multivariate_normal(mean.ravel(), Cov, 1).T
    masked = mask(Z,sampled)
    return np.argmax(masked)

# returns based on Expectation improvement algorithm
def EI(mean,C,var,sampled,X_grid,i, max_val):
    tradeoff = 0.01
    std = np.sqrt(var)
    z = (mean - max_val - tradeoff) / std
    masked = mask(mean,sampled)
    return np.argmax((masked - max_val - tradeoff)*ndtr(z) + std*norm.pdf(z))
    
    
# gets one repetition of random samples (x iterations)
def rando(X_grid, Y_grid, already_sampled, num_iter,num_meas,scheme:str):
    for i in range(0,num_iter):
        called_scheme = globals()[scheme]
        next_sample_index = called_scheme(already_sampled,X_grid)
        already_sampled = np.append(already_sampled,next_sample_index)
    return already_sampled

# gets one repetition of random samples (x iterations)
def random(sampled,X_grid):
    #print(X_grid.shape[0])
    #print(sampled.size)
    index = np.random.randint(0,X_grid.shape[0]-sampled.size)
    
    sorted_sampled = np.sort(sampled)
    # accounts for the fact that several indices have been taken out already
    for i in sorted_sampled:
        if i<=index:
            index+=1
    return index

# will take a dictionary and process it using the following methods
# - process_data; returns a list of minregrets for each function
# - plot_all; plots data
def eat_indices(data_indices,data_org,data_names):
    minregrets = dict()
    for data_num in range(0,len(data_indices)):
        curr_data = data_org[data_num]
        curr_name = data_names[data_num]
        curr_ind = data_indices[data_num]
        minregret_ = process_data(curr_ind,curr_data, curr_name, np.amax(curr_data),np.amin(curr_data))
        minregrets[curr_name] = minregret_
        plot_all(minregret_, methods, curr_name,2*data_num)
    return minregrets

#shape of data should be an array of 7 arrays, each being a different method
def process_data(data_ind,data_org,data_name,max_,min_):
    minregret_ = dict()
    temp = np.array(data_org)
    for method in range(0,len(methods)):
        c_method = []
        queries_ = data_ind[data_name,methods[method]]
        #print(data_ind[data_name,methods[method]])
        for i in range(0, len(queries_)):
            c_method.append(h.calc_min_regret(temp[queries_[i]], max_)/(max_-min_))
        minregret_[methods[method]] = c_method
    
    #minregret_ = np.array(minregret_)
    #minregret_[method] = minregret_/(max_-min_)
        
    return minregret_


    

    
# plots all of the min regrets passed through with different figure numbers
def plot_all(minregret_total, label_names,plot_type:str,fig_num):
    for name in range(0,len(label_names)):
        # first name param is for color, second name is for figure number
        plot_mm(minregret_total[label_names[name]],label_names[name],plot_type,name,fig_num)
        

# plots both the mean and the median of the min regrets
# uses both plot_mean and plot_mm
def plot_mm(minregret_,label_name,plot_type:str,color,fig_num):

    plot_mean(minregret_,label_name,plot_type,color,fig_num)
    plot_med(minregret_,label_name,plot_type,color,fig_num+1)

# plots the median of the min regrets
def plot_med(minregret_,label_name,plot_type:str,color,fig_num):
    plt.figure(fig_num,figsize = FIGSIZE)
    plt.plot(np.median(minregret_,axis = 0),c = colors[color],label = label_name)
    plt.title('Median of min regret over runs: '+ plot_type)
    plt.legend(loc ='best')

# plots the mean of the min regrets
def plot_mean(minregret_,label_name,plot_type:str,color,fig_num):
    plt.figure(fig_num,figsize = FIGSIZE)
    #print(minregret_)
    plt.plot(np.mean(minregret_,axis = 0),c = colors[color],label = label_name)
    plt.title('Avg of min regret over runs: '+ plot_type)
    plt.legend(loc ='best')

# visualises the data
def plot_data(data, XY, plot_name:str,three='n'):
    h.plot_data(data,XY,plot_name,three)

# removes layers from an array
def flatten(a):
    return np.ndarray.flatten(a)


# the following code supresses output
class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
    
def use_plotly(min_list, data_names,x,data,x_label='',y_label=''):
    xrev = x[::-1]
    color = px.colors.qualitative.Plotly
    colors = []
    for a in color:
        colors.append('rgba('+str(int(a[1:3],16))+','+str(int(a[3:5],16))+','+str(int(a[5:7],16)))
    for a in range(len(data_names)):
        fig = go.Figure()
        #fig = px.line(x = range(921),y=np.mean(m['exploit'],axis=0))
        for method in methods:
            curr_color = colors[methods.index(method)]            
            std_min = np.ndarray.flatten(np.std(min_list[a][method],axis = 0))
            mean_min = np.ndarray.flatten(np.mean(min_list[a][method],axis=0))
            ci = 1.96 * std_min/10
            fig.add_trace(go.Scatter(x=x, y=mean_min,
                    mode='lines',
                    name=method, line_color=curr_color+',1)',line_shape='spline'))
            fig.add_trace(go.Scatter(
                x=x+xrev,
                y=[mean_min[a]+ci[a] for a in range(len(mean_min))]+[mean_min[a]-ci[a] for a in range(len(mean_min))][::-1],
                fill='toself',
                fillcolor=curr_color+',.2)',
                line_color='rgba(255,255,255,0)',
                showlegend=False,
                name='test',
            ))
        fig.update_layout(
            title = data_names[a],
            xaxis_title= x_label+'Sample Number',
            yaxis_title=y_label+'Normalized Minimum Regret',
            plot_bgcolor='rgba(255,255,255,.9)',
            yaxis_gridcolor = 'rgba(0,0,0,.2)',
            xaxis_gridcolor = 'rgba(0,0,0,.2)',
            xaxis_zeroline = True,
            yaxis_zeroline = True,
            xaxis_zerolinecolor ='rgba(0,0,0,.2)',
            yaxis_zerolinecolor = 'rgba(0,0,0,.2)',
            height = 650,
            width = 800,
            font_size=20
        )
        sor = np.sort(np.ndarray.flatten(data))
        top10p = (sor[-1]-sor[-int(len(sor)/10)])/(sor[-1]-sor[0])
        top10 = (sor[-1]-sor[-10])/(sor[-1]-sor[0])
        top5 = (sor[-1]-sor[-5])/(sor[-1]-sor[0])
        top2 = (sor[-1]-sor[-2])/(sor[-1]-sor[0])
        
        fig.add_trace(go.Scatter(
            
            x=[0,len(data)],
            y=[top10p,top10p],
            legendgroup="group2",
            name="top 10%",
            mode="lines",
            line_color='rgba(120,120,120,1)',
        ))
        fig.add_trace(go.Scatter(
            
            x=[0,len(data)],
            y=[top10,top10],
            legendgroup="group2",
            name="top 10",
            mode="lines",
            line_color='rgba(80,80,80,1)',
        ))
        fig.add_trace(go.Scatter(
            
            x=[0,len(data)],
            y=[top5,top5],
            legendgroup="group2",
            name="top 5",
            mode="lines",
            line_color='rgba(40,40,40,1)',
        ))
        fig.add_trace(go.Scatter(
            
            x=[0,len(data)],
            y=[top2,top2],
            legendgroup="group2",
            name="top 2",
            mode="lines",
            line_color='rgba(0,0,0,1)',
        ))
        fig.show()

def nolog_plotly(min_list,data_names,data):
    for a in range(len(min_list)):
        use_plotly([min_list[a]],[data_names[a]],list(range(len(min_list[a]['exploit'][0]))),data[a])

def log_plotly(min_list, data_names,data):
    for a in range(len(min_list)):
        use_plotly([min_list[a]],[data_names[a]],[np.log(a+1)for a in list(range(len(min_list[a]['exploit'][0])))],data[a],x_label='Log ')

def loglog_plotly(min_list,data_names,data):
    log_min_list = []
    for name in range(len(min_list)):
        tmp = dict()
        for a in methods:
            tp = []
            for b in range(len(min_list[name][a])):
                temp = []
                for c in range(len(min_list[name][a][b])):
                    temp.append([20*-np.log10(min_list[name][a][b][c][0]+.0001)])
                tp.append(temp)
            tmp[a] = (np.array(tp))
        log_min_list.append(tmp)
    for a in range(len(log_min_list)):
        use_plotly([log_min_list[a]],[data_names[a]],[np.log(a+1)for a in list(range(len(min_list[a]['exploit'][0])))],data[a],x_label='Log ',y_label='Log ')

def ylog_plotly(min_list,data_names,data):
    log_min_list = []
    for name in range(len(min_list)):
        tmp = dict()
        for a in methods:
            tp = []
            for b in range(len(min_list[name][a])):
                temp = []
                for c in range(len(min_list[name][a][b])):
                    temp.append([20*-np.log10(min_list[name][a][b][c][0]+.00001)])
                tp.append(temp)
            tmp[a] = (np.array(tp))
        log_min_list.append(tmp)
    for a in range(len(log_min_list)):
        use_plotly([log_min_list[a]],[data_names[a]],list(range(len(min_list[a]['exploit'][0]))),data[a],y_label='Log ')

def use_plotly_log(min_list, data_names,x,data,x_label='',y_label=''):
    xrev = x[::-1]
    ys = []
    y2s = []
    color = px.colors.qualitative.Plotly
    colors = []
    for a in color:
        colors.append('rgba('+str(int(a[1:3],16))+','+str(int(a[3:5],16))+','+str(int(a[5:7],16)))
    for a in range(len(data_names)):
        fig = go.Figure()
        #fig = px.line(x = range(921),y=np.mean(m['exploit'],axis=0))
        for method in methods:
            curr_color = colors[methods.index(method)]            
            std_min = np.ndarray.flatten(np.std(min_list[a][method],axis = 0))
            mean_min = np.ndarray.flatten(-20*np.log10(np.mean(min_list[a][method],axis=0)+.00001))
            mean_min2 = np.ndarray.flatten(np.mean(min_list[a][method],axis=0))
            ci = 1.96 * std_min/10
            fig.add_trace(go.Scatter(x=x, y=mean_min,
                    mode='lines',
                    name=method, line_color=curr_color+',1)',line_shape='spline'))
            y=np.array([mean_min2[a]+ci[a] for a in range(len(mean_min))]+[mean_min2[a]-ci[a] for a in range(len(mean_min))][::-1])
            y[y<0] = 0
            fig.add_trace(go.Scatter(
                x=x+xrev,
                y=-20*np.log10(y+.00001),
                fill='toself',
                fillcolor=curr_color+',.2)',
                line_color='rgba(255,255,255,0)',
                showlegend=False,
                name='test',
                legendgroup = 'group1'
            ))
        
        fig.update_layout(
            title = data_names[a],
            xaxis_title= x_label+'Sample Number',
            yaxis_title=y_label+'Normalized Minimum Regret',
            plot_bgcolor='rgba(255,255,255,.9)',
            yaxis_gridcolor = 'rgba(0,0,0,.2)',
            xaxis_gridcolor = 'rgba(0,0,0,.2)',
            xaxis_zeroline = True,
            yaxis_zeroline = True,
            xaxis_zerolinecolor ='rgba(0,0,0,.2)',
            yaxis_zerolinecolor = 'rgba(0,0,0,.2)',
            height = 650,
            width = 800,
            font_size=20
        )
        
        fig.show()
        
def version_test():
    print("00:27")



########## older versions for testing #########
# plots all of the min regrets passed through with different figure numbers
def plot_all2(minregret_total, label_names,plot_type:str,fig_num):
    for name in range(0,len(label_names)):
        # first name param is for color, second name is for figure number
        plot_mm(minregret_total[name],label_names[name],plot_type,name,fig_num)

#shape of data should be an array of 7 arrays, each being a different method
def process_data2(data_ind,data_org,data_name,max_,min_):
    minregret_ = []
    temp = np.array(data_org)
    for method in range(0,len(methods)):
        minregret_.append([])
        queries_ = data_ind[data_name,methods[method]]
        #print(data_ind[data_name,methods[method]])
        for i in range(0, len(queries_)):
            minregret_[method].append(h.calc_min_regret(temp[queries_[i]], max_)/(max_-min_))
    
    #minregret_ = np.array(minregret_)
    #minregret_[method] = minregret_/(max_-min_)
        
    return minregret_
    

def eat_indices2(data_indices,data_org,data_names):
    minregrets = []
    for data_num in range(0,len(data_indices)):
        curr_data = data_org[data_num]
        curr_name = data_names[data_num]
        curr_ind = data_indices[data_num]
        minregret_ = process_data2(curr_ind,curr_data, curr_name, np.amax(curr_data),np.amin(curr_data))
        plot_all2(minregret_, methods, curr_name,2*data_num)
