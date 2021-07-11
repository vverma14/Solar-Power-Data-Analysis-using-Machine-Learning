"""
Parameter Search for Optimized ML 
Solar Panel Regression

First Written: 11/29/2020

Emily Ford
"""
#%% Preliminaries
#%% Preliminaries
from __future__ import absolute_import, division, print_function  # Python 2/3 compatibility

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import datetime
import random
import scipy.stats as stat

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

#import tensorflow as tf
#tf.enable_eager_execution() # Fix executing eagerly to be true
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models  import Sequential, K, load_model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop, Adagrad

# All plot format settings
plt.rc('axes', linewidth=2,labelsize=15,labelweight='bold') # box lines, X/Y axis label size and weight
plt.rc('font', weight='bold',size=13) # Tick label size and weight
plt.rc('xtick.major',width=2,size=7) # Increase xtick width and length
plt.rc('xtick.minor',width=2,size=3.5) # Increase xtick width and length
plt.rc('ytick.major',width=2,size=7) # Increase ytick width and length
plt.rc('ytick.minor',width=2,size=3.5) # Increase xtick width and length

act = 'relu' # Activation function for ANN
learnr = 0.001 # Learning rate for optimization
opti = RMSprop(learning_rate=learnr) # Optimization scheme for ANN
n_epoch = 200 # Epoches to train ANN model

def create_model(n_layer, n_neuron, dropout,  n_components=5, output_shape=1, act=act, opti=opti): # ANN model generation
    """ANN regressor."""                                    
    model = Sequential()
    # for the firt layer we need to specify the input dimensions
    first=True
    for j in range(n_layer): # For loop from 0 to nl1-1 (does not include nl1 value)
        if first:
            model.add(Dense(n_neuron, input_dim=n_components, activation=act))
            first=False
        else: 
            model.add(Dense(int(n_neuron*(2**-j)), activation=act))
        if dropout !=0:
            model.add(Dropout(dropout))

    model.add(Dense(output_shape, activation='linear')) # Output layer
    #model.add(Dense(output_shape, activation='sigmoid'))
    model.summary()
    model.compile(optimizer=opti, loss='mean_squared_error', metrics=['mae'])
    return model # ANN model generated

#%% Inputs
nfold = 3 # Number of n-fold cross-validations to evaluate
randomsearch =0 # 1 = RandomizedCV Search, else Grid search
n_iter_search = 30

# Choice of ML and individual options to run
ml_options = [("ANN",KerasRegressor(build_fn=create_model, 
                    epochs=n_epoch,verbose=0)), 
              ("RF",RandomForestRegressor()), 
              ("ET",ExtraTreesRegressor()),
              ("GBT",GradientBoostingRegressor())]

if randomsearch == 1: # Dictionaries of sets of parameters to tune
    n_layer = stat.randint(1,8)  # numlayer/neuon radmonly select from the permutated list (start, end-1)
    n_neuron = stat.randint(15,86)
    #dropout = stat.uniform(loc=0,scale=0.25) # % to delete data from each neuron for record. Random values between 0 and 0+n
    dropout = (0, 0.05, 0.1) # 0.15, 0.2, 0.25, 0.30)
    
    n_estimators_rf = stat.randint(50,226)
    n_estimators_et = n_estimators_rf 
    n_estimators_gbt = n_estimators_rf 
    max_depth = stat.randint(8,21)
    min_samples_split = stat.randint(2,11)
    min_samples_leaf = stat.randint(2,11)

else:  
    n_layer = [2,3,4]
    n_neuron = list(range(50,61))
    dropout = [0.0, 0.05] #[0.05,0.1] # % to delete data from each neuron for record.

    n_estimators_rf =  list(range(95,106))
    n_estimators_et =  list(range(185,196))
    n_estimators_gbt =  list(range(188,201))
    max_depth = [10,15,20,25]
    min_samples_split = [4,6]
    min_samples_leaf = [2,6]

outmodel = 'Grid'
filename = 'Both Plants Weather Sensor Data.csv' # Input Weather Data
filename_output = 'Both Plants Generation Yield Sum Data.csv' # Output Sum of all the Solar Panels' Daily Yield
normalizer = StandardScaler()

#%% Load Data
scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error','r2']

df_in = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+'\\'+filename,header=0) #Import the input .csv file
#df_in = pd.read_csv(os.getcwd()+'\\'+filename,header=0) #Import the input .csv file

x_day = pd.to_datetime(df_in['DATE_TIME']).dt.date # Year-Month-Day
for i in range(x_day.size):
    x_day[i]= x_day[i].toordinal() # Convert each date to the Gregorian ordinal
x_day =  x_day.to_numpy().astype('float64').reshape(-1,1) # Change to float64 and update shape

x_time = pd.to_datetime(df_in['DATE_TIME']).dt.time # Time (Military)
for i in range(x_time.size):
    x_time[i]= x_time[i].hour*60+x_time[i].minute # Convert each time be out of total 1440 minutes in a day
x_time = x_time.to_numpy(dtype=object).reshape(-1,1)

x_data = df_in.iloc[:,3:6].values # Ambient Temp, Module Temp, Irradiation
x_orig = np.concatenate((x_day,x_time,x_data),axis = 1) # Combine Date, Time, columns with rest of inputs
x = normalizer.fit_transform(x_orig) # Normalize by Standard Scaler
x_label = df_in.iloc[:,1:3] # Plant ID, Source Key

df_out = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+'\\'+filename_output,header=0) #Import the output .csv file
#df_out = pd.read_csv(os.getcwd()+'\\'+filename_output,header=0) #Import the output .csv file
y_data = df_out.iloc[:,1].values.reshape(-1,1) # Sum of Daily Yield that aligns perfectly with input date/time
y_data_norm = normalizer.fit_transform(y_data) # Normalize by Standard Scaler

# 'pipeline name__parameter' for grid search definitions. Dictionary of dictionaries
params_to_tune = {'ANN':{'n_layer': n_layer,
                    'n_neuron': n_neuron,
                    'dropout': dropout},
                'RF':{'n_estimators': n_estimators_rf,
                    'max_depth': max_depth,      
                    'min_samples_split': min_samples_split, 
                    'min_samples_leaf': min_samples_leaf},
                'ET':{'n_estimators': n_estimators_et,
                    'max_depth': max_depth,      
                    'min_samples_split': min_samples_split, 
                    'min_samples_leaf': min_samples_leaf},  
                'GBT':{'n_estimators': n_estimators_gbt,
                    'max_depth': max_depth,      
                    'min_samples_split': min_samples_split, 
                    'min_samples_leaf': min_samples_leaf}}

#%% Various ML Trials
# Split into test and train data (already normalized)
x_train, x_test, y_train, y_test = train_test_split(x.reshape(len(x),-1),
                                                    y_data_norm,test_size=0.25)
filepath_m =os.path.dirname(os.path.realpath(__file__))+'\\'

def gridrun(ml_name,grid_search,x_train,y_train,x_test,y_test,normalizer,y_data):
        G = grid_search.fit(x_train, y_train)
        #grid_result = pd.DataFrame(G.cv_results_) # All results of grid search
        
        # Predict Output
        y_pred_test = G.predict(x_test) # Predict on unseen validation data set
        error = printplotresults(y_test.reshape(-1,1), y_pred_test.reshape(-1,1),normalizer,y_data)
        elapsed=time.time()-t # End time of analysis
        tracksave(ml_name,G.best_estimator_,error,outmodel,filename,elapsed) # Call the regressor
        
def printplotresults(Y_test_norm, Y_pred_test_norm,normalizer,Y):
    ''' Print values of errors to screen. Plot resulting fits '''
    # Preliminaries
    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # %% Convert X and Y data to Original Scales for Plotting   
    normalizer.fit(Y)
    Y_test = normalizer.inverse_transform(Y_test_norm)
    Y_pred_test = normalizer.inverse_transform(Y_pred_test_norm)

    print("\nScaled Values")
    print('Mean Square Error   E Data is {:.2E}'.format(mean_squared_error(Y_test[:,0],Y_pred_test[:,0]))) # Regression Metric
    print('RMSE                E Data is {:.2E}'.format(math.sqrt(mean_squared_error(Y_test[:,0],Y_pred_test[:,0])))) # Regression Metric
    print('Mean Absolute Error E Data is {:.2E}'.format(mean_absolute_error(Y_test[:,0],Y_pred_test[:,0]))) # Regression Metric
    print('Coeff. of Det (R^2) E Data is {:.3f}'.format(r2_score(Y_test[:,0],Y_pred_test[:,0]))) # Regression Metric
    
    column = ["MSE E Model","MAE E Model","COD E Model"]    
    error = pd.DataFrame(np.array([[mean_squared_error(Y_test[:,0],Y_pred_test[:,0]),
    mean_absolute_error(Y_test[:,0],Y_pred_test[:,0]),
    r2_score(Y_test[:,0],Y_pred_test[:,0])]]),
    columns =column)

    return error

def tracksave(mltype,ml,error,outmodel,trainfile,elapsed):
    params = ml.get_params() # Dictionary of ML model's input parameters

    if mltype == 'ANN':
        tracknew = pd.DataFrame(np.array([[outmodel, params['n_layer'],
        params['n_neuron'], params['dropout'], 
        act, str(opti), learnr, n_epoch]]),
        columns =["Model Name","Number of Hidden Layers (not counting dropout between)",
        "Starting # Neurons/Layer","Dropout Rate","Activation","Optimization","Learning Rate","Epochs"])
        savefile = 'ANN Parameter'
    elif mltype == 'RF' or mltype == 'ET' or mltype == 'GBT':
        tracknew = pd.DataFrame(np.array([[outmodel, str(ml), params['n_estimators'],
        params['max_depth'],params['min_samples_split'],params['min_samples_leaf']]]),
        columns =["Model Name","Forest", "n_estimators",
        "max_depth","min_samples_split","min_samples_leaf"])
        savefile = 'Forest Parameter'
            
    # Import Model Tracker, Export with new line of information 
    tracknew = pd.concat([tracknew, error], axis=1)
    track = pd.read_excel(filepath_m+savefile+'.xlsx',header=0,index_col=0)
    d = datetime.datetime.today()
    tracknew = pd.concat([tracknew, pd.DataFrame(np.array([[trainfile,d.strftime('%m-%d-%Y'),elapsed]]),
            columns=["Test Data","Date","Run Time (sec)"])], axis=1) # Combine Columns
        
    track = pd.concat([track,tracknew], axis=0, join='outer',ignore_index=True)
    track.to_excel(filepath_m+savefile+'.xlsx')

for j in range(len(ml_options)): # Loop through each of the seleceted ML models
    t=time.time() # Start time of analysis
    
    if randomsearch == 1:
        grid_search = RandomizedSearchCV(ml_options[j][1],param_distributions=params_to_tune[ml_options[j][0]],n_iter=n_iter_search, scoring=scoring,cv=nfold,refit='neg_mean_squared_error') # n-fold validation
    else:
        grid_search = GridSearchCV(ml_options[j][1],param_grid=params_to_tune[ml_options[j][0]],scoring=scoring,cv=nfold,refit='neg_mean_squared_error') # n-fold grid validation

    gridrun(ml_options[j][0],grid_search,x_train,y_train,x_test,y_test,normalizer,y_data)

    print('Successful end of ML test '+ml_options[j][0])

# %%
