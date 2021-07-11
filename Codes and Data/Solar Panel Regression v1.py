"""
Solar Panel Regression to Predict Output
Electricity Generated given Weather Sensor Information

First Written: 10/18/2020

Emily Ford
"""
#%% Preliminaries
from __future__ import absolute_import, division, print_function  # Python 2/3 compatibility

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
import time
import math
import datetime
import random

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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

def create_model(n_layer, n_neuron,act, dropout, opti, n_components=5, output_shape=1): # ANN model generation
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
saveon   = 1       # 1 = Yes Save inputs and output results
numtrial = 1       # Number of trials to run back-to-back
n_fold = 5         # Cross-Validation # of folds
plotshow = 1       # Show plots or not
plotval = 0        # Show ANN history plot or not
importance = 0     # Print out impurity-based feature importances for each forest tested

act = 'relu' # Activation function for ANN
learnr = 0.001 # Learning rate for optimization
opti = RMSprop(learning_rate=learnr) # Optimizer (RMSprop lr =.003, SGD lr = 0.01, momentum=0.5, Adam lr =0.001 default, Adagrad lr = 0.01 recommended)
n_epoch = 800 # Epoches to train ANN model
n_layer = 4   # Number of layers
n_neuron = 58 # Starting # of neurons per layer (decreases by 2^(n-1) layer)
dropout = 0.0 # Dropout rate

degree = 1 # Polynomial for linear combination

ml_options = [("Linear", LinearRegression(normalize=False)), # Choice of ML and individual options to run
            ("ANN",KerasRegressor(build_fn=create_model,n_layer=n_layer,n_neuron=n_neuron,dropout=dropout, 
                act=act,opti=opti,epochs=n_epoch,verbose=0)),

            ("RF",RandomForestRegressor(n_estimators=99,max_depth=14,
                            min_samples_split=4,min_samples_leaf=2)), 

            ("ET",ExtraTreesRegressor(n_estimators=186,max_depth=20,
                           min_samples_split=3,min_samples_leaf=3)),

            ("GBT",GradientBoostingRegressor(n_estimators=190,max_depth=11,
                              min_samples_split=6,min_samples_leaf=8))]

outmodel = 'Final/Official'
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

# Split into folds for CV testing and training
n_Data = y_data.shape[0]
holistic_ind = np.random.permutation(n_Data) # Randomly shuffle data
ind = np.zeros((n_Data//n_fold,n_fold)).astype('int32')  # Zeros array to store each fold's integer index

for i in range(n_fold): # split the data into folds
    if i ==0:
        ind[:,i] = holistic_ind[0:int(round(n_Data * ((i+1)/n_fold)))]  # Grab first segment
    else:
        ind[:,i] = holistic_ind[int(round(n_Data*(i/n_fold))): int(round(n_Data*((i+1)/n_fold)))] # Grab remaining segments

#%% Various ML Trials
filepath_m =os.path.dirname(os.path.realpath(__file__))+'\\'

def printplotresults(Y_test_norm, Y_pred_test_norm,plotshow, normalizer,Y):
    ''' Print values of errors to screen. Plot resulting fits '''
    # %% Convert Y data to Original Scales for Plotting   
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

    # %% Plotting Y Data    
    if plotshow == 1:  
        fig = plt.figure(figsize =(6,6)) # Fix figure size    
        ax = fig.add_subplot(111)
        sortorder= np.argsort(Y_test,axis=0)   # Sort test data smallest to largest and remember indices
        Y_test2 = Y_test[sortorder]           # Re-ordered Y test for plotting
        Y_test2 = Y_test2.reshape(-1,1)       # Reshape for indecies call
        # Plot +/- 10 % of experimental results bars about line
        plt.plot((Y_test2[0,0],Y_test2[len(Y_test)-1,0]),(Y_test2[0,0]+0.05*Y_test2[0,0],Y_test2[len(Y_test)-1,0]+0.05*Y_test2[len(Y_test)-1,0]), '--', color='black') # Exp vs. Exp Test Data ,label='1:1 Line +5%'
        plt.plot((Y_test2[0,0],Y_test2[len(Y_test)-1,0]),(Y_test2[0,0]-0.05*Y_test2[0,0],Y_test2[len(Y_test)-1,0]-0.05*Y_test2[len(Y_test)-1,0]), '--', color='black') # Exp vs. Exp Test Data ,label='1:1 Line -5%'
        plt.scatter(Y_test[:,0],Y_pred_test[:,0], color='blue',label ='Predicted Data') # M vs.Stiffness Test Data
        plt.plot(Y_test[:,0],Y_test[:,0], color='black',label='1:1 Line') # M vs. M Test Data
        
        plt.xlabel('Total Daily Yield (kW)',fontsize=15,weight='bold',labelpad=20)
        plt.ylabel('ML Predicted Total Daily Yield (kW)',fontsize=15,weight='bold') #$\mathbf{}
        #plt.ticklabel_format(axis='both', style='sci')   
        min_g = 0 #round(0.7*Y_test2[0,0])
        max_g = 2.1*10**5 #round(1.3*Y_test2[len(Y_test)-1,0])
        diff_g = 5.0*10**4 # kW E4 

        plt.ylim(min_g, max_g) # Min, Max
        plt.xlim(min_g, max_g) # Min, Max
        plt.yticks(np.arange(min_g, max_g, diff_g)) # Min, Max, Step Size
        plt.xticks(np.arange(min_g, max_g, diff_g)) # Min, Max, Step Size        
        
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

        plt.tight_layout()
        plt.show()
        
    return error

def tracksave(mltype,ml,error,outmodel,trainfile,elapsed):
    if mltype != 'ANN':
        params = ml.get_params() # Dictionary of ML model's input parameters
    if mltype == 'ANN':
        tracknew = pd.DataFrame(np.array([[outmodel, n_layer, n_neuron,dropout, 
        act, str(opti), learnr, n_epoch]]),
        columns =["Model Name","Number of Hidden Layers (not counting dropout between)",
        "Starting # Neurons/Layer","Dropout Rate","Activation","Optimization","Learning Rate","Epochs"])
        savefile = '_Proj ANN Models'
    elif mltype == 'Linear':
        tracknew = pd.DataFrame(np.array([[outmodel, degree]]),
        columns =["Model Name","Degree"])
        savefile = '_Proj Linear Models'
    elif mltype == 'RF' or mltype == 'ET' or mltype == 'GBT':
        tracknew = pd.DataFrame(np.array([[outmodel, str(ml), params['n_estimators'],
        params['max_depth'],params['min_samples_split'],params['min_samples_leaf']]]),
        columns =["Model Name","Forest", "n_estimators",
        "max_depth","min_samples_split","min_samples_leaf"])
        savefile = '_Proj Forest Models'
            
    # Import Model Tracker, Export with new line of information 
    tracknew = pd.concat([tracknew, error], axis=1)
    track = pd.read_excel(filepath_m+savefile+'.xlsx',header=0,index_col=0)
    d = datetime.datetime.today()
    tracknew = pd.concat([tracknew, pd.DataFrame(np.array([[trainfile,d.strftime('%m-%d-%Y'),elapsed]]),
            columns=["Test Data","Date","Run Time (sec)"])], axis=1) # Combine Columns
        
    track = pd.concat([track,tracknew], axis=0, join='outer',ignore_index=True)
    track.to_excel(filepath_m+savefile+'.xlsx')

# Test various ML
for i in range(numtrial): # Loop over number of trials
    for j in range(len(ml_options)): # Loop through each of the selected ML models
        for k in range(n_fold): # Loop through each fold of data
            # indexing and get the data
            x_test = x[ind[:,k]]
            y_test = y_data_norm[ind[:,k]]
            x_train= x[np.delete(ind, k, axis=1).reshape(ind.shape[0]*(n_fold-1))] # Call all columns except the test one
            y_train = y_data_norm[np.delete(ind, k, axis=1).reshape(ind.shape[0]*(n_fold-1))] # Call all columns except the test one

            t=time.time() # Start time of analysis 

            if ml_options[j][0] == 'ANN': # Handle KerasRegressor model training/predicting
                hist = ml_options[j][1].fit(x_train, y_train,validation_data=(x_test, y_test)) # Output is history of training (not model)
                y_pred = ml_options[j][1].predict(x_test) 

                if plotval == 1:  # Validation loss (History) plot
                    plt.figure()
                    ax = plt.subplot()
                    ax.plot(hist.history["loss"],'r', marker='.', label="Train Loss")
                    ax.plot(hist.history["val_loss"],'b', marker='.', label="Validation Loss")
                    plt.ylabel("Mean Squared Error")
                    plt.xlabel("Epoches")
                    ax.legend()
                    plt.show()

            else: # Linear or Forests
                model = ml_options[j][1].fit(x_train, y_train)
                y_pred = model.predict(x_test) 
            
            error = printplotresults(y_test.reshape(-1,1), y_pred.reshape(-1,1),plotshow,normalizer,y_data)
            
            if importance == 1 and (ml_options[j][0] == 'RF' or ml_options[j][0] == 'ET' or ml_options[j][0] == 'GBT'):
                print('\n' + ml_options[j][0]+' Fold '+str(k))
                importances = ml_options[j][1].feature_importances_
                imp_indices = np.argsort(importances)[::-1]
                for f in range(len(importances)):
                    print("%d. feature %d (%e)" % (f + 1, imp_indices[f], importances[imp_indices[f]]))

            elapsed=time.time()-t # End time of analysis
            if saveon == 1:
                tracksave(ml_options[j][0],ml_options[j][1],error,outmodel+' cv '+str(k),filename,elapsed)
        print('Successful end of ML test '+ml_options[j][0]+' Trial '+str(i))

# %%
