#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 21:30:36 2018

@author: vedanshu
"""

"""
overfitting
~~~~~~~~~~~
Plot graphs to illustrate the problem of overfitting.  
"""

# Standard library
import json

# My library
import network as nt

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def make_plots(filename, num_epochs, 
               training_set_size,
               training_cost_xmin=200, 
               test_accuracy_xmin=200, 
               test_cost_xmin=0, 
               training_accuracy_xmin=0):
    """Load the results from ``filename``, and generate the corresponding
    plots. """
    f = open(filename, "r")
    test_cost, test_accuracy, training_cost, training_accuracy \
        = json.load(f)
    f.close()
    plot_training_cost(training_cost, num_epochs, training_cost_xmin)
    plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin)
    plot_test_cost(test_cost, num_epochs, test_cost_xmin)
    plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size)
    plot_overlay(test_accuracy, training_accuracy, num_epochs,
                 min(test_accuracy_xmin, training_accuracy_xmin),
                 training_set_size)

def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs), 
            training_cost[training_cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.show()

def plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_accuracy_xmin, num_epochs), 
            [accuracy/100.0 
             for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([test_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the test data')
    plt.show()

def plot_test_cost(test_cost, num_epochs, test_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_cost_xmin, num_epochs), 
            test_cost[test_cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([test_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the test data')
    plt.show()

def plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs), 
            [accuracy*100.0/training_set_size 
             for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the training data')
    plt.show()

def plot_overlay(test_accuracy, training_accuracy, num_epochs, xmin,
                 training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy/100.0 for accuracy in test_accuracy], 
            color='#2A6EA6',
            label="Accuracy on the test data")
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy*100.0/training_set_size 
             for accuracy in training_accuracy], 
            color='#FFA933',
            label="Accuracy on the training data")
    ax.grid(True)
    ax.set_xlim([xmin, num_epochs])
    ax.set_xlabel('Epoch')
#    ax.set_ylim([90, 100])
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    state = {0: 'NSW', 1: 'QLD', 2: 'SA', 3: 'TAS', 4: 'VIC'}
    year = {0: '2015', 1: '2016', 2: '2017'}
    
    df_nsw = pd.DataFrame()
    df_qld = pd.DataFrame()
    df_sa = pd.DataFrame()
    df_tas = pd.DataFrame()
    df_vic = pd.DataFrame()
    
    df = {'NSW': df_nsw, 'QLD': df_qld, 'SA': df_sa, 'TAS': df_tas, 'VIC': df_vic}
    
    df_nsw_test = pd.DataFrame()
    df_qld_test = pd.DataFrame()
    df_sa_test = pd.DataFrame()
    df_tas_test = pd.DataFrame()
    df_vic_test = pd.DataFrame()
    
    df_test = {'NSW': df_nsw_test, 'QLD': df_qld_test, 'SA': df_sa_test, 'TAS': df_tas_test, 'VIC': df_vic_test}
    
    for st in state.values():
        for ye in year.values():
            for mn in range(1,13):
                if mn < 10:            
                    dataset = pd.read_csv('./datasets/train/' + st + '/PRICE_AND_DEMAND_' + ye + '0' + str(mn) +'_' + st + '1.csv')
                else:
                    dataset = pd.read_csv('./datasets/train/' + st + '/PRICE_AND_DEMAND_' + ye + str(mn) +'_' + st + '1.csv')
                df[st] = df[st].append(dataset.iloc[:,1:3])
        df[st] = df[st].set_index('SETTLEMENTDATE')
       
    for st in state.values():
        dataset = pd.read_csv('./datasets/test/' + st + '/PRICE_AND_DEMAND_201801_' + st + '1.csv')
        df_test[st] = df_test[st].append(dataset.iloc[:,1:3])
        df_test[st] = df_test[st].set_index('SETTLEMENTDATE')
    
#    plt.plot(df['NSW'].iloc[:,0].values)
#    plt.show()
#    plt.plot(df['QLD'].iloc[:,0].values)
#    plt.show()
#    plt.plot(df['SA'].iloc[:,0].values)
#    plt.show()
#    plt.plot(df['TAS'].iloc[:,0].values)
#    plt.show()
#    plt.plot(df['VIC'].iloc[:,0].values)
#    plt.show()
    
    df['NSW'].iloc[:,0] = nt.estimated_autocorrelation(df['NSW'].iloc[:,0].values)
    df['QLD'].iloc[:,0] = nt.estimated_autocorrelation(df['QLD'].iloc[:,0].values)
    df['SA'].iloc[:,0] = nt.estimated_autocorrelation(df['SA'].iloc[:,0].values)
    df['TAS'].iloc[:,0] = nt.estimated_autocorrelation(df['TAS'].iloc[:,0].values)
    df['VIC'].iloc[:,0] = nt.estimated_autocorrelation(df['VIC'].iloc[:,0].values)
    
    df_test['NSW'].iloc[:,0] = nt.estimated_autocorrelation(df_test['NSW'].iloc[:,0].values)
    df_test['QLD'].iloc[:,0] = nt.estimated_autocorrelation(df_test['QLD'].iloc[:,0].values)
    df_test['SA'].iloc[:,0] = nt.estimated_autocorrelation(df_test['SA'].iloc[:,0].values)
    df_test['TAS'].iloc[:,0] = nt.estimated_autocorrelation(df_test['TAS'].iloc[:,0].values)
    df_test['VIC'].iloc[:,0] = nt.estimated_autocorrelation(df_test['VIC'].iloc[:,0].values)
    
#    plt.plot(df['NSW'].iloc[:,0].values)
#    plt.show()
#    plt.plot(df['QLD'].iloc[:,0].values)
#    plt.show()
#    plt.plot(df['SA'].iloc[:,0].values)
#    plt.show()
#    plt.plot(df['TAS'].iloc[:,0].values)
#    plt.show()
#    plt.plot(df['VIC'].iloc[:,0].values)
#    plt.show()
    
    # numpy array
    list_hourly_load_NSW = df['NSW'].iloc[:,0].values
    list_hourly_load_QLD = df['QLD'].iloc[:,0].values
    list_hourly_load_SA = df['SA'].iloc[:,0].values
    list_hourly_load_TAS = df['TAS'].iloc[:,0].values
    list_hourly_load_VIC = df['VIC'].iloc[:,0].values
    
    list_hourly_load_NSW_test = df_test['NSW'].iloc[:,0].values
    list_hourly_load_QLD_test = df_test['QLD'].iloc[:,0].values
    list_hourly_load_SA_test = df_test['SA'].iloc[:,0].values
    list_hourly_load_TAS_test = df_test['TAS'].iloc[:,0].values
    list_hourly_load_VIC_test = df_test['VIC'].iloc[:,0].values
    
    # the length of the sequnce for predicting the future value
    sequence_length = 47
    
    # convert the vector to a 2D matrix
    matrix_load_NSW = nt.convertSeriesToMatrix(list_hourly_load_NSW, sequence_length)
    matrix_load_QLD = nt.convertSeriesToMatrix(list_hourly_load_QLD, sequence_length)
    matrix_load_SA = nt.convertSeriesToMatrix(list_hourly_load_SA, sequence_length)
    matrix_load_TAS = nt.convertSeriesToMatrix(list_hourly_load_TAS, sequence_length)
    matrix_load_VIC = nt.convertSeriesToMatrix(list_hourly_load_VIC, sequence_length)
    
    # numpy array
    matrix_load_NSW = np.array(matrix_load_NSW)
    matrix_load_QLD = np.array(matrix_load_QLD)
    matrix_load_SA = np.array(matrix_load_SA)
    matrix_load_TAS = np.array(matrix_load_TAS)
    matrix_load_VIC = np.array(matrix_load_VIC)

    y_test_NSW = np.array(list_hourly_load_NSW_test)
    y_test_QLD = np.array(list_hourly_load_QLD_test)
    y_test_SA = np.array(list_hourly_load_SA_test)
    y_test_TAS = np.array(list_hourly_load_TAS_test)
    y_test_VIC = np.array(list_hourly_load_VIC_test)
        
    # shift all data by mean    
    # shifted_value = matrix_load.mean()
    # matrix_load -= shifted_value
#    print ("Data  shape: ", matrix_load_NSW.shape)
    
    # split dataset: 90% for training and 10% for testing
    train_row_NSW = int(round(0.9 * matrix_load_NSW.shape[0]))
    train_set_NSW = matrix_load_NSW[:train_row_NSW, :]
    
    train_row_QLD = int(round(0.9 * matrix_load_QLD.shape[0]))
    train_set_QLD = matrix_load_QLD[:train_row_QLD, :]
    
    train_row_SA = int(round(0.9 * matrix_load_SA.shape[0]))
    train_set_SA = matrix_load_SA[:train_row_SA, :]
    
    train_row_TAS = int(round(0.9 * matrix_load_TAS.shape[0]))
    train_set_TAS = matrix_load_TAS[:train_row_TAS, :]
    
    train_row_VIC = int(round(0.9 * matrix_load_VIC.shape[0]))
    train_set_VIC = matrix_load_VIC[:train_row_VIC, :]
    
    # shuffle the training set (but do not shuffle the test set)
    np.random.shuffle(train_set_NSW)
    np.random.shuffle(train_set_QLD)
    np.random.shuffle(train_set_SA)
    np.random.shuffle(train_set_TAS)
    np.random.shuffle(train_set_VIC)
    
    # the training set
    X_train_NSW = train_set_NSW[:, :-1].transpose()
    X_train_QLD = train_set_QLD[:, :-1].transpose()
    X_train_SA = train_set_SA[:, :-1].transpose()
    X_train_TAS = train_set_TAS[:, :-1].transpose()
    X_train_VIC = train_set_VIC[:, :-1].transpose()
    # the last column is the true value to compute the mean-squared-error loss
    y_train_NSW = train_set_NSW[:, -1] 
    y_train_QLD = train_set_QLD[:, -1] 
    y_train_SA = train_set_SA[:, -1] 
    y_train_TAS = train_set_TAS[:, -1] 
    y_train_VIC = train_set_VIC[:, -1] 
    
    # the test set
    X_validation_NSW = matrix_load_NSW[train_row_NSW:, :-1].transpose()
    y_validation_NSW = matrix_load_NSW[train_row_NSW:, -1]
    
    X_validation_QLD = matrix_load_QLD[train_row_QLD:, :-1].transpose()
    y_test_QLD = matrix_load_QLD[train_row_QLD:, -1]
    
    X_validation_SA = matrix_load_SA[train_row_SA:, :-1].transpose()
    y_validation_SA = matrix_load_SA[train_row_SA:, -1]
    
    X_validation_TAS = matrix_load_TAS[train_row_TAS:, :-1].transpose()
    y_validation_TAS = matrix_load_TAS[train_row_TAS:, -1]
    
    X_validation_VIC = matrix_load_VIC[train_row_VIC:, :-1].transpose()
    y_validation_VIC = matrix_load_VIC[train_row_VIC:, -1]
    
    #eemd = EEMD()
    #eIMFs_NSW = eemd(df['NSW'].iloc[:,0].values)
    net = nt.Network([46,10,1], nt.Activation.sigmoid, nt.QuadraticCost)
    dim = sum(w.size + b.size for w,b in zip(net.weights,net.biases))
    net.cso(100,X_train_NSW[:,0].reshape(46,1),y_train_NSW[0].reshape(1,1),net.objectiveFunction,-0.6,0.6,dim ,500)
    net.set_weight_bias(np.array(net.get_Gbest()))
    
    fname = "results"
    num_epochs = 13
    lmbda = 2
    
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(
            X_train_NSW,y_train_NSW, num_epochs, 10, 0.01, X_validation_NSW, y_validation_NSW, 
            lmbda, monitor_evaluation_cost = True,
            monitor_evaluation_accuracy = True,
            monitor_training_cost = True,
            monitor_training_accuracy = True)
    
    
    f = open(fname, "w")
    json.dump([evaluation_cost, evaluation_accuracy, training_cost, training_accuracy], f)
    f.close()
        
    make_plots(fname, num_epochs, training_set_size = X_validation_NSW.shape[1],
               training_cost_xmin = 0,
               test_accuracy_xmin = 0,
               test_cost_xmin = 0, 
               training_accuracy_xmin = 0)
