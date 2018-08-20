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
from sklearn.model_selection import TimeSeriesSplit
from os import path

def make_plots(filename, num_epochs, 
               training_cost_xmin=200, 
               test_accuracy_xmin=200, 
               test_cost_xmin=0, 
               training_accuracy_xmin=0,
               plt_training_cost = False,
               plt_training_accuracy = False,
               plt_test_cost = False,
               plt_test_accuracy = True,
               plt_overlay = False):
    """Load the results from ``filename``, and generate the corresponding
    plots. """
    f = open(filename, "r")
    test_cost, test_mape, test_rmse, test_mae, training_cost, training_mape, \
        training_rmse, training_mae = json.load(f)
    f.close()
    if plt_training_cost:
        plot_training_cost(training_cost, num_epochs, training_cost_xmin)
    if plt_test_accuracy:
        plot_test_accuracy(test_mape, num_epochs, test_accuracy_xmin)
    if plt_test_cost:
        plot_test_cost(test_cost, num_epochs, test_cost_xmin)
    if plt_training_accuracy:
        plot_training_accuracy(training_mape, num_epochs, 
                               training_accuracy_xmin)
    if plt_overlay:
        plot_overlay(test_mape, training_mape, num_epochs,
                 min(test_accuracy_xmin, training_accuracy_xmin))

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
            [accuracy
             for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([test_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Error on the test data')
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
                           training_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs), 
            [accuracy 
             for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Error on the training data')
    plt.show()

def plot_overlay(test_accuracy, training_accuracy, num_epochs, xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy for accuracy in test_accuracy], 
            color='#2A6EA6',
            label="Error on the test data")
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy 
             for accuracy in training_accuracy], 
            color='#FFA933',
            label="Error on the training data")
    ax.grid(True)
    ax.set_xlim([xmin, num_epochs])
    ax.set_xlabel('Epoch')
    ax.set_ylim([0, 20])
    plt.legend(loc="lower right")
    plt.show()

def timeSeriesSplit(cso = False):
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
       
    # numpy array
    list_hourly_load_NSW = np.array(df['NSW'])
    list_hourly_load_QLD = np.array(df['QLD'])
    list_hourly_load_SA = np.array(df['SA'])
    list_hourly_load_TAS = np.array(df['TAS'])
    list_hourly_load_VIC = np.array(df['VIC'])
       
    # the length of the sequnce for predicting the future value
    sequence_length = 84
    x_size = 36
    hidden = 10
    y_size = 48
    
    # normalizing
    matrix_load_NSW = list_hourly_load_NSW / np.linalg.norm(list_hourly_load_NSW)
    matrix_load_QLD = list_hourly_load_QLD / np.linalg.norm(list_hourly_load_QLD)
    matrix_load_SA = list_hourly_load_SA / np.linalg.norm(list_hourly_load_SA)
    matrix_load_TAS = list_hourly_load_TAS / np.linalg.norm(list_hourly_load_TAS)
    matrix_load_VIC = list_hourly_load_VIC / np.linalg.norm(list_hourly_load_VIC)
    
    matrix_load_NSW = matrix_load_NSW[:-(len(matrix_load_NSW) % sequence_length)]
    matrix_load_QLD = matrix_load_QLD[:-(len(matrix_load_QLD) % sequence_length)]
    matrix_load_SA = matrix_load_SA[:-(len(matrix_load_SA) % sequence_length)]
    matrix_load_TAS = matrix_load_TAS[:-(len(matrix_load_TAS) % sequence_length)]
    matrix_load_VIC = matrix_load_VIC[:-(len(matrix_load_VIC) % sequence_length)]
    
    matrix_load_NSW = matrix_load_NSW.reshape(-1, sequence_length)
    matrix_load_QLD = matrix_load_QLD.reshape(-1, sequence_length)
    matrix_load_SA = matrix_load_SA.reshape(-1, sequence_length)
    matrix_load_TAS = matrix_load_TAS.reshape(-1, sequence_length)
    matrix_load_VIC = matrix_load_VIC.reshape(-1, sequence_length)
    
    # shuffle the training set (but do not shuffle the test set)
    np.random.shuffle(matrix_load_NSW)
    np.random.shuffle(matrix_load_QLD)
    np.random.shuffle(matrix_load_SA)
    np.random.shuffle(matrix_load_TAS)
    np.random.shuffle(matrix_load_VIC)
    
    # the training set
    X_NSW = matrix_load_NSW[:, :x_size]
    X_QLD = matrix_load_QLD[:, :x_size]
    X_SA = matrix_load_SA[:, :x_size]
    X_TAS = matrix_load_TAS[:, :x_size]
    X_VIC = matrix_load_VIC[:, :x_size]
    
    # the last column is the true value to compute the mean-squared-error loss
    y_NSW = matrix_load_NSW[:, x_size:]
    y_QLD = matrix_load_QLD[:, x_size:]
    y_SA = matrix_load_SA[:, x_size:]
    y_TAS = matrix_load_TAS[:, x_size:]
    y_VIC = matrix_load_VIC[:, x_size:]
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    X = {'NSW': X_NSW, 'QLD': X_QLD, 'SA': X_SA, 'TAS': X_TAS, 'VIC': X_VIC}
    y = {'NSW': y_NSW, 'QLD': y_QLD, 'SA': y_SA, 'TAS': y_TAS, 'VIC': y_VIC}
    
    for st in state.values():
        print("State: ", st)
        i = 1
        for train_index, test_index in tscv.split(X[st]):
            X_train, X_test = X[st][train_index], X[st][test_index]
            y_train, y_test = y[st][train_index], y[st][test_index]
            
            print("Train and validation from state ", st, " split ", i)
            net = nt.Network([x_size, hidden, y_size], nt.Activation.tanh, nt.QuadraticCost)
            if cso:
                fname = "kernelBiasTimeSeries" + st + ".npy"
                if not path.exists(fname):
                    print("Weights and biases initialization for state ",st, " in progress...")
                    randInt = np.random.randint(X_train.shape[0])
                    net.cso(100,X_train[randInt].reshape(x_size,1),y_train[randInt].reshape(y_size,1),
                                net.multiObjectiveFunction,-0.6,0.6,net.dim ,100)
                    net.set_weight_bias(np.array(net.get_Gbest()))
                    np.save(fname, np.array(net.get_Gbest()))
                net.set_weight_bias(np.load(fname))

            if cso:
                fname = "results_" + st + "_TS_" + str(i) + "CSO"
            else:
                fname = "results_" + st + "_TS_" + str(i) + "GD"
            num_epochs = 1500
            lmbda = 2
            
            evaluation_cost, eval_mape, eval_rmse, eval_mae, training_cost, training_mape, training_rmse, training_mae = net.SGD(
                    X_train.transpose(),y_train.transpose(), num_epochs, 
                    10, 0.01, 
                    X_test.transpose(), y_test.transpose(), 
                    lmbda, monitor_evaluation_cost = True,
                    monitor_evaluation_accuracy = True,
                    monitor_training_cost = True,
                    monitor_training_accuracy = True,
                    output2D = True)
            
            f = open(fname, "w")
            json.dump([evaluation_cost, eval_mape, eval_rmse, eval_mae, training_cost, training_mape, training_rmse, training_mae], f)
            f.close()
                
#            make_plots(fname, num_epochs,
#                       training_cost_xmin = 0,
#                       test_accuracy_xmin = 0,
#                       test_cost_xmin = 0, 
#                       training_accuracy_xmin = 0)
            i = i+1

def fiveFoldCrossValidation(cso = False):
    #State and year to use for training and testing
    state = {0: 'NSW', 1: 'QLD', 2: 'SA', 3: 'TAS', 4: 'VIC'}
#    state = {0: 'NSW'}
    year = {0: '2015', 1: '2016', 2: '2017'}
#    year = {0: '2015'}
    
    hidden = 10
    
    #Training and testing batches
    x_batches = {}
    y_batches = {} 
    
    #parameters for 5 fold validation 
    set_size = 84 
    x_size = 36
    y_size = 48
    x_batches_validation_fold ={}
    y_batches_validation_fold ={}
    x_batches_train_fold = {}
    y_batches_train_fold = {}

    df_nsw = pd.DataFrame()
    df_qld = pd.DataFrame()
    df_sa = pd.DataFrame()
    df_tas = pd.DataFrame()
    df_vic = pd.DataFrame()
    
    df = {'NSW': df_nsw, 'QLD': df_qld, 'SA': df_sa, 'TAS': df_tas, 'VIC': df_vic}
    
    for st in state.values():
        for ye in year.values():
            for mn in range(1,13):
                if mn < 10:            
                    dataset = pd.read_csv('./datasets/train/' + st + '/PRICE_AND_DEMAND_' + ye + '0' + str(mn) +'_' + st + '1.csv')
                else:
                    dataset = pd.read_csv('./datasets/train/' + st + '/PRICE_AND_DEMAND_' + ye + str(mn) +'_' + st + '1.csv')
                df[st] = df[st].append(dataset.iloc[:,1:3])
        df[st] = df[st].set_index('SETTLEMENTDATE')
    
    TS_NSW = np.array(df['NSW'])
    TS_QLD = np.array(df['QLD'])
    TS_SA = np.array(df['SA'])
    TS_TAS = np.array(df['TAS'])
    TS_VIC = np.array(df['VIC'])
      
    #Normalizing the dataset
    TS_NSW = TS_NSW / np.linalg.norm(TS_NSW)
    TS_QLD = TS_QLD / np.linalg.norm(TS_QLD)
    TS_SA = TS_SA / np.linalg.norm(TS_SA)
    TS_TAS = TS_TAS / np.linalg.norm(TS_TAS)
    TS_VIC = TS_VIC / np.linalg.norm(TS_VIC)

    """ Making the dataset size divisible by num_period """
    TS_NSW = TS_NSW[:(len(TS_NSW) -(len(TS_NSW) % set_size))] 
    TS_QLD = TS_QLD[:(len(TS_QLD)- (len(TS_QLD) % set_size))]
    TS_SA = TS_SA[:(len(TS_SA) -(len(TS_SA) % set_size))]
    TS_TAS = TS_TAS[:(len(TS_TAS) -(len(TS_TAS) % set_size))]
    TS_VIC = TS_VIC[:(len(TS_VIC) - (len(TS_VIC) % set_size))] 
    
    """ Making our training dataset with batch size of num_period """
    TS_batches = {'NSW': TS_NSW.reshape(-1, set_size).transpose(),
                 'QLD': TS_QLD.reshape(-1, set_size).transpose(),
                 'SA': TS_SA.reshape(-1, set_size).transpose(),
                 'TAS': TS_TAS.reshape(-1, set_size).transpose(),
                 'VIC': TS_VIC.reshape(-1, set_size).transpose()}
    
    x_batches = {'NSW': TS_batches['NSW'][:x_size,:],
                 'QLD': TS_batches['QLD'][:x_size,:],
                 'SA': TS_batches['SA'][:x_size,:],
                 'TAS': TS_batches['TAS'][:x_size,:],
                 'VIC': TS_batches['VIC'][:x_size,:]}
    
    y_batches = {'NSW': TS_batches['NSW'][x_size:,:],
                 'QLD': TS_batches['QLD'][x_size:,:],
                 'SA': TS_batches['SA'][x_size:,:],
                 'TAS': TS_batches['TAS'][x_size:,:],
                 'VIC': TS_batches['VIC'][x_size:,:]}

    #Making validation set
    x_batches_validation_fold[1] = {'NSW': x_batches['NSW'][:, np.arange(0,x_batches['NSW'].shape[1],5)],
                             'QLD': x_batches['QLD'][:, np.arange(0,x_batches['QLD'].shape[1],5)],
                             'SA': x_batches['SA'][:, np.arange(0,x_batches['SA'].shape[1],5)],
                             'TAS': x_batches['TAS'][:, np.arange(0,x_batches['TAS'].shape[1],5)],
                             'VIC': x_batches['VIC'][:, np.arange(0,x_batches['VIC'].shape[1],5)]}
    
    x_batches_validation_fold[2] = {'NSW': x_batches['NSW'][:, np.arange(1,x_batches['NSW'].shape[1],5)],
                             'QLD': x_batches['QLD'][:, np.arange(1,x_batches['QLD'].shape[1],5)],
                             'SA': x_batches['SA'][:, np.arange(1,x_batches['SA'].shape[1],5)],
                             'TAS': x_batches['TAS'][:, np.arange(1,x_batches['TAS'].shape[1],5)],
                             'VIC': x_batches['VIC'][:, np.arange(1,x_batches['VIC'].shape[1],5)]}
    
    x_batches_validation_fold[3] = {'NSW': x_batches['NSW'][:, np.arange(2,x_batches['NSW'].shape[1],5)],
                             'QLD': x_batches['QLD'][:, np.arange(2,x_batches['QLD'].shape[1],5)],
                             'SA': x_batches['SA'][:, np.arange(2,x_batches['SA'].shape[1],5)],
                             'TAS': x_batches['TAS'][:, np.arange(2,x_batches['TAS'].shape[1],5)],
                             'VIC': x_batches['VIC'][:, np.arange(2,x_batches['VIC'].shape[1],5)]}
    
    x_batches_validation_fold[4] = {'NSW': x_batches['NSW'][:,np.arange(3,x_batches['NSW'].shape[1],5)],
                             'QLD': x_batches['QLD'][:, np.arange(3,x_batches['QLD'].shape[1],5)],
                             'SA': x_batches['SA'][:, np.arange(3,x_batches['SA'].shape[1],5)],
                             'TAS': x_batches['TAS'][:, np.arange(3,x_batches['TAS'].shape[1],5)],
                             'VIC': x_batches['VIC'][:, np.arange(3,x_batches['VIC'].shape[1],5)]}
    
    x_batches_validation_fold[5] = {'NSW': x_batches['NSW'][:, np.arange(4,x_batches['NSW'].shape[1],5)],
                             'QLD': x_batches['QLD'][:, np.arange(4,x_batches['QLD'].shape[1],5)],
                             'SA': x_batches['SA'][:, np.arange(4,x_batches['SA'].shape[1],5)],
                             'TAS': x_batches['TAS'][:, np.arange(4,x_batches['TAS'].shape[1],5)],
                             'VIC': x_batches['VIC'][:, np.arange(4,x_batches['VIC'].shape[1],5)]}
     
    y_batches_validation_fold[1] = {'NSW': y_batches['NSW'][:, np.arange(0,y_batches['NSW'].shape[1],5)],
                             'QLD': y_batches['QLD'][:, np.arange(0,y_batches['QLD'].shape[1],5)],
                             'SA': y_batches['SA'][:, np.arange(0,y_batches['SA'].shape[1],5)],
                             'TAS': y_batches['TAS'][:, np.arange(0,y_batches['TAS'].shape[1],5)],
                             'VIC': y_batches['VIC'][:, np.arange(0,y_batches['VIC'].shape[1],5)]}
    
    y_batches_validation_fold[2] = {'NSW': y_batches['NSW'][:, np.arange(1,y_batches['NSW'].shape[1],5)],
                             'QLD': y_batches['QLD'][:, np.arange(1,y_batches['QLD'].shape[1],5)],
                             'SA': y_batches['SA'][:, np.arange(1,y_batches['SA'].shape[1],5)],
                             'TAS': y_batches['TAS'][:, np.arange(1,y_batches['TAS'].shape[1],5)],
                             'VIC': y_batches['VIC'][:, np.arange(1,y_batches['VIC'].shape[1],5)]}
    
    y_batches_validation_fold[3] = {'NSW': y_batches['NSW'][:, np.arange(2,y_batches['NSW'].shape[1],5)],
                             'QLD': y_batches['QLD'][:, np.arange(2,y_batches['QLD'].shape[1],5)],
                             'SA': y_batches['SA'][:, np.arange(2,y_batches['SA'].shape[1],5)],
                             'TAS': y_batches['TAS'][:, np.arange(2,y_batches['TAS'].shape[1],5)],
                             'VIC': y_batches['VIC'][:, np.arange(2,y_batches['VIC'].shape[1],5)]}
    
    y_batches_validation_fold[4] = {'NSW': y_batches['NSW'][:, np.arange(3,y_batches['NSW'].shape[1],5)],
                             'QLD': y_batches['QLD'][:, np.arange(3,y_batches['QLD'].shape[1],5)],
                             'SA': y_batches['SA'][:, np.arange(3,y_batches['SA'].shape[1],5)],
                             'TAS': y_batches['TAS'][:, np.arange(3,y_batches['TAS'].shape[1],5)],
                             'VIC': y_batches['VIC'][:, np.arange(3,y_batches['VIC'].shape[1],5)]}
    
    y_batches_validation_fold[5] = {'NSW': y_batches['NSW'][:, np.arange(4,y_batches['NSW'].shape[1],5)],
                             'QLD': y_batches['QLD'][:, np.arange(4,y_batches['QLD'].shape[1],5)],
                             'SA': y_batches['SA'][:, np.arange(4,y_batches['SA'].shape[1],5)],
                             'TAS': y_batches['TAS'][:, np.arange(4,y_batches['TAS'].shape[1],5)],
                             'VIC': y_batches['VIC'][:, np.arange(4,y_batches['VIC'].shape[1],5)]}
 
    
    #Making training sets
    x_batches_train_fold[1] = {'NSW': x_batches['NSW'][:, [x for x in np.arange(0,x_batches['NSW'].shape[1]) if x not in np.arange(0,x_batches['NSW'].shape[1],5)] ],
                             'QLD': x_batches['QLD'][:, [x for x in np.arange(0,x_batches['QLD'].shape[1]) if x not in np.arange(0,x_batches['QLD'].shape[1],5)] ],
                             'SA': x_batches['SA'][:, [x for x in np.arange(0,x_batches['SA'].shape[1]) if x not in np.arange(0,x_batches['SA'].shape[1],5)] ],
                             'TAS': x_batches['TAS'][:, [x for x in np.arange(0,x_batches['TAS'].shape[1]) if x not in np.arange(0,x_batches['TAS'].shape[1],5)] ],
                             'VIC': x_batches['VIC'][:, [x for x in np.arange(0,x_batches['VIC'].shape[1]) if x not in np.arange(0,x_batches['VIC'].shape[1],5)] ]}
    
    x_batches_train_fold[2] = {'NSW': x_batches['NSW'][:, [x for x in np.arange(1,x_batches['NSW'].shape[1]) if x not in np.arange(1,x_batches['NSW'].shape[1],5)] ],
                             'QLD': x_batches['QLD'][:, [x for x in np.arange(1,x_batches['QLD'].shape[1]) if x not in np.arange(1,x_batches['QLD'].shape[1],5)] ],
                             'SA': x_batches['SA'][:, [x for x in np.arange(1,x_batches['SA'].shape[1]) if x not in np.arange(1,x_batches['SA'].shape[1],5)] ],
                             'TAS': x_batches['TAS'][:, [x for x in np.arange(1,x_batches['TAS'].shape[1]) if x not in np.arange(1,x_batches['TAS'].shape[1],5)] ],
                             'VIC': x_batches['VIC'][:, [x for x in np.arange(1,x_batches['VIC'].shape[1]) if x not in np.arange(1,x_batches['VIC'].shape[1],5)] ]}
    
    x_batches_train_fold[3] = {'NSW': x_batches['NSW'][:, [x for x in np.arange(2,x_batches['NSW'].shape[1]) if x not in np.arange(2,x_batches['NSW'].shape[1],5)] ],
                             'QLD': x_batches['QLD'][:, [x for x in np.arange(2,x_batches['QLD'].shape[1]) if x not in np.arange(2,x_batches['QLD'].shape[1],5)] ],
                             'SA': x_batches['SA'][:, [x for x in np.arange(2,x_batches['SA'].shape[1]) if x not in np.arange(2,x_batches['SA'].shape[1],5)] ],
                             'TAS': x_batches['TAS'][:, [x for x in np.arange(2,x_batches['TAS'].shape[1]) if x not in np.arange(2,x_batches['TAS'].shape[1],5)] ],
                             'VIC': x_batches['VIC'][:, [x for x in np.arange(2,x_batches['VIC'].shape[1]) if x not in np.arange(2,x_batches['VIC'].shape[1],5)] ]}
    
    x_batches_train_fold[4] = {'NSW': x_batches['NSW'][:, [x for x in np.arange(3,x_batches['NSW'].shape[1]) if x not in np.arange(3,x_batches['NSW'].shape[1],5)] ],
                             'QLD': x_batches['QLD'][:, [x for x in np.arange(3,x_batches['QLD'].shape[1]) if x not in np.arange(3,x_batches['QLD'].shape[1],5)] ],
                             'SA': x_batches['SA'][:, [x for x in np.arange(3,x_batches['SA'].shape[1]) if x not in np.arange(3,x_batches['SA'].shape[1],5)] ],
                             'TAS': x_batches['TAS'][:, [x for x in np.arange(3,x_batches['TAS'].shape[1]) if x not in np.arange(3,x_batches['TAS'].shape[1],5)]],
                             'VIC': x_batches['VIC'][:, [x for x in np.arange(3,x_batches['VIC'].shape[1]) if x not in np.arange(3,x_batches['VIC'].shape[1],5)]]}
    
    x_batches_train_fold[5] = {'NSW': x_batches['NSW'][:, [x for x in np.arange(4,x_batches['NSW'].shape[1]) if x not in np.arange(4,x_batches['NSW'].shape[1],5)] ],
                             'QLD': x_batches['QLD'][:, [x for x in np.arange(4,x_batches['QLD'].shape[1]) if x not in np.arange(4,x_batches['QLD'].shape[1],5)] ],
                             'SA': x_batches['SA'][:, [x for x in np.arange(4,x_batches['SA'].shape[1]) if x not in np.arange(4,x_batches['SA'].shape[1],5)] ],
                             'TAS': x_batches['TAS'][:, [x for x in np.arange(4,x_batches['TAS'].shape[1]) if x not in np.arange(4,x_batches['TAS'].shape[1],5)] ],
                             'VIC': x_batches['VIC'][:, [x for x in np.arange(4,x_batches['VIC'].shape[1]) if x not in np.arange(4,x_batches['VIC'].shape[1],5)] ]}

    y_batches_train_fold[1] = {'NSW': y_batches['NSW'][:, [x for x in np.arange(0,y_batches['NSW'].shape[1]) if x not in np.arange(0,y_batches['NSW'].shape[1],5)] ],
                             'QLD': y_batches['QLD'][:, [x for x in np.arange(0,y_batches['QLD'].shape[1]) if x not in np.arange(0,y_batches['QLD'].shape[1],5)] ],
                             'SA': y_batches['SA'][:, [x for x in np.arange(0,y_batches['SA'].shape[1]) if x not in np.arange(0,y_batches['SA'].shape[1],5)] ],
                             'TAS': y_batches['TAS'][:, [x for x in np.arange(0,y_batches['TAS'].shape[1]) if x not in np.arange(0,y_batches['TAS'].shape[1],5)] ],
                             'VIC': y_batches['VIC'][:, [x for x in np.arange(0,y_batches['VIC'].shape[1]) if x not in np.arange(0,y_batches['VIC'].shape[1],5)] ]}
    
    y_batches_train_fold[2] = {'NSW': y_batches['NSW'][:, [x for x in np.arange(1,y_batches['NSW'].shape[1]) if x not in np.arange(1,y_batches['NSW'].shape[1],5)] ],
                             'QLD': y_batches['QLD'][:, [x for x in np.arange(1,y_batches['QLD'].shape[1]) if x not in np.arange(1,y_batches['QLD'].shape[1],5)] ],
                             'SA': y_batches['SA'][:, [x for x in np.arange(1,y_batches['SA'].shape[1]) if x not in np.arange(1,y_batches['SA'].shape[1],5)] ],
                             'TAS': y_batches['TAS'][:, [x for x in np.arange(1,y_batches['TAS'].shape[1]) if x not in np.arange(1,y_batches['TAS'].shape[1],5)] ],
                             'VIC': y_batches['VIC'][:, [x for x in np.arange(1,y_batches['VIC'].shape[1]) if x not in np.arange(1,y_batches['VIC'].shape[1],5)] ]}
    
    y_batches_train_fold[3] = {'NSW': y_batches['NSW'][:, [x for x in np.arange(2,y_batches['NSW'].shape[1]) if x not in np.arange(2,y_batches['NSW'].shape[1],5)] ],
                             'QLD': y_batches['QLD'][:, [x for x in np.arange(2,y_batches['QLD'].shape[1]) if x not in np.arange(2,y_batches['QLD'].shape[1],5)] ],
                             'SA': y_batches['SA'][:, [x for x in np.arange(2,y_batches['SA'].shape[1]) if x not in np.arange(2,y_batches['SA'].shape[1],5)] ],
                             'TAS': y_batches['TAS'][:, [x for x in np.arange(2,y_batches['TAS'].shape[1]) if x not in np.arange(2,y_batches['TAS'].shape[1],5)] ],
                             'VIC': y_batches['VIC'][:, [x for x in np.arange(2,y_batches['VIC'].shape[1]) if x not in np.arange(2,y_batches['VIC'].shape[1],5)] ]}
    
    y_batches_train_fold[4] = {'NSW': y_batches['NSW'][:, [x for x in np.arange(3,y_batches['NSW'].shape[1]) if x not in np.arange(3,y_batches['NSW'].shape[1],5)] ],
                             'QLD': y_batches['QLD'][:, [x for x in np.arange(3,y_batches['QLD'].shape[1]) if x not in np.arange(3,y_batches['QLD'].shape[1],5)] ],
                             'SA': y_batches['SA'][:, [x for x in np.arange(3,y_batches['SA'].shape[1]) if x not in np.arange(3,y_batches['SA'].shape[1],5)] ],
                             'TAS': y_batches['TAS'][:, [x for x in np.arange(3,y_batches['TAS'].shape[1]) if x not in np.arange(3,y_batches['TAS'].shape[1],5)] ],
                             'VIC': y_batches['VIC'][:, [x for x in np.arange(3,y_batches['VIC'].shape[1]) if x not in np.arange(3,y_batches['VIC'].shape[1],5)] ]}

    y_batches_train_fold[5] = {'NSW': y_batches['NSW'][:, [x for x in np.arange(4,y_batches['NSW'].shape[1]) if x not in np.arange(4,y_batches['NSW'].shape[1],5)] ],
                             'QLD': y_batches['QLD'][:, [x for x in np.arange(4,y_batches['QLD'].shape[1]) if x not in np.arange(4,y_batches['QLD'].shape[1],5)] ],
                             'SA': y_batches['SA'][:, [x for x in np.arange(4,y_batches['SA'].shape[1]) if x not in np.arange(4,y_batches['SA'].shape[1],5)] ],
                             'TAS': y_batches['TAS'][:, [x for x in np.arange(4,y_batches['TAS'].shape[1]) if x not in np.arange(4,y_batches['TAS'].shape[1],5)] ],
                             'VIC': y_batches['VIC'][:, [x for x in np.arange(4,y_batches['VIC'].shape[1]) if x not in np.arange(4,y_batches['VIC'].shape[1],5)] ]}


    for st in state.values():
        for fold in np.arange(1,6):
            print("Train and validation from state ", st, " fold ", fold)
            net = nt.Network([x_size, hidden, y_size], nt.Activation.tanh, nt.QuadraticCost)
            if cso:
                fname = "kernelBias5Fold" + st + ".npy"
                if not path.exists(fname):
                    print("Weights and biases initialization for state ",st ," in progress...")
                    randInt = np.random.randint(x_batches[st].shape[1])
                    net.cso(100,x_batches[st][:, randInt].reshape(x_size,1),
                                    y_batches[st][:, randInt].reshape(y_size,1),
                                    net.multiObjectiveFunction,-0.6,0.6,net.dim ,100)
                    
                    net.set_weight_bias(np.array(net.get_Gbest()))
                    np.save(fname, np.array(net.get_Gbest()))
            
                net.set_weight_bias(np.load(fname))
            
            num_epochs = 1500
            lmbda = 2
            
            if cso:
                fname = "results_"+ st + "_5Fold_" + str(fold) + "CSO"
            else:
                fname = "results_"+ st + "_5Fold_" + str(fold) + "GD" #GD: Gaussian Distribution
            
            evaluation_cost, eval_mape, eval_rmse, eval_mae, training_cost, training_mape, training_rmse, training_mae = net.SGD(
                    x_batches_train_fold[fold][st],y_batches_train_fold[fold][st],
                    num_epochs, 10, 0.01, x_batches_validation_fold[fold][st],
                    y_batches_validation_fold[fold][st],
                    lmbda, monitor_evaluation_cost = True,
                    monitor_evaluation_accuracy = True,
                    monitor_training_cost = True,
                    monitor_training_accuracy = True,
                    output2D = True)
            
            f = open(fname, "w")
            json.dump([evaluation_cost, eval_mape, eval_rmse, eval_mae, training_cost, training_mape, training_rmse, training_mae], f)
            f.close()
            
#            make_plots(fname, num_epochs,
#                       training_cost_xmin = 0,
#                       test_accuracy_xmin = 0,
#                       test_cost_xmin = 0, 
#                       training_accuracy_xmin = 0)
            
       
