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
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit

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
    
#    df['NSW'].iloc[:,0] = nt.estimated_autocorrelation(df['NSW'].iloc[:,0].values)
#    df['QLD'].iloc[:,0] = nt.estimated_autocorrelation(df['QLD'].iloc[:,0].values)
#    df['SA'].iloc[:,0] = nt.estimated_autocorrelation(df['SA'].iloc[:,0].values)
#    df['TAS'].iloc[:,0] = nt.estimated_autocorrelation(df['TAS'].iloc[:,0].values)
#    df['VIC'].iloc[:,0] = nt.estimated_autocorrelation(df['VIC'].iloc[:,0].values)
#    
#    df_test['NSW'].iloc[:,0] = nt.estimated_autocorrelation(df_test['NSW'].iloc[:,0].values)
#    df_test['QLD'].iloc[:,0] = nt.estimated_autocorrelation(df_test['QLD'].iloc[:,0].values)
#    df_test['SA'].iloc[:,0] = nt.estimated_autocorrelation(df_test['SA'].iloc[:,0].values)
#    df_test['TAS'].iloc[:,0] = nt.estimated_autocorrelation(df_test['TAS'].iloc[:,0].values)
#    df_test['VIC'].iloc[:,0] = nt.estimated_autocorrelation(df_test['VIC'].iloc[:,0].values)
    
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
    
#    list_hourly_load_NSW_test = df_test['NSW'].iloc[:,0].values
#    list_hourly_load_QLD_test = df_test['QLD'].iloc[:,0].values
#    list_hourly_load_SA_test = df_test['SA'].iloc[:,0].values
#    list_hourly_load_TAS_test = df_test['TAS'].iloc[:,0].values
#    list_hourly_load_VIC_test = df_test['VIC'].iloc[:,0].values
    
    min_max_scaler = preprocessing.MinMaxScaler()
    
    list_hourly_load_NSW = min_max_scaler.fit_transform(list_hourly_load_NSW.reshape(-1, 1))
    list_hourly_load_QLD = min_max_scaler.fit_transform(list_hourly_load_QLD.reshape(-1, 1))
    list_hourly_load_SA = min_max_scaler.fit_transform(list_hourly_load_SA.reshape(-1, 1))
    list_hourly_load_TAS = min_max_scaler.fit_transform(list_hourly_load_TAS.reshape(-1, 1))
    list_hourly_load_VIC = min_max_scaler.fit_transform(list_hourly_load_VIC.reshape(-1, 1))

#    list_hourly_load_NSW_test = min_max_scaler.fit_transform(list_hourly_load_NSW_test)
#    list_hourly_load_QLD_test = min_max_scaler.fit_transform(list_hourly_load_QLD_test)
#    list_hourly_load_SA_test = min_max_scaler.fit_transform(list_hourly_load_SA_test)
#    list_hourly_load_TAS_test = min_max_scaler.fit_transform(list_hourly_load_TAS_test)
#    list_hourly_load_VIC_test = min_max_scaler.fit_transform(list_hourly_load_VIC_test)
    
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

#    y_test_NSW = np.array(list_hourly_load_NSW_test)
#    y_test_QLD = np.array(list_hourly_load_QLD_test)
#    y_test_SA = np.array(list_hourly_load_SA_test)
#    y_test_TAS = np.array(list_hourly_load_TAS_test)
#    y_test_VIC = np.array(list_hourly_load_VIC_test)
    
    # shuffle the training set (but do not shuffle the test set)
    np.random.shuffle(matrix_load_NSW)
    np.random.shuffle(matrix_load_QLD)
    np.random.shuffle(matrix_load_SA)
    np.random.shuffle(matrix_load_TAS)
    np.random.shuffle(matrix_load_VIC)
    
    # the training set
    X_NSW = matrix_load_NSW[:, :-1]
    X_QLD = matrix_load_QLD[:, :-1]
    X_SA = matrix_load_SA[:, :-1]
    X_TAS = matrix_load_TAS[:, :-1]
    X_VIC = matrix_load_VIC[:, :-1]
    
    X_NSW = X_NSW.transpose().reshape(46,X_NSW.shape[0]).transpose()
    X_QLD = X_QLD.transpose().reshape(46,X_QLD.shape[0]).transpose()
    X_SA = X_SA.transpose().reshape(46,X_SA.shape[0]).transpose()
    X_TAS = X_TAS.transpose().reshape(46,X_TAS.shape[0]).transpose()
    X_VIC = X_VIC.transpose().reshape(46,X_VIC.shape[0]).transpose()
    
    # the last column is the true value to compute the mean-squared-error loss
    y_NSW = matrix_load_NSW[:, -1] 
    y_QLD = matrix_load_QLD[:, -1] 
    y_SA = matrix_load_SA[:, -1] 
    y_TAS = matrix_load_TAS[:, -1] 
    y_VIC = matrix_load_VIC[:, -1] 
    
    y_NSW = y_NSW.flatten()
    y_QLD = y_QLD.flatten() 
    y_SA = y_SA.flatten() 
    y_TAS = y_TAS.flatten() 
    y_VIC = y_VIC.flatten() 
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    evaluation_cost_list_NSW = []
    evaluation_accuracy_list_NSW = []
    training_cost_list_NSW = []
    training_accuracy_list_NSW = []
    
    print("NSW:")
    for train_index, test_index in tscv.split(X_NSW):
        i = 1
        X_train_NSW, X_test_NSW = X_NSW[train_index], X_NSW[test_index]
        y_train_NSW, y_test_NSW = y_NSW[train_index], y_NSW[test_index]
        
        net_NSW = nt.Network([46,10,1], nt.Activation.sigmoid, nt.QuadraticCost)
        dim = sum(w.size + b.size for w,b in zip(net_NSW.weights,net_NSW.biases))
        net_NSW.cso(100,X_train_NSW[0].reshape(46,1),y_train_NSW[0].reshape(1,1),net_NSW.objectiveFunction,-0.6,0.6,dim ,500)
        net_NSW.set_weight_bias(np.array(net_NSW.get_Gbest()))
        
        fname = "results_NSW_" + str(i)
        num_epochs = 500
        lmbda = 2
        
        X_train_NSW = X_train_NSW.transpose().reshape(46, X_train_NSW.shape[0])
        X_test_NSW = X_test_NSW.transpose().reshape(46, X_test_NSW.shape[0])
        evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net_NSW.SGD(
                X_train_NSW,y_train_NSW, num_epochs, 10, 0.01, X_test_NSW, y_test_NSW, 
                lmbda, monitor_evaluation_cost = True,
                monitor_evaluation_accuracy = True,
                monitor_training_cost = True,
                monitor_training_accuracy = True)
        
        evaluation_cost_list_NSW.append(evaluation_cost)
        evaluation_accuracy_list_NSW.append(evaluation_accuracy)
        training_cost_list_NSW.append(training_cost)
        training_accuracy_list_NSW.append(training_accuracy)
        
        f = open(fname, "w")
        json.dump([evaluation_cost, evaluation_accuracy, training_cost, training_accuracy], f)
        f.close()
            
        make_plots(fname, num_epochs, training_set_size = X_test_NSW.shape[1],
                   training_cost_xmin = 0,
                   test_accuracy_xmin = 0,
                   test_cost_xmin = 0, 
                   training_accuracy_xmin = 0)
        i = i+1
        
    evaluation_cost_list_QLD = []
    evaluation_accuracy_list_QLD = []
    training_cost_list_QLD = []
    training_accuracy_list_QLD = [] 
    
    print("QLD:")
    for train_index, test_index in tscv.split(X_QLD):
        i = 1
        X_train_QLD, X_test_QLD = X_QLD[train_index], X_QLD[test_index]
        y_train_QLD, y_test_QLD = y_QLD[train_index], y_QLD[test_index]
        
        net_QLD = nt.Network([46,10,1], nt.Activation.sigmoid, nt.QuadraticCost)
        dim = sum(w.size + b.size for w,b in zip(net_QLD.weights,net_QLD.biases))
        net_QLD.cso(100,X_train_QLD[0].reshape(46,1),y_train_QLD[0].reshape(1,1),net_QLD.objectiveFunction,-0.6,0.6,dim ,500)
        net_QLD.set_weight_bias(np.array(net_QLD.get_Gbest()))
        
        fname = "results_QLD_" + str(i)
        num_epochs = 500
        lmbda = 2
        
        X_train_QLD = X_train_QLD.transpose().reshape(46, X_train_QLD.shape[0])
        X_test_QLD = X_test_QLD.transpose().reshape(46, X_test_QLD.shape[0])
        
        evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net_QLD.SGD(
                X_train_NSW,y_train_NSW, num_epochs, 10, 0.01, X_test_NSW, y_test_NSW, 
                lmbda, monitor_evaluation_cost = True,
                monitor_evaluation_accuracy = True,
                monitor_training_cost = True,
                monitor_training_accuracy = True)
        
        evaluation_cost_list_QLD.append(evaluation_cost)
        evaluation_accuracy_list_QLD.append(evaluation_accuracy)
        training_cost_list_QLD.append(training_cost)
        training_accuracy_list_QLD.append(training_accuracy)
        
        f = open(fname, "w")
        json.dump([evaluation_cost, evaluation_accuracy, training_cost, training_accuracy], f)
        f.close()
            
        make_plots(fname, num_epochs, training_set_size = X_test_QLD.shape[1],
                   training_cost_xmin = 0,
                   test_accuracy_xmin = 0,
                   test_cost_xmin = 0, 
                   training_accuracy_xmin = 0)
        i = i + 1

    evaluation_cost_list_SA = []
    evaluation_accuracy_list_SA = []
    training_cost_list_SA = []
    training_accuracy_list_SA = [] 
    
    print("SA:")    
    for train_index, test_index in tscv.split(X_SA):
        i = 1
        X_train_SA, X_test_SA = X_SA[train_index], X_SA[test_index]
        y_train_SA, y_test_SA = y_SA[train_index], y_SA[test_index]
        
        net_SA = nt.Network([46,10,1], nt.Activation.sigmoid, nt.QuadraticCost)
        dim = sum(w.size + b.size for w,b in zip(net_SA.weights,net_SA.biases))
        net_SA.cso(100,X_train_SA[0].reshape(46,1),y_train_SA[0].reshape(1,1),net_SA.objectiveFunction,-0.6,0.6,dim ,500)
        net_SA.set_weight_bias(np.array(net_SA.get_Gbest()))
        
        fname = "results_SA_" + str(i)
        num_epochs = 500
        lmbda = 2
        
        X_train_SA = X_train_SA.transpose().reshape(46, X_train_SA.shape[0])
        X_test_SA = X_test_SA.transpose().reshape(46, X_test_SA.shape[0])
        
        evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net_SA.SGD(
                X_train_SA,y_train_SA, num_epochs, 10, 0.01, X_test_SA, y_test_SA, 
                lmbda, monitor_evaluation_cost = True,
                monitor_evaluation_accuracy = True,
                monitor_training_cost = True,
                monitor_training_accuracy = True)
        
        evaluation_cost_list_SA.append(evaluation_cost)
        evaluation_accuracy_list_SA.append(evaluation_accuracy)
        training_cost_list_SA.append(training_cost)
        training_accuracy_list_SA.append(training_accuracy)
        
        f = open(fname, "w")
        json.dump([evaluation_cost, evaluation_accuracy, training_cost, training_accuracy], f)
        f.close()
            
        make_plots(fname, num_epochs, training_set_size = X_test_SA.shape[1],
                   training_cost_xmin = 0,
                   test_accuracy_xmin = 0,
                   test_cost_xmin = 0, 
                   training_accuracy_xmin = 0)
        i = i+1


    evaluation_cost_list_TAS = []
    evaluation_accuracy_list_TAS = []
    training_cost_list_TAS = []
    training_accuracy_list_TAS = []
        
    print("TAS:")
    for train_index, test_index in tscv.split(X_TAS):
        i = 1
        X_train_TAS, X_test_TAS = X_TAS[train_index], X_TAS[test_index]
        y_train_TAS, y_test_TAS = y_TAS[train_index], y_TAS[test_index]
        
        net_TAS = nt.Network([46,10,1], nt.Activation.sigmoid, nt.QuadraticCost)
        dim = sum(w.size + b.size for w,b in zip(net_TAS.weights,net_TAS.biases))
        net_TAS.cso(100,X_train_TAS[0].reshape(46,1),y_train_TAS[0].reshape(1,1),net_TAS.objectiveFunction,-0.6,0.6,dim ,500)
        net_TAS.set_weight_bias(np.array(net_TAS.get_Gbest()))
        
        fname = "results_TAS_" + str(i)
        num_epochs = 500
        lmbda = 2
        
        X_train_TAS = X_train_TAS.transpose().reshape(46, X_train_TAS.shape[0])
        X_test_TAS = X_test_TAS.transpose().reshape(46, X_test_TAS.shape[0])
        
        evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net_TAS.SGD(
                X_train_TAS,y_train_TAS, num_epochs, 10, 0.01, X_test_TAS, y_test_TAS, 
                lmbda, monitor_evaluation_cost = True,
                monitor_evaluation_accuracy = True,
                monitor_training_cost = True,
                monitor_training_accuracy = True)
        
        evaluation_cost_list_TAS.append(evaluation_cost)
        evaluation_accuracy_list_TAS.append(evaluation_accuracy)
        training_cost_list_TAS.append(training_cost)
        training_accuracy_list_TAS.append(training_accuracy)
        
        f = open(fname, "w")
        json.dump([evaluation_cost, evaluation_accuracy, training_cost, training_accuracy], f)
        f.close()
            
        make_plots(fname, num_epochs, training_set_size = X_test_TAS.shape[1],
                   training_cost_xmin = 0,
                   test_accuracy_xmin = 0,
                   test_cost_xmin = 0, 
                   training_accuracy_xmin = 0)  
        i = i + 1


    evaluation_cost_list_VIC = []
    evaluation_accuracy_list_VIC = []
    training_cost_list_VIC = []
    training_accuracy_list_VIC = []
        
    print("VIC:")
    for train_index, test_index in tscv.split(X_VIC):
        i = 1
        X_train_VIC, X_test_VIC = X_VIC[train_index], X_VIC[test_index]
        y_train_VIC, y_test_VIC = y_VIC[train_index], y_VIC[test_index]
        
        net_VIC = nt.Network([46,10,1], nt.Activation.sigmoid, nt.QuadraticCost)
        dim = sum(w.size + b.size for w,b in zip(net_VIC.weights,net_VIC.biases))
        net_VIC.cso(100,X_train_VIC[0].reshape(46,1),y_train_VIC[0].reshape(1,1),net_VIC.objectiveFunction,-0.6,0.6,dim ,500)
        net_VIC.set_weight_bias(np.array(net_VIC.get_Gbest()))
        
        fname = "results_VIC_" + str(i)
        num_epochs = 500
        lmbda = 2
        
        X_train_VIC = X_train_VIC.transpose().reshape(46, X_train_VIC.shape[0])
        X_test_VIC = X_test_VIC.transpose().reshape(46, X_test_VIC.shape[0])
        
        evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net_VIC.SGD(
                X_train_VIC,y_train_VIC, num_epochs, 10, 0.01, X_test_VIC, y_test_VIC, 
                lmbda, monitor_evaluation_cost = True,
                monitor_evaluation_accuracy = True,
                monitor_training_cost = True,
                monitor_training_accuracy = True)

        evaluation_cost_list_VIC.append(evaluation_cost)
        evaluation_accuracy_list_VIC.append(evaluation_accuracy)
        training_cost_list_VIC.append(training_cost)
        training_accuracy_list_VIC.append(training_accuracy)        
        
        f = open(fname, "w")
        json.dump([evaluation_cost, evaluation_accuracy, training_cost, training_accuracy], f)
        f.close()
            
        make_plots(fname, num_epochs, training_set_size = X_test_VIC.shape[1],
                   training_cost_xmin = 0,
                   test_accuracy_xmin = 0,
                   test_cost_xmin = 0, 
                   training_accuracy_xmin = 0) 
        i = i + 1

    #eemd = EEMD()
    #eIMFs_NSW = eemd(df['NSW'].iloc[:,0].values)





    
    
    
      

       
