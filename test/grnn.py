# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from neupy import algorithms, estimators

#State and year to use for training and testing
state = {0: 'NSW', 1: 'QLD', 2: 'SA', 3: 'TAS', 4: 'VIC'}
#state = {0: 'NSW'}
#year = {0: '2015', 1: '2016', 2: '2017'}
year = {0: '2015'}
year_test = {0: '2016', 1: '2017'}

#forecasting parameters
num_periods = 12
f_horizon = 24  #forecast horizon

#Training and testing batches
x_batches = {}
y_batches = {} 
x_batches_test = {}
y_batches_test = {}

def loadData():
    global x_batches
    global y_batches
    global x_batches_test
    global y_batches_test
    
    df_nsw = pd.DataFrame()
    df_qld = pd.DataFrame()
    df_sa = pd.DataFrame()
    df_tas = pd.DataFrame()
    df_vic = pd.DataFrame()
    
    df_nsw_test = pd.DataFrame()
    df_qld_test = pd.DataFrame()
    df_sa_test = pd.DataFrame()
    df_tas_test = pd.DataFrame()
    df_vic_test = pd.DataFrame()
    
    df = {'NSW': df_nsw, 'QLD': df_qld, 'SA': df_sa, 'TAS': df_tas, 'VIC': df_vic}
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
        for ye in year_test.values():
            for mn in range(1,13):
                if mn < 10:            
                    dataset = pd.read_csv('./datasets/train/' + st + '/PRICE_AND_DEMAND_' + ye + '0' + str(mn) +'_' + st + '1.csv')
                else:
                    dataset = pd.read_csv('./datasets/train/' + st + '/PRICE_AND_DEMAND_' + ye + str(mn) +'_' + st + '1.csv')
                df_test[st] = df_test[st].append(dataset.iloc[:,1:3])
        df_test[st] = df_test[st].set_index('SETTLEMENTDATE')
    
    #for st in state.values():
    #    dataset = pd.read_csv('./datasets/test/' + st + '/PRICE_AND_DEMAND_201801_' + st + '1.csv')
    #    df_test[st] = df_test[st].append(dataset.iloc[:,1:3])
    #    df_test[st] = df_test[st].set_index('SETTLEMENTDATE')
    
    TS_NSW = np.array(df['NSW'])
    TS_QLD = np.array(df['QLD'])
    TS_SA = np.array(df['SA'])
    TS_TAS = np.array(df['TAS'])
    TS_VIC = np.array(df['VIC'])
    
    TS_NSW_test = np.array(df_test['NSW'])
    TS_QLD_test = np.array(df_test['QLD'])
    TS_SA_test = np.array(df_test['SA'])
    TS_TAS_test = np.array(df_test['TAS'])
    TS_VIC_test = np.array(df_test['VIC'])
    
    """ Making the dataset size divisible by num_period """
    x_data_nsw = TS_NSW[:(len(TS_NSW) - f_horizon -((len(TS_NSW)-f_horizon) % num_periods))] 
    x_data_qld = TS_QLD[:(len(TS_QLD)- f_horizon - ((len(TS_QLD)-f_horizon) % num_periods))]
    x_data_sa = TS_SA[:(len(TS_SA)- f_horizon -((len(TS_SA)-f_horizon) % num_periods))]
    x_data_tas = TS_TAS[:(len(TS_TAS)- f_horizon -((len(TS_TAS)-f_horizon) % num_periods))]
    x_data_vic = TS_VIC[:(len(TS_VIC)-f_horizon-((len(TS_VIC)-f_horizon) % num_periods))] 
    """ Making our training dataset with batch size of num_period """
    x_batches = {'NSW': x_data_nsw,
                 'QLD': x_data_qld,
                 'SA': x_data_sa,
                 'TAS': x_data_tas,
                 'VIC': x_data_vic}
    
    y_data_nsw = TS_NSW[f_horizon:(len(TS_NSW)-f_horizon-((len(TS_NSW)-f_horizon) % num_periods)+f_horizon)]
    y_data_qld = TS_QLD[f_horizon:(len(TS_QLD)-f_horizon-((len(TS_QLD)-f_horizon) % num_periods)+f_horizon)]
    y_data_sa = TS_SA[f_horizon:(len(TS_SA)-f_horizon-((len(TS_SA)-f_horizon) % num_periods)+f_horizon)]
    y_data_tas = TS_TAS[f_horizon:(len(TS_TAS)-f_horizon-((len(TS_TAS)-f_horizon) % num_periods)+f_horizon)]
    y_data_vic = TS_VIC[f_horizon:(len(TS_VIC)-f_horizon-((len(TS_VIC)-f_horizon) % num_periods)+f_horizon)]
    
    y_batches = {'NSW': y_data_nsw,
                     'QLD': y_data_qld,
                     'SA': y_data_sa,
                     'TAS': y_data_tas,
                     'VIC': y_data_vic}
    
    """ Making the dataset size divisible by num_period """
    x_data_nsw_test = TS_NSW_test[:(len(TS_NSW_test)-f_horizon-((len(TS_NSW_test)-f_horizon) % num_periods))] 
    x_data_qld_test = TS_QLD_test[:(len(TS_QLD_test)-f_horizon-((len(TS_QLD_test)-f_horizon) % num_periods))]
    x_data_sa_test = TS_SA_test[:(len(TS_SA_test)-f_horizon-((len(TS_SA_test)-f_horizon) % num_periods))]
    x_data_tas_test = TS_TAS_test[:(len(TS_TAS_test)-f_horizon-((len(TS_TAS_test)-f_horizon) % num_periods))]
    x_data_vic_test = TS_VIC_test[:(len(TS_VIC_test)-f_horizon-((len(TS_VIC_test)-f_horizon) % num_periods))] 
    """ Making our training dataset with batch size of num_period """
    x_batches_test = {'NSW': x_data_nsw_test,
                 'QLD': x_data_qld_test,
                 'SA': x_data_sa_test,
                 'TAS': x_data_tas_test,
                 'VIC': x_data_vic_test}
    
    y_data_nsw_test = TS_NSW_test[f_horizon:(len(TS_NSW_test)-(len(TS_NSW_test) % num_periods)+f_horizon)]
    y_data_qld_test = TS_QLD_test[f_horizon:(len(TS_QLD_test)-(len(TS_QLD_test) % num_periods)+f_horizon)]
    y_data_sa_test = TS_SA_test[f_horizon:(len(TS_SA_test)-(len(TS_SA_test) % num_periods)+f_horizon)]
    y_data_tas_test = TS_TAS_test[f_horizon:(len(TS_TAS_test)-(len(TS_TAS_test) % num_periods)+f_horizon)]
    y_data_vic_test = TS_VIC_test[f_horizon:(len(TS_VIC_test)-(len(TS_VIC_test) % num_periods)+f_horizon)]
    
    y_batches_test = {'NSW': y_data_nsw_test,
                     'QLD': y_data_qld_test,
                     'SA': y_data_sa_test,
                     'TAS': y_data_tas_test,
                     'VIC': y_data_vic_test}

def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
if __name__ == "__main__":
    loadData()
    nw = algorithms.GRNN(std=0.1, verbose=False)
    for st in state.values():
        nw.train(x_batches[st], y_batches[st])
        y_predicted = nw.predict(x_batches_test[st])
        pred = np.nan_to_num(y_predicted)
        print("MAPE for state ", st, mean_absolute_percentage_error(y_batches_test[st], pred))
        
        