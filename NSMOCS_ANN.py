import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import swarm as sm
from math import sqrt

#State and year to use for training and testing
#state = {0: 'NSW', 1: 'QLD', 2: 'SA', 3: 'TAS', 4: 'VIC'}
state = {0: 'NSW'}
#year = {0: '2015', 1: '2016', 2: '2017'}
year = {0: '2015'}
year_test = {0: '2016'}

#forecasting parameters
num_periods = 12
f_horizon = 24  #forecast horizon

hidden = 10

#Training and testing batches
x_batches = {}
y_batches = {} 
x_batches_test = {}
y_batches_test = {}

#parameters for 5 fold validation 
set_size = 84 
x_size = 36
y_size = 48
x_batches_validation_fold ={}
y_batches_validation_fold ={}
x_batches_train_fold = {}
y_batches_train_fold = {}

def loadData5Folds():
    global x_batches
    global y_batches
    global x_batches_test
    global y_batches_test
    global x_batches_validation_fold
    global y_batches_validation_fold
    global x_batches_train_fold
    global y_batches_train_fold
    
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
    x_batches = {'NSW': x_data_nsw.reshape(-1, num_periods).transpose(),
                 'QLD': x_data_qld.reshape(-1, num_periods).transpose(),
                 'SA': x_data_sa.reshape(-1, num_periods).transpose(),
                 'TAS': x_data_tas.reshape(-1, num_periods).transpose(),
                 'VIC': x_data_vic.reshape(-1, num_periods).transpose()}
    
    y_data_nsw = TS_NSW[f_horizon:(len(TS_NSW)-f_horizon-((len(TS_NSW)-f_horizon) % num_periods)+f_horizon)]
    y_data_qld = TS_QLD[f_horizon:(len(TS_QLD)-f_horizon-((len(TS_QLD)-f_horizon) % num_periods)+f_horizon)]
    y_data_sa = TS_SA[f_horizon:(len(TS_SA)-f_horizon-((len(TS_SA)-f_horizon) % num_periods)+f_horizon)]
    y_data_tas = TS_TAS[f_horizon:(len(TS_TAS)-f_horizon-((len(TS_TAS)-f_horizon) % num_periods)+f_horizon)]
    y_data_vic = TS_VIC[f_horizon:(len(TS_VIC)-f_horizon-((len(TS_VIC)-f_horizon) % num_periods)+f_horizon)]
    
    y_batches = {'NSW': y_data_nsw.reshape(-1, num_periods).transpose(),
                     'QLD': y_data_qld.reshape(-1, num_periods).transpose(),
                     'SA': y_data_sa.reshape(-1, num_periods).transpose(),
                     'TAS': y_data_tas.reshape(-1, num_periods).transpose(),
                     'VIC': y_data_vic.reshape(-1, num_periods).transpose()}
    
    """ Making the dataset size divisible by num_period """
    x_data_nsw_test = TS_NSW_test[:(len(TS_NSW_test)-f_horizon-((len(TS_NSW_test)-f_horizon) % num_periods))] 
    x_data_qld_test = TS_QLD_test[:(len(TS_QLD_test)-f_horizon-((len(TS_QLD_test)-f_horizon) % num_periods))]
    x_data_sa_test = TS_SA_test[:(len(TS_SA_test)-f_horizon-((len(TS_SA_test)-f_horizon) % num_periods))]
    x_data_tas_test = TS_TAS_test[:(len(TS_TAS_test)-f_horizon-((len(TS_TAS_test)-f_horizon) % num_periods))]
    x_data_vic_test = TS_VIC_test[:(len(TS_VIC_test)-f_horizon-((len(TS_VIC_test)-f_horizon) % num_periods))] 
    """ Making our training dataset with batch size of num_period """
    x_batches_test = {'NSW': x_data_nsw_test.reshape(-1, num_periods).transpose(),
                 'QLD': x_data_qld_test.reshape(-1, num_periods).transpose(),
                 'SA': x_data_sa_test.reshape(-1, num_periods).transpose(),
                 'TAS': x_data_tas_test.reshape(-1, num_periods).transpose(),
                 'VIC': x_data_vic_test.reshape(-1, num_periods).transpose()}
    
    y_data_nsw_test = TS_NSW_test[f_horizon:(len(TS_NSW_test)-(len(TS_NSW_test) % num_periods)+f_horizon)]
    y_data_qld_test = TS_QLD_test[f_horizon:(len(TS_QLD_test)-(len(TS_QLD_test) % num_periods)+f_horizon)]
    y_data_sa_test = TS_SA_test[f_horizon:(len(TS_SA_test)-(len(TS_SA_test) % num_periods)+f_horizon)]
    y_data_tas_test = TS_TAS_test[f_horizon:(len(TS_TAS_test)-(len(TS_TAS_test) % num_periods)+f_horizon)]
    y_data_vic_test = TS_VIC_test[f_horizon:(len(TS_VIC_test)-(len(TS_VIC_test) % num_periods)+f_horizon)]
    
    y_batches_test = {'NSW': y_data_nsw_test.reshape(-1, num_periods).transpose(),
                     'QLD': y_data_qld_test.reshape(-1, num_periods).transpose(),
                     'SA': y_data_sa_test.reshape(-1, num_periods).transpose(),
                     'TAS': y_data_tas_test.reshape(-1, num_periods).transpose(),
                     'VIC': y_data_vic_test.reshape(-1, num_periods).transpose()}

def initializeKernelBias(st):
    #Swarm intelligence to get initial weights and biases 
    print("Weights and biases initialization in progress...")
    initializer = sm.Swarm([num_periods,hidden, num_periods], sm.Activation.relu)
    initializer.cso(100,x_batches[st][:,0].reshape(num_periods,1),
                    y_batches[st][:,0].reshape(num_periods,1),
                    initializer.multiObjectiveFunction,-0.6,0.6,initializer.dim ,100)
    
    initializer.set_weight_bias(np.array(initializer.get_Gbest()))
    
    fname = "kernelBias" + st + ".npy"
    np.save(fname, [initializer.weights, initializer.biases])

def loadKernelBias(st):
    fname = "kernelBias" + st +".npy"
    return np.load(fname)

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
    ax.set_title('Accuracy (%) on the test data')
    plt.show()
    
def plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs), 
            [accuracy
             for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the training data')
    plt.show()
    
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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

def tensorGraph(initState = 'NSW'):
    weights_obj, biases_obj = loadKernelBias(initState)
    
    weights = [w for w in weights_obj]
    biases = [b for b in biases_obj]
    
    #RNN designning
    tf.reset_default_graph()
    
    inputs = num_periods	#input vector size
    
    output = num_periods	#output vector size
    learning_rate = 0.01
    
    x = tf.placeholder(tf.float32, [inputs, None])
    y = tf.placeholder(tf.float32, [output, None])

    weights = {
        'hidden': tf.Variable(tf.cast(weights[0], tf.float32)),
        'output': tf.Variable(tf.cast(weights[1], tf.float32))
    }
    
    biases = {
        'hidden': tf.Variable(tf.cast(biases[0],tf.float32)),
        'output': tf.Variable(tf.cast(biases[1],tf.float32))
    }
    
    hidden_layer = tf.add(tf.matmul(weights['hidden'], x), biases['hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)
    
    output_layer = tf.matmul(weights['output'], hidden_layer) + biases['output']
    
    loss = tf.reduce_mean(tf.square(output_layer - y))    #define the cost function which evaluates the quality of our model
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
    training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 
    
    init = tf.global_variables_initializer()           #initialize all the variables
    epochs = 3000     #number of iterations or training cycles, includes both the FeedFoward and Backpropogation
    
    y_pred = {'NSW': [], 'QLD': [], 'SA': [], 'TAS': [], 'VIC': []}
    
    print("Training the ANN...")
    for st in state.values():
            print("State: ", st, end='\n')
            with tf.Session() as sess:
                init.run()
                cost_training = []
                cost_test = []
                error_train = []
                error_test = []
                for ep in range(epochs):
                    sess.run(training_op, feed_dict={x: x_batches[st], y: y_batches[st]})
                    cost_training.append(loss.eval(feed_dict={x: x_batches[st], y: y_batches[st]}))
                    cost_test.append(loss.eval(feed_dict={x: x_batches_test[st], y: y_batches_test[st]}))
                    
                    pred = sess.run(output_layer, feed_dict={x: x_batches[st]})
                    error_train.append(mean_absolute_percentage_error(y_batches[st],pred))
                    pred = sess.run(output_layer, feed_dict={x: x_batches_test[st]})
                    error_test.append(mean_absolute_percentage_error(y_batches_test[st],pred))
                    if ep % 1000 == 0:
                        print("Epoch: ", ep)
                print("Cost for state ", st)
                plot_training_cost(cost_training, epochs, 1000)
                plot_test_cost(cost_test, epochs, 1000)
                print("\n")
                print("Error for state ", st)
                plot_test_accuracy(error_test, epochs, 1000)
                plot_training_accuracy(error_train, epochs, 1000, x_batches[st].shape[0])
                y_pred[st] = sess.run(output_layer, feed_dict={x: x_batches_test[st]})
            print("\n")
    
    for st in state.values():
        print("Mape for ", st, " = ", mean_absolute_percentage_error(y_batches_test[st],y_pred[st]), end = '\n')    
        plt.title("Forecast vs Actual", fontsize=14)
        plt.plot(pd.Series(np.ravel(y_batches_test[st])), "b.", markersize=2, label="Actual")
        plt.plot(pd.Series(np.ravel(y_pred[st])), "r.", markersize=2, label="Forecast")
        plt.legend(loc="upper left")
        plt.xlabel("Time Periods")
        plt.show()

def initializeKernelBias5Fold(st):
    #Swarm intelligence to get initial weights and biases 
    print("Weights and biases initialization in progress...")
    initializer = sm.Swarm([x_size, hidden, y_size], sm.Activation.relu)
    initializer.cso(100,x_batches[st][:,0].reshape(x_size,1),
                    y_batches[st][:,0].reshape(y_size,1),
                    initializer.multiObjectiveFunction,-0.6,0.6,initializer.dim ,100)
    
    initializer.set_weight_bias(np.array(initializer.get_Gbest()))
    
    fname = "kernelBias5Fold" + st + ".npy"
    np.save(fname, [initializer.weights, initializer.biases])

def loadKernelBias5Fold(st):
    fname = "kernelBias5Fold" + st +".npy"
    return np.load(fname)

def tensorGraph5Fold(initState = 'NSW'):
    weights_obj, biases_obj = loadKernelBias5Fold(initState)
    
    weights = [tf.convert_to_tensor(w, dtype=tf.float32) for w in weights_obj]
    biases = [tf.convert_to_tensor(b, dtype=tf.float32) for b in biases_obj]
    
    #RNN designning
    #tf.reset_default_graph()
    
    inputs = x_size	#input vector size
    output = y_size	#output vector size
    learning_rate = 0.01
    
    x = tf.placeholder(tf.float32, [inputs, None])
    y = tf.placeholder(tf.float32, [output, None])

    #L2 regulizer
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.2)
    weights = {
        'hidden': tf.get_variable("w_hidden", initializer = weights[0], regularizer=regularizer),
        'output': tf.get_variable("w_output", initializer = weights[1], regularizer=regularizer)
    }
    
    biases = {
        'hidden': tf.get_variable("b_hidden", initializer = biases[0]),
        'output': tf.get_variable("b_output", initializer = biases[1])
    }
    
    hidden_layer = tf.add(tf.matmul(weights['hidden'], x), biases['hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)
    
    output_layer = tf.matmul(weights['output'], hidden_layer) + biases['output']
    
    loss = tf.reduce_mean(tf.square(output_layer - y))    #define the cost function which evaluates the quality of our model
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
    training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 
    
    #L2 regulizer
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    loss += reg_term
    
    init = tf.global_variables_initializer()           #initialize all the variables
    epochs = 2500     #number of iterations or training cycles, includes both the FeedFoward and Backpropogation
    
    pred = {'NSW': [], 'QLD': [], 'SA': [], 'TAS': [], 'VIC': []}
    y_pred = {1: pred, 2: pred, 3: pred, 4: pred, 5: pred}
    
    print("Training the ANN...")
    for st in state.values():
        for fold in np.arange(1,2):
            print("State: ", st, end='\n')
            print("Fold : ", fold)
            
            with tf.Session() as sess:
                init.run()
                cost_training = []
                cost_test = []
                error_train = []
                error_test = []
                for ep in range(epochs):
                    sess.run(training_op, feed_dict={x: x_batches_train_fold[fold][st], y: y_batches_train_fold[fold][st]})
                    cost_training.append(loss.eval(feed_dict={x: x_batches_train_fold[fold][st], y: y_batches_train_fold[fold][st]}))
                    cost_test.append(loss.eval(feed_dict={x: x_batches_validation_fold[fold][st], y: y_batches_validation_fold[fold][st]}))
                    
                    #MAPE Error
                    pred = sess.run(output_layer, feed_dict={x: x_batches_train_fold[fold][st]})
                    error_train.append(mean_absolute_percentage_error(y_batches_train_fold[fold][st],pred))
                    pred = sess.run(output_layer, feed_dict={x: x_batches_validation_fold[fold][st]})
                    error_test.append(mean_absolute_percentage_error(y_batches_validation_fold[fold][st],pred))
                    
                    #RMSE
#                    error_train.append(sqrt(cost_training[-1]))
#                    error_test.append(sqrt(cost_test[-1]))
                    if ep % 1000 == 0:
                        print("Epoch: ", ep)
                print("Cost for state ", st)
                plot_training_cost(cost_training, epochs, 1000)
                plot_test_cost(cost_test, epochs, 1000)
                print("\n")
                print("Error for state ", st)
                plot_test_accuracy(error_test, epochs, 1000)
                plot_training_accuracy(error_train, epochs, 1000, x_batches_train_fold[fold][st].shape[0])
                y_pred[fold][st] = sess.run(output_layer, feed_dict={x: x_batches_validation_fold[fold][st]})
            print("\n")
    
    for st in state.values():
        for fold in np.arange(1,6):
            print("Mape for ", st, "; fold ",fold,  " = ", mean_absolute_percentage_error(y_batches_validation_fold[fold][st],y_pred[fold][st]), end = '\n')    
            plt.title("Forecast vs Actual", fontsize=14)
            plt.plot(pd.Series(np.ravel(y_batches_validation_fold[fold][st])), "b.", markersize=2, label="Actual")
            plt.plot(pd.Series(np.ravel(y_pred[fold][st])), "r.", markersize=2, label="Forecast")
            plt.legend(loc="upper left")
            plt.xlabel("Time Periods")
            plt.show()

