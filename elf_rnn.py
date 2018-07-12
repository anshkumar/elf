import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data pre-processing
state = {0: 'NSW', 1: 'QLD', 2: 'SA', 3: 'TAS', 4: 'VIC'}
year = {0: '2015', 1: '2016', 2: '2017'}
#year = {0: '2017'}

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
    dataset = pd.read_csv('./datasets/test/' + st + '/PRICE_AND_DEMAND_201801_' + st + '1.csv')
    df_test[st] = df_test[st].append(dataset.iloc[:,1:3])
    df_test[st] = df_test[st].set_index('SETTLEMENTDATE')


plt.plot(df['NSW'].iloc[:,0].values)
plt.show()
plt.plot(df['QLD'].iloc[:,0].values)
plt.show()
plt.plot(df['SA'].iloc[:,0].values)
plt.show()
plt.plot(df['TAS'].iloc[:,0].values)
plt.show()
plt.plot(df['VIC'].iloc[:,0].values)
plt.show()


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

num_periods = 20
f_horizon = 1  #forecast horizon

""" Making the dataset size divisible by num_period """
x_data_nsw = TS_NSW[:(len(TS_NSW) - f_horizon -((len(TS_NSW)-f_horizon) % num_periods))] 
x_data_qld = TS_QLD[:(len(TS_QLD)- f_horizon - ((len(TS_QLD)-f_horizon) % num_periods))]
x_data_sa = TS_SA[:(len(TS_SA)- f_horizon -((len(TS_SA)-f_horizon) % num_periods))]
x_data_tas = TS_TAS[:(len(TS_TAS)- f_horizon -((len(TS_TAS)-f_horizon) % num_periods))]
x_data_vic = TS_VIC[:(len(TS_VIC)-f_horizon-((len(TS_VIC)-f_horizon) % num_periods))] 
""" Making our training dataset with batch size of num_period """
x_batches = {'NSW': x_data_nsw.reshape(-1, num_periods, 1),
             'QLD': x_data_qld.reshape(-1, num_periods, 1),
             'SA': x_data_sa.reshape(-1, num_periods, 1),
             'TAS': x_data_tas.reshape(-1, num_periods, 1),
             'VIC': x_data_vic.reshape(-1, num_periods, 1)}


y_data_nsw = TS_NSW[1:(len(TS_NSW)-f_horizon-((len(TS_NSW)-f_horizon) % num_periods)+f_horizon)]
y_data_qld = TS_QLD[1:(len(TS_QLD)-f_horizon-((len(TS_QLD)-f_horizon) % num_periods)+f_horizon)]
y_data_sa = TS_SA[1:(len(TS_SA)-f_horizon-((len(TS_SA)-f_horizon) % num_periods)+f_horizon)]
y_data_tas = TS_TAS[1:(len(TS_TAS)-f_horizon-((len(TS_TAS)-f_horizon) % num_periods)+f_horizon)]
y_data_vic = TS_VIC[1:(len(TS_VIC)-f_horizon-((len(TS_VIC)-f_horizon) % num_periods)+f_horizon)]

y_batches = {'NSW': y_data_nsw.reshape(-1, num_periods, 1),
                 'QLD': y_data_qld.reshape(-1, num_periods, 1),
                 'SA': y_data_sa.reshape(-1, num_periods, 1),
                 'TAS': y_data_tas.reshape(-1, num_periods, 1),
                 'VIC': y_data_vic.reshape(-1, num_periods, 1)}

""" Making the dataset size divisible by num_period """
x_data_nsw_test = TS_NSW_test[:(len(TS_NSW_test)-f_horizon-((len(TS_NSW_test)-f_horizon) % num_periods))] 
x_data_qld_test = TS_QLD_test[:(len(TS_QLD_test)-f_horizon-((len(TS_QLD_test)-f_horizon) % num_periods))]
x_data_sa_test = TS_SA_test[:(len(TS_SA_test)-f_horizon-((len(TS_SA_test)-f_horizon) % num_periods))]
x_data_tas_test = TS_TAS_test[:(len(TS_TAS_test)-f_horizon-((len(TS_TAS_test)-f_horizon) % num_periods))]
x_data_vic_test = TS_VIC_test[:(len(TS_VIC_test)-f_horizon-((len(TS_VIC_test)-f_horizon) % num_periods))] 
""" Making our training dataset with batch size of num_period """
x_batches_test = {'NSW': x_data_nsw_test.reshape(-1, num_periods, 1),
             'QLD': x_data_qld_test.reshape(-1, num_periods, 1),
             'SA': x_data_sa_test.reshape(-1, num_periods, 1),
             'TAS': x_data_tas_test.reshape(-1, num_periods, 1),
             'VIC': x_data_vic_test.reshape(-1, num_periods, 1)}


y_data_nsw_test = TS_NSW_test[1:(len(TS_NSW_test)-(len(TS_NSW_test) % num_periods)+f_horizon)]
y_data_qld_test = TS_QLD_test[1:(len(TS_QLD_test)-(len(TS_QLD_test) % num_periods)+f_horizon)]
y_data_sa_test = TS_SA_test[1:(len(TS_SA_test)-(len(TS_SA_test) % num_periods)+f_horizon)]
y_data_tas_test = TS_TAS_test[1:(len(TS_TAS_test)-(len(TS_TAS_test) % num_periods)+f_horizon)]
y_data_vic_test = TS_VIC_test[1:(len(TS_VIC_test)-(len(TS_VIC_test) % num_periods)+f_horizon)]

y_batches_test = {'NSW': y_data_nsw_test.reshape(-1, num_periods, 1),
                 'QLD': y_data_qld_test.reshape(-1, num_periods, 1),
                 'SA': y_data_sa_test.reshape(-1, num_periods, 1),
                 'TAS': y_data_tas_test.reshape(-1, num_periods, 1),
                 'VIC': y_data_vic_test.reshape(-1, num_periods, 1)}

#RNN designning
tf.reset_default_graph()

inputs = 1	#input vector size
hidden = 100	
output = 1	#output vector size

X = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])


basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

learning_rate = 0.001   #small learning rate so we don't overshoot the minimum

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])          #shape of results
 
loss = tf.reduce_mean(tf.square(outputs - y))    #define the cost function which evaluates the quality of our model
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #gradient descent method
training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 

init = tf.global_variables_initializer()           #initialize all the variables
epochs = 1000     #number of iterations or training cycles, includes both the FeedFoward and Backpropogation
mape = []

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

y_pred = {'NSW': [], 'QLD': [], 'SA': [], 'TAS': [], 'VIC': []}

for st in state.values():
    print("State: ", st, end='\n')
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        init.run()
        for ep in range(epochs):
            sess.run(training_op, feed_dict={X: x_batches[st], y: y_batches[st]})
            if ep % 100 == 0:
                mse = loss.eval(feed_dict={X: x_batches[st], y: y_batches[st]})
                print(ep, "MSE:", mse)
        y_pred[st] = sess.run(outputs, feed_dict={X: x_batches_test[st]})
    print("\n")

for st in state.values():
    print("Mape for ", st, " = ", mean_absolute_percentage_error(y_batches_test[st],y_pred[st]), end = '\n')    
    plt.title("Forecast vs Actual", fontsize=14)
    plt.plot(pd.Series(np.ravel(y_batches_test[st])), "b.", markersize=2, label="Actual")
    plt.plot(pd.Series(np.ravel(y_pred[st])), "r.", markersize=2, label="Forecast")
    plt.legend(loc="upper left")
    plt.xlabel("Time Periods")
    plt.show()


