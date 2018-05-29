# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:16:25 2018

@author: ansh
"""

import pandas as pd
import numpy as np
#from PyEMD import EEMD
import matplotlib.pyplot as plt
from math import gamma, pi, sin
from random import normalvariate, randint, random

import intelligence

class QuadraticCost(object):
    @staticmethod
    def fn(a,y):
        return 0.5*np.linalg.norm(a-y)**2
        
    @staticmethod
    def delta(activation, z, a, y):
        if(activation == Activation.sigmoid):
            return (a-y)*Activation.sigmoid_prime(z)
        elif(activation == Activation.tanh):
            return (a-y)*Activation.sigmoid_prime(z)    # TODO: update for tanh
        elif(activation == Activation.relu):
            return (a-y)*(z > 0)

class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(activation, z, a, y):
        if(activation == Activation.sigmoid):
            return a-y
        else:
            return (a - y + 1 - y/a)
        
class Activation(object):
    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))
    
    @staticmethod
    def sigmoid_prime(z):
        return Activation.sigmoid(z)*(1 - Activation.sigmoid(z))

    @staticmethod        
    def tanh(z):
        return 2*Activation.sigmoid(2*z) - 1
        
    @staticmethod
    def relu(z):
        return z * (z > 0)

class Network(intelligence.sw):
    def __init__(self, sizes, activate, cost = CrossEntropyCost, backprop = True):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        self.activate = activate
        self.isbackpropUsed = backprop
        
        if(backprop == False):
            self.x = np.array([])
            for w in self.weights:
                self.x = np.insert(self.x, self.x.size, w.reshape(w.size))
            for b in self.biases:
                self.x = np.insert(self.x, self.x.size, b.reshape(b.size))
                
    def default_weight_initializer(self):
        self.biases = [np.random.randn(x,1) for x in self.sizes[1:]]
        self.weights = [np.random.randn(x,y)/np.sqrt(x) 
                        for x,y in zip(self.sizes[1:], self.sizes[:-1])]
    
    def large_weight_initializer(self):
#        self.biases = [np.random.randn(x,1) for x in self.sizes[1:]]
        self.weights = [np.random.randn(x,y) 
                        for x,y in zip(self.sizes[1:], self.sizes[:-1])]
    
    def feedforward(self,a):
        if(self.isbackpropUsed == True):
            for w in self.weights:
    #        for b,w in zip(self.biases, self.weights):
    #            a  = sigmoid(np.matmul(w,a) + b)
                a  = (self.activate)(np.matmul(w,a))
            return a
        else:
            lIt = 0
            rIt = 0
            weight = []
            bias = []
            for x,y in zip(self.sizes[1:], self.sizes[:-1]):
                rIt += x*y
                #print(lIt)
                #print(rIt)
                weight.append(a[lIt:rIt].reshape((x,y)))
                lIt = rIt
            for x in self.sizes[1:]:
                #print(lIt)
                #print(rIt)
                rIt += x
                bias.append(a[lIt:rIt].reshape((x,1)))
                lIt = rIt
            
            a = self.input
            for b,w in zip(bias, weight):
                a  = (self.activate)(np.matmul(w,a) + b)
            #print(weight)
            #print(bias)
            return a
                
        
    def SGD(self, train_imgs, train_labels, epochs, mini_batch_size, eta,
            evaluation_imgs = None, evaluation_labels = None, lmbda = 0.0,
            monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False,
            monitor_training_cost = False,
            monitor_training_accuracy = False):
   
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []  
        n_train = train_imgs.shape[1]
        if evaluation_imgs: n_eval = evaluation_imgs/784
        
        for i in range(epochs): 
            for j in range(0, int(n_train/mini_batch_size)):
                #taking transpose below in very much important
                X = train_imgs[:,j*mini_batch_size:j*mini_batch_size + mini_batch_size]
                y = train_labels.reshape((1,1)).repeat(mini_batch_size, axis = 0).transpose()
                self.update_mini_batch(X, y, mini_batch_size, eta, lmbda, train_labels.shape[0])
                               
            print("Epochs {0}".format(i))
            if monitor_training_cost:
                cost = self.total_cost(train_imgs, train_labels, lmbda)
                evaluation_cost.append(cost)
                print("Cost on training data: {1}".format(i, cost))
            if monitor_training_accuracy:
                accuracy = self.evaluate(train_imgs, train_labels, convert = True)
                evaluation_accuracy.append(accuracy)
#                print("Accuracy on training data: {0}/{1}".format(accuracy, 
#                      n_train))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_imgs, evaluation_labels, lmbda, convert = True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {1}".format(i, cost))
            if monitor_evaluation_accuracy:
                accuracy = self.evaluate(evaluation_imgs, evaluation_labels)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {0}/{1}".format(accuracy, 
                      n_eval)) 
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
    
    def update_mini_batch(self, X, y, mini_batch_size, eta, lmbda, n):
#        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
#        delta_nabla_b, delta_nabla_w = self.backprop(X, y, mini_batch_size)
        delta_nabla_w = self.backprop(X, y, mini_batch_size)
#        nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
#        self.biases = [b - (eta/mini_batch_size)*nb 
#                            for b,nb in zip(self.biases, nabla_b)]
#       L2 regulrization        
#        self.weights = [(1 - eta*(lmbda/n))*w - (eta/mini_batch_size)*nw 
#                            for w,nw in zip(self.weights, nabla_w)]
#       L1 regulrization
        self.weights = [(w - eta*(lmbda/n)*np.sign(w)) - (eta/mini_batch_size)*nw 
                            for w,nw in zip(self.weights, nabla_w)]
                            
    def evaluate(self, test_imgs, test_labels, convert = False):
        results = []
        for i in range(0, test_imgs.shape[1]):
            X = test_imgs[:, i]
            X = X.reshape(X.shape[0], 1)    #Very much important         
            results.append(self.feedforward(X))
        return results
        
    def backprop(self, X, y, mini_batch_size):
        activations = [X]
        z = []
        delta = [np.zeros((x, mini_batch_size)) 
                        for x in self.sizes[1:]]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for w in self.weights:
#        for b,w in zip(self.biases, self.weights):
#            z.append(np.matmul(w,activations[-1]) + b)
            z.append(np.matmul(w,activations[-1]))
            activations.append((self.activate)(z[-1]))
#        Quadratic cost function    
#        delta_nabla_b[-1] = (activations[-1] - y)*sigmoid_prime(z[-1])
            
        delta[-1] = (self.cost).delta(self.activate, z[-1], activations[-1], y)
        delta_nabla_w[-1] = np.dot(delta[-1], activations[-2].transpose())
                        
#        for l in range(2,self.num_layers):
#            delta[-l] = np.dot(self.weights[-l + 1].transpose(), 
#                                        delta[-l+1])*\
#                                        sigmoid_prime(z[-l])
#            delta_nabla_w[-l] = np.dot(delta[-l], 
#                                        activations[-l-1].transpose())
#        return (delta, delta_nabla_w)
        return delta_nabla_w
        
    def total_cost(self, data_imgs, data_labels, lmbda, convert = False):
        cost = 0.0
        for j in range(0, int(data_labels.shape[0])):
            X = data_imgs[j*784:j*784 + 784].reshape((1, 784)).transpose()
            if convert:
                y = data_labels[j:j + 1]
            else:
                y = data_labels[j:j + 1, :].transpose()
            a = self.feedforward(X)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a,y)/data_labels.shape[0]
        cost += 0.5*(lmbda/data_labels.shape[0])*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost     
        
    """
    Cuckoo Search Optimization
    """
    
    def objectiveFunction(self,x):
        #print(x)
        y_prime = self.feedforward(x);
        #print(y_prime)
        #print(self.output)
        return sum(abs(u-v) for u,v in zip(y_prime, self.output))/x.shape[0]

    def cso(self, n, x, y, function, lb, ub, dimension, iteration, pa=0.25,
                 nest=100):
        """
        :param n: number of agents
        :param function: test function
        :param lb: lower limits for plot axes
        :param ub: upper limits for plot axes
        :param dimension: space dimension
        :param iteration: number of iterations
        :param pa: probability of cuckoo's egg detection (default value is 0.25)
        :param nest: number of nests (default value is 100)
        """

        super(Network, self).__init__()

        self.__Nests = []
        
        self.input = x
        self.output = y

        beta = 3 / 2
        sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (
            gamma((1 + beta) / 2) * beta *
            2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.array([normalvariate(0, 1) for k in range(dimension)]) * sigma
        v = np.array([normalvariate(0, 1) for k in range(dimension)])
        step = u / abs(v) ** (1 / beta)

        self.__agents = np.random.uniform(lb, ub, (n, dimension))
        self.__nests = np.random.uniform(lb, ub, (nest, dimension))
        Pbest = self.__nests[np.array([function(x)
                                       for x in self.__nests]).argmin()]
        Gbest = Pbest
        self._points(self.__agents)

        for t in range(iteration):

            for i in self.__agents:
                val = randint(0, nest - 1)
                if function(i) < function(self.__nests[val]):
                    self.__nests[val] = i

            fnests = [(function(self.__nests[i]), i) for i in range(nest)]
            fnests.sort()
            fcuckoos = [(function(self.__agents[i]), i) for i in range(n)]
            fcuckoos.sort(reverse=True)

            nworst = nest // 2
            worst_nests = [fnests[-i - 1][1] for i in range(nworst)]

            for i in worst_nests:
                if random() < pa:
                    self.__nests[i] = np.random.uniform(lb, ub, (1, dimension))

            if nest > n:
                mworst = n
            else:
                mworst = nest

            for i in range(mworst):

                if fnests[i][0] < fcuckoos[i][0]:
                    self.__agents[fcuckoos[i][1]] = self.__nests[fnests[i][1]]

            self.__nests = np.clip(self.__nests, lb, ub)
            self.__Levyfly(step, Pbest, n, dimension)
            self.__agents = np.clip(self.__agents, lb, ub)
            self._points(self.__agents)
            self.__nest()

            Pbest = self.__nests[np.array([function(x)
                                        for x in self.__nests]).argmin()]

            if function(Pbest) < function(Gbest):
                Gbest = Pbest

        self._set_Gbest(Gbest)

    def __nest(self):
        self.__Nests.append([list(i) for i in self.__nests])

    def __Levyfly(self, step, Pbest, n, dimension):

        for i in range(n):
            stepsize = 0.2 * step * (self.__agents[i] - Pbest)
            self.__agents[i] += stepsize * np.array([normalvariate(0, 1)
                                                    for k in range(dimension)])

    def get_nests(self):
        """Return a history of cuckoos nests (return type: list)"""

        return self.__Nests
    
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


    """
    Data pre-processing
    """

def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

# define a function to convert a vector of time series into a 2D matrix
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix

state = {0: 'NSW', 1: 'QLD', 2: 'SA', 3: 'TAS', 4: 'VIC'}
year = {0: '2015', 1: '2016', 2: '2017'}

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
                dataset = pd.read_csv('./datasets/' + st + '/PRICE_AND_DEMAND_' + ye + '0' + str(mn) +'_' + st + '1.csv')
            else:
                dataset = pd.read_csv('./datasets/' + st + '/PRICE_AND_DEMAND_' + ye + str(mn) +'_' + st + '1.csv')
            df[st] = df[st].append(dataset.iloc[:,1:3])
    df[st] = df[st].set_index('SETTLEMENTDATE')
    

plt.figure(1)
plt.subplot(211)
plt.plot(df['NSW'].iloc[:,0].values)
plt.figure(2)
plt.subplot(211)
plt.plot(df['QLD'].iloc[:,0].values)
plt.figure(3)
plt.subplot(211)
plt.plot(df['SA'].iloc[:,0].values)
plt.figure(4)
plt.subplot(211)
plt.plot(df['TAS'].iloc[:,0].values)
plt.figure(5)
plt.subplot(211)
plt.plot(df['VIC'].iloc[:,0].values)

df['NSW'].iloc[:,0] = estimated_autocorrelation(df['NSW'].iloc[:,0].values)
df['QLD'].iloc[:,0] = estimated_autocorrelation(df['QLD'].iloc[:,0].values)
df['SA'].iloc[:,0] = estimated_autocorrelation(df['SA'].iloc[:,0].values)
df['TAS'].iloc[:,0] = estimated_autocorrelation(df['TAS'].iloc[:,0].values)
df['VIC'].iloc[:,0] = estimated_autocorrelation(df['VIC'].iloc[:,0].values)

plt.figure(1)
plt.subplot(212)
plt.plot(df['NSW'].iloc[:,0].values)
plt.figure(2)
plt.subplot(212)
plt.plot(df['QLD'].iloc[:,0].values)
plt.figure(3)
plt.subplot(212)
plt.plot(df['SA'].iloc[:,0].values)
plt.figure(4)
plt.subplot(212)
plt.plot(df['TAS'].iloc[:,0].values)
plt.figure(5)
plt.subplot(212)
plt.plot(df['VIC'].iloc[:,0].values)

plt.show()

# numpy array
list_hourly_load_NSW = df['NSW'].iloc[:,0].values
list_hourly_load_QLD = df['QLD'].iloc[:,0].values
list_hourly_load_SA = df['SA'].iloc[:,0].values
list_hourly_load_TAS = df['TAS'].iloc[:,0].values
list_hourly_load_VIC = df['VIC'].iloc[:,0].values

# the length of the sequnce for predicting the future value
sequence_length = 47

# convert the vector to a 2D matrix
matrix_load_NSW = convertSeriesToMatrix(list_hourly_load_NSW, sequence_length)
matrix_load_QLD = convertSeriesToMatrix(list_hourly_load_QLD, sequence_length)
matrix_load_SA = convertSeriesToMatrix(list_hourly_load_SA, sequence_length)
matrix_load_TAS = convertSeriesToMatrix(list_hourly_load_TAS, sequence_length)
matrix_load_VIC = convertSeriesToMatrix(list_hourly_load_VIC, sequence_length)

# shift all data by mean
matrix_load_NSW = np.array(matrix_load_NSW)
matrix_load_QLD = np.array(matrix_load_QLD)
matrix_load_SA = np.array(matrix_load_SA)
matrix_load_TAS = np.array(matrix_load_TAS)
matrix_load_VIC = np.array(matrix_load_VIC)

# shifted_value = matrix_load.mean()
# matrix_load -= shifted_value
print ("Data  shape: ", matrix_load_NSW.shape)

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
X_train_NSW = train_set_NSW[:, :-1]
X_train_QLD = train_set_QLD[:, :-1]
X_train_SA = train_set_SA[:, :-1]
X_train_TAS = train_set_TAS[:, :-1]
X_train_VIC = train_set_VIC[:, :-1]
# the last column is the true value to compute the mean-squared-error loss
y_train_NSW = train_set_NSW[:, -1] 
y_train_QLD = train_set_QLD[:, -1] 
y_train_SA = train_set_SA[:, -1] 
y_train_TAS = train_set_TAS[:, -1] 
y_train_VIC = train_set_VIC[:, -1] 
# the test set
X_test_NSW = matrix_load_NSW[train_row_NSW:, :-1]
y_test_NSW = matrix_load_NSW[train_row_NSW:, -1]

X_test_QLD = matrix_load_QLD[train_row_QLD:, :-1]
y_test_QLD = matrix_load_QLD[train_row_QLD:, -1]

X_test_SA = matrix_load_SA[train_row_SA:, :-1]
y_test_SA = matrix_load_SA[train_row_SA:, -1]

X_test_TAS = matrix_load_TAS[train_row_TAS:, :-1]
y_test_TAS = matrix_load_TAS[train_row_TAS:, -1]

X_test_VIC = matrix_load_VIC[train_row_VIC:, :-1]
y_test_VIC = matrix_load_VIC[train_row_VIC:, -1]

#eemd = EEMD()
#eIMFs_NSW = eemd(df['NSW'].iloc[:,0].values)
net = Network([46,10,1], Activation.tanh, backprop = False)
net.cso(100,X_train_NSW[0,:].reshape(46,1),y_train_NSW[0].reshape(1,1),net.objectiveFunction,-0.6,0.6,net.x.size ,500)