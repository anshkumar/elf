# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:16:25 2018

@author: vedanshu
"""

import numpy as np
#from PyEMD import EEMD
from math import gamma, pi, sin, sqrt
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
        a[a == 0] = 1e-10
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
    def __init__(self, sizes, activate, cost = CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        self.activate = activate
#        
#        if(backprop == False):
#            self.x = np.array([])
#            for w in self.weights:
#                self.x = np.insert(self.x, self.x.size, w.reshape(w.size))
#            for b in self.biases:
#                self.x = np.insert(self.x, self.x.size, b.reshape(b.size))
                
    def default_weight_initializer(self):
        self.biases = [np.random.randn(x,1) for x in self.sizes[1:]]
        self.weights = [np.random.randn(x,y)/np.sqrt(x) 
                        for x,y in zip(self.sizes[1:], self.sizes[:-1])]
    
    def large_weight_initializer(self):
        self.biases = [np.random.randn(x,1) for x in self.sizes[1:]]
        self.weights = [np.random.randn(x,y) 
                        for x,y in zip(self.sizes[1:], self.sizes[:-1])]
    
    def feedforward(self,a):
        for b,w in zip(self.biases, self.weights):
            a  = (self.activate)(np.matmul(w,a) + b)
        return a
                
    def set_weight_bias(self, a):
        lIt = 0
        rIt = 0
        self.weights = []
        self.biases = []
        for x,y in zip(self.sizes[1:], self.sizes[:-1]):
            rIt += x*y
            self.weights.append(a[lIt:rIt].reshape((x,y)))
            lIt = rIt
        for x in self.sizes[1:]:
            rIt += x
            self.biases.append(a[lIt:rIt].reshape((x,1)))
            lIt = rIt
           
    def SGD(self, train_x, train_y, epochs, mini_batch_size, eta,
            evaluation_x = None, evaluation_y = None, lmbda = 0.0,
            monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False,
            monitor_training_cost = False,
            monitor_training_accuracy = False):
   
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []  
        n_train = train_x.shape[1]
        if evaluation_x.size: n_eval = evaluation_x.shape[1]
#        print("biases ", [b.shape for b in self.biases])
        for i in range(epochs): 
            for j in range(0, int(n_train/mini_batch_size)):
                #taking transpose below in very much important
                X = train_x[:,j*mini_batch_size:(j+1)*mini_batch_size]
                y = train_y[j*mini_batch_size:(j+1)*mini_batch_size]
                self.update_mini_batch(X, y, mini_batch_size, eta, lmbda, train_y.shape[0])
                               
            if i % 100 == 0:
                print("Epochs {0}".format(i))
            if monitor_training_cost:
                cost = self.total_cost(train_x, train_y, lmbda)
                training_cost.append(cost)
                if i % 100 == 0:
                    print("Cost on training data: {1}".format(i, cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(train_x, train_y)
                training_accuracy.append(accuracy)
                if i % 100 == 0:
                    print("MAPE on training data: {0}".format(accuracy))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_x, evaluation_y, lmbda)
                evaluation_cost.append(cost)
                if i % 100 == 0:
                    print("Cost on evaluation data: {1}".format(i, cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_x, evaluation_y)
                evaluation_accuracy.append(accuracy)
                if i % 100 == 0:
                    print("MAPE on evaluation data: {0}".format(accuracy)) 
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
    
    def update_mini_batch(self, X, y, mini_batch_size, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
#        print("nabla_b", [b.shape for b in nabla_b])
        
        delta_nabla_b, delta_nabla_w = self.backprop(X, y, mini_batch_size)
#        delta_nabla_w = self.backprop(X, y, mini_batch_size)
        nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.biases = [b - (eta/mini_batch_size)*nb 
                            for b,nb in zip(self.biases, nabla_b)]


#       L2 regulrization        
#        self.weights = [(1 - eta*(lmbda/n))*w - (eta/mini_batch_size)*nw 
#                            for w,nw in zip(self.weights, nabla_w)]


#       L1 regulrization
        self.weights = [(w - eta*(lmbda/n)*np.sign(w)) - (eta/mini_batch_size)*nw 
                            for w,nw in zip(self.weights, nabla_w)]
                            
    def backprop(self, X, y, mini_batch_size):
        activations = [X]
        z = []
        delta = [np.zeros((x, mini_batch_size)) 
                        for x in self.sizes[1:]]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        
#        for w in self.weights:
        for b,w in zip(self.biases, self.weights):
            z.append(np.matmul(w,activations[-1]) + b)
#            z.append(np.matmul(w,activations[-1]))
            activations.append((self.activate)(z[-1]))

#        Quadratic cost function    
#        delta_nabla_b[-1] = (activations[-1] - y)*sigmoid_prime(z[-1])
            
        delta[-1] = (self.cost).delta(self.activate, z[-1], activations[-1], y)
        delta_nabla_w[-1] = np.dot(delta[-1], activations[-2].transpose())
                        
        for l in range(2,self.num_layers):
            delta[-l] = np.dot(self.weights[-l + 1].transpose(), 
                                        delta[-l+1])*\
                                        (self.activate)(z[-l])
            delta_nabla_w[-l] = np.dot(delta[-l], 
                                        activations[-l-1].transpose())
        
        delta_nabla_b = [b.sum(axis = 1).reshape((b.shape[0], 1)) 
                        for b in delta]
        return (delta_nabla_b, delta_nabla_w)
#        return delta_nabla_w
     
    def mean_absolute_percentage_error(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def accuracy(self, X, Y):
        results = np.zeros(X.shape[1])
        for i in range(0, X.shape[1]):
            x = X[:, i]
            x = x.reshape((x.shape[0], 1))    #Very much important           
            results[i] = self.feedforward(x).item(0)
        return self.mean_absolute_percentage_error(Y, results)
        
    def total_cost(self, X, Y, lmbda, convert = False):
        cost = 0.0
        for j in range(0, int(Y.shape[0])):
            x = X[:, j]
            y = Y[j]
            x = x.reshape((x.shape[0], 1))    #Very much important   
            a = self.feedforward(x)
            cost += self.cost.fn(a,y)/Y.shape[0]
        cost += 0.5*(lmbda/Y.shape[0])*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost     
        
    """
    Cuckoo Search Optimization
    """
    
#    def objectiveFunction(self,x):
#        self.set_weight_bias(x)
#        y_prime = self.feedforward(self.input)
#        return sum(abs(u-v) for u,v in zip(y_prime, self.output))/x.shape[0]

    def multiObjectiveFunction(self,x):
        self.set_weight_bias(x)
        y_prime = self.feedforward(self.input)
        ob1 = sum(abs(u-v) for u,v in zip(y_prime, self.output))/x.shape[0]
        ob2 = sqrt(sum((u-v)**2 for u,v in zip(y_prime, self.output))/x.shape[0])
        ob3 = sum(abs((u-v)/v) for u,v in zip(y_prime, self.output))/x.shape[0]
        ob4 = sqrt(sum((abs((u-v)/v) - ob3)**2 for u,v in zip(y_prime, self.output))/x.shape[0])
        return min([ob1,ob2,ob3,ob4])
    
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


