# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 15:16:25 2018

@author: vedanshu
"""

import numpy as np
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
            return (a-y)*Activation.tanh_prime(z)
        elif(activation == Activation.relu):
            return (a-y)*(z > 0)
        
class Accuracy(object):
    @staticmethod
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true - y_pred))) 
 
    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

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
    def tanh_prime(z):
        return 1 - (Activation.tanh(z))**2
        
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
        self.dim = sum(x*(y+1) for x,y in zip(self.sizes[1:], self.sizes[:-1]))
               
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
            monitor_training_accuracy = False,
            output2D = False):
   
        evaluation_cost, eval_mape, eval_rmse, eval_mae = [], [] , [] , []
        training_cost, training_mape, training_rmse, training_mae = [], [] , [] , []
        n_train = train_x.shape[1]
        for i in range(epochs): 
            for j in range(0, int(n_train/mini_batch_size)):
                #taking transpose below in very much important
                X = train_x[:,j*mini_batch_size:(j+1)*mini_batch_size]
                if output2D:
                    y = train_y[:, j*mini_batch_size:(j+1)*mini_batch_size]
                else:
                    y = train_y[j*mini_batch_size:(j+1)*mini_batch_size]
                self.update_mini_batch(X, y, mini_batch_size, eta, lmbda, train_x.shape[1])
                               
            if i % 100 == 0:
                print("Epochs {0}".format(i))
            if monitor_training_cost:
                cost = self.total_cost(train_x, train_y, lmbda, output2D)
                training_cost.append(cost)
                if i % 100 == 0:
                    print("Cost on training data: {1}".format(i, cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(train_x, train_y, output2D, Accuracy.mape)
                training_mape.append(accuracy)
                if i % 100 == 0:
                    print("MAPE on training data: {0}".format(accuracy))
                    
                accuracy = self.accuracy(train_x, train_y, output2D, Accuracy.rmse)
                training_rmse.append(accuracy)
                if i % 100 == 0:
                    print("RMSE on training data: {0}".format(accuracy))
                    
                accuracy = self.accuracy(train_x, train_y, output2D, Accuracy.mae)
                training_mae.append(accuracy)
                if i % 100 == 0:
                    print("MAE on training data: {0}".format(accuracy))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_x, evaluation_y, lmbda, output2D)
                evaluation_cost.append(cost)
                if i % 100 == 0:
                    print("Cost on evaluation data: {1}".format(i, cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_x, evaluation_y, output2D, Accuracy.mape)
                eval_mape.append(accuracy)
                if i % 100 == 0:
                    print("MAPE on evaluation data: {0}".format(accuracy)) 
                    
                accuracy = self.accuracy(evaluation_x, evaluation_y, output2D, Accuracy.rmse)
                eval_rmse.append(accuracy)
                if i % 100 == 0:
                    print("RMSE on evaluation data: {0}".format(accuracy)) 
                    
                accuracy = self.accuracy(evaluation_x, evaluation_y, output2D, Accuracy.mae)
                eval_mae.append(accuracy)
                if i % 100 == 0:
                    print("MAE on evaluation data: {0}".format(accuracy)) 
        return evaluation_cost, eval_mape, eval_rmse, eval_mae, training_cost, training_mape, training_rmse, training_mae
    
    def update_mini_batch(self, X, y, mini_batch_size, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        delta_nabla_b, delta_nabla_w = self.backprop(X, y, mini_batch_size)
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
        
        for b,w in zip(self.biases, self.weights):
            z.append(np.matmul(w,activations[-1]) + b)
            activations.append((self.activate)(z[-1]))
           
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
        
    def accuracy(self, X, Y, output2D = False, AccFunc = Accuracy.mape):
        if not output2D:
            results = np.zeros(X.shape[1])
            for i in range(0, X.shape[1]):
                x = X[:, i]
                x = x.reshape((x.shape[0], 1))    #Very much important           
                results[i] = self.feedforward(x).item(0)
            return AccFunc(Y, results)
        else:
            results = []
            for i in range(0, X.shape[1]):
                x = X[:, i]
                x = x.reshape((x.shape[0], 1))    #Very much important           
                results.append(self.feedforward(x))
            return AccFunc(Y, np.hstack(results))
        
    def total_cost(self, X, Y, lmbda, output2D = False):
        cost = 0.0
        for j in range(0, int(X.shape[1])):
            x = X[:, j]
            if output2D:
                y = Y[:,j]
            else:
                y = Y[j]
            x = x.reshape((x.shape[0], 1))    #Very much important   
            a = self.feedforward(x)
            cost += self.cost.fn(a,y)/Y.shape[0]
        cost += 0.5*(lmbda/Y.shape[0])*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost     
        
    """
    Cuckoo Search Optimization
    """

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
    
