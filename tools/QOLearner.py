#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from qiskit import QuantumCircuit
from measurements import *
from quantum import *
from unitary import *
from lipschitz import *


# In[5]:


class QOLearnerBase:
    """
    circuits: [inputCircuit, *, outputCircuit]
    n: dimension about target operator, pos: eg. [0, 1, 2], len(pos) should equal to n
    expected: single num => target , list [...] => probability distribution
    """
    def __init__(self, circuits, n, pos, expected, *, vqc_layers_num = 3, lipschitz_regularization=False):
        self.circuits = circuits
        if not isinstance(self.circuits[0], list):
            self.circuits[0] = [self.circuits[0]]
            
        self.n = n
        self.initMeasurements()
        self.pos = pos
        self.expected = self.initExpected(expected)
        
        self.params = []
        self.losses = []
        
        self.vqc_layers_num = vqc_layers_num 
        self.lipschitz_regularization = lipschitz_regularization
        
    def initMeasurements(self):
        n = len(self.circuits[0][0].qubits)
        
        self.measurements = getMeasurements(n)
        
        dictionary = dict()
        for i in range(2 ** n):
            binnum = bin(i)[2:].rjust(n, '0')
            dictionary[binnum] = self.measurements[i]
        
        self.dictionary = dictionary
    
    def initExpected(self, expected):
        total = 0
        for value in expected.values():
            total += value

        return_expected = dict()
        for k in self.dictionary.keys():
            if k not in expected:
                return_expected[k] = 0.0
            else:
                return_expected[k] = expected[k] / total

        return return_expected
    
    ###############################################
    
    def cost(self, circuit):
        state = getStatevector(circuit)
        loss = 0.0
        
        for k in self.expected.keys():
            expected_p = self.expected[k]
            p = measure(self.dictionary[k], state)
            loss += (expected_p - p) ** 2
        
        return loss / len(self.measurements)
        
    def L(self, p):
        loss = 0.0
        
        for circuit in self.circuits[0]:
            # input circuit
            circuit = circuit.copy()
            targetCircuit = QuantumCircuit(len(circuit.qubits), len(circuit.qubits))
            
            # parametered part => unitary gate / VQC
            if self.strategy == 'unitary':
                U = buildU(self.n, p)
                targetCircuit.append(U, self.pos)
            else:
                index = 0
                for layer in range(self.vqc_layers_num):
                    for i in range(self.n):
                        targetCircuit.u2(p[index], p[index + 1], self.pos[i])
                        index += 2

                    if layer + 1 == self.vqc_layers_num:
                        break

                    for i in range(self.n):
                        if i + 1 != self.n:
                            targetCircuit.cx(self.pos[i], self.pos[i + 1])
                        elif self.n > 2:
                            targetCircuit.cx(self.pos[i], self.pos[0])

            # remained circuit
            circuit = circuit.compose(targetCircuit).compose(self.circuits[1])
            
            loss += self.cost(circuit)
            if self.lipschitz_regularization:
                loss += calculateLipschitz(targetCircuit, self.circuits[1], self.measurements) * self.regularization_lambda
    
        return loss
        
    def gradientCalculator(self, p, epsilon):
        derivates = np.zeros(len(p))
        
        l_p = self.L(p)
        for i, param in enumerate(p):
            p[i] += epsilon
            l_pe = self.L(p)
            p[i] -= epsilon

            derivates[i] = (l_pe - l_p)

        return derivates / epsilon, l_p
    
    def fit(self, *, epoch=200, epsilon=0.01, stepsize=0.01, strategy='unitary', early_stopping=False, regularization_lambda=1.0):
        self.params = []
        self.losses = []
        self.regularization_lambda = regularization_lambda
        
        # strategy: unitary => buildU / VQC
        self.strategy = strategy
        
        # 1. initialize the parameters with uniformly random nunbers
        size = 4 ** self.n if strategy == 'unitary' else self.vqc_layers_num * self.n * 2
        p = np.random.uniform(-1, 1, size)

        # Adam parameters:
        momentum, s = 0, 0
        ## Stepsize
        alpha = stepsize
        belta1 = 0.9; belta2 = 0.999
        e = 1e-8

        t = 0 # epoch counter
        
        lastloss = 0
        while t < epoch:
            # repeat
            t += 1
            grad, loss = self.gradientCalculator(p, epsilon)
            self.losses.append(loss)
            self.params.append(p)

            if early_stopping and abs(loss - lastloss) < 1e-5:
                break
            lastloss = loss

            # Adam Optimizer
            momentum = belta1 * momentum + (1 - belta1) * grad
            s = belta2 * s + (1 - belta2) * (grad ** 2)
            m_ = momentum / (1 - belta1 ** t)
            s_ = s / (1 - belta2 ** t)

            # update parameters
            p = p - alpha * (m_ / ( s_ ** 0.5 + e ))

            # log
            print('epoch: [{}/{}] - loss: {} end.'.format(t, epoch, loss))


# In[ ]:




