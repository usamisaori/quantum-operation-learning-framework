#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from quantum import *
from measurements import *


# In[5]:


def calculateLipschitz(targetCircuit, afterCircuit, measurements):
    lip = 0.0
    
    t = getDensityMatrix(targetCircuit)
    a = getDensityMatrix(afterCircuit)

    for measurement in measurements:
        # M => M @ E_after @ E_target
        measurement = measurement @ a @ t
        
        D = Dag(measurement) @ measurement
        D = D @ D
        lip += abs(np.trace(D))
        
    return lip


# In[5]:


def calculateLipschitz2(targetCircuit, afterCircuit, measurements):
    lip = 0.0
    
    t = getDensityMatrix(targetCircuit)
    a = getDensityMatrix(afterCircuit)

    for measurement in measurements:
        # M => M @ E_after @ E_target
        measurement = measurement @ a @ t
        
        D = Dag(measurement) @ measurement
        D = D @ D
        lip += np.trace(D)
        
    return lip


# In[4]:


from scipy.linalg import sqrtm

def calculateLipschitz3(targetCircuit, afterCircuit, measurements):
    lip = 0.0
    
    t = getDensityMatrix(targetCircuit)
    a = getDensityMatrix(afterCircuit)

    for measurement in measurements:
        # M => M @ E_after @ E_target
        measurement = measurement @ a @ t
        
        D = Dag(measurement) @ measurement
        sD = sqrtm(Dag(D) @ D)
        D = sD @ sD
        lip += np.trace(D)
        
    return lip


# In[2]:


from scipy.linalg import sqrtm

def trace_distance(rou, sigma):
    A = rou - sigma
    A_ = sqrtm( np.dot( A.conj().T, A ) )

    return 0.5 * np.linalg.norm( np.trace(A_) )


# In[3]:


def L1_distance(P, Q):
    res = 0
    for (p, q) in zip(P, Q):
        res += np.abs(p - q)
    
    return res


# In[6]:


def calculateExactLipschitz(inputCircuits, noiseInputCircuits, targetCircuit, afterCircuit, measurements):
    res = 0.0
    t = getDensityMatrix(targetCircuit)
    p = getDensityMatrix(afterCircuit)
    
    for (circuit, noiseCircuit) in zip(inputCircuits, noiseInputCircuits):
        rou = getDensityMatrix(circuit)
        sigma = getDensityMatrix(noiseCircuit)
        
        d = trace_distance(rou, sigma)
        
        rou_ = Dag(p) @ Dag(t) @ rou @ t @ p
        sigma_ = Dag(p) @ Dag(t) @ sigma @ t @ p
        
        P, Q = [], []
        
        for measurement in measurements:
            m = measurement
            probability_p = np.abs(np.trace(Dag(m) @ rou_ @ m))
            probability_q = np.abs(np.trace(Dag(m) @ sigma_ @ m))
            
            P.append(probability_p)
            Q.append(probability_q)
            
        D = L1_distance(P, Q)
        
        K = D / d
        if K > res:
            res = K
    
    return res


# In[4]:


def calculateExactLipschitz2(inputCircuits, noiseInputCircuits, targetCircuit, afterCircuit, measurements):
    res = 0.0
    t = getDensityMatrix(targetCircuit)
    p = getDensityMatrix(afterCircuit)
    
    for i in range(len(noiseInputCircuits)):
        for j in range(len(noiseInputCircuits)):
            if i >= j: 
                continue
            rou = getDensityMatrix(noiseInputCircuits[i])
            sigma = getDensityMatrix(noiseInputCircuits[j])
            d = trace_distance(rou, sigma)
        
            rou_ = Dag(p) @ Dag(t) @ rou @ t @ p
            sigma_ = Dag(p) @ Dag(t) @ sigma @ t @ p

            P, Q = [], []

            for measurement in measurements:
                m = measurement
                probability_p = np.abs(np.trace(Dag(m) @ rou_ @ m))
                probability_q = np.abs(np.trace(Dag(m) @ sigma_ @ m))

                P.append(probability_p)
                Q.append(probability_q)

            D = L1_distance(P, Q)

            K = D / d
            if K > res:
                res = K
    return res


# In[ ]:




