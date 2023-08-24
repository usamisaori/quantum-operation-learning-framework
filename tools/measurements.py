#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[1]:


from functools import reduce

Dag = lambda matrix: matrix.conj().T
Kron = lambda *matrices: reduce(np.kron, matrices)


# In[5]:


def getMeasurements(n):
    psi_0 = np.array([1.0, 0.0])
    psi_1 = np.array([0.0, 1.0])
    I = np.eye(2)

    M_0 = psi_0.reshape([2, 1]) @ psi_0.reshape([1, 2]).conj()
    M_1 = psi_1.reshape([2, 1]) @ psi_1.reshape([1, 2]).conj()
    
    M = [M_0, M_1]
    
    measurements = []
    for i in range(2 ** n):
        binnum = bin(i)[2:].rjust(n, '0')

        temp = []
        indices = map(lambda x: int(x), list(binnum))
        for index in indices:
            temp.append(M[index])
        
        measurements.append( Kron(*temp) )
    
    return measurements


# In[6]:


def measure(M, state):
    return Dag(state) @ Dag(M) @ M @ state


# In[ ]:




