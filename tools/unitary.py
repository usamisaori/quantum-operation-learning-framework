#!/usr/bin/env python
# coding: utf-8

# In[1]:


import qiskit
import numpy as np


# In[2]:


def generateCombs(n, combs, comb):
    """
        n = 3 => 000 ... 333
    """
    if len(comb) == n:
        combs.append(comb)
        
        return

    for i in range(4):
        generateCombs(n, combs, comb + str(i))


# In[3]:


combs = dict()

for i in range(2, 5):
    combs_ = []
    generateCombs(i, combs_, '')
    combs[i] = combs_


# In[5]:


from collections import defaultdict

from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.extensions import HamiltonianGate
from qiskit.extensions import UnitaryGate

matrixes = [
    Pauli(label='I').to_matrix(),
    Pauli(label='X').to_matrix(),
    Pauli(label='Y').to_matrix(),
    Pauli(label='Z').to_matrix(),
]

Hermitians = defaultdict(list)


# In[6]:


def buildU(n, p, time=1.0):
    H = np.zeros((2 ** n, 2 ** n)).astype('complex64')
    
    # global Hermitians
    if n not in Hermitians:
        # global combs
        for comb in combs[n]:
            temp = np.kron(matrixes[int(comb[0])], matrixes[int(comb[1])])
            
            for i in range(2, n):
                temp = np.kron(temp, matrixes[int(comb[i])])
                
            Hermitians[n].append(temp)
            
    for para, hermitian in zip(p, Hermitians[n]):
        H += para * hermitian
    
    # time: simulation time
    # return type => DensityMatrix (qiskit)
    U = HamiltonianGate(H, time)
    
    return UnitaryGate(U.to_matrix(), label='U')


# In[ ]:




