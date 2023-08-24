#!/usr/bin/env python
# coding: utf-8

# In[1]:


import qiskit
from qiskit import QuantumCircuit


# In[5]:


def getVQCCircuit(circuit, n, p, pos=None, *, vqc_layers_num = 3):
    if pos == None:
        pos = list(range(n))
    
    index = 0
    for layer in range(vqc_layers_num ): # vqc_layers_num = 3
        for i in range(n):
            circuit.u2(p[index], p[index + 1], pos[i])
            index += 2

        if layer + 1 == vqc_layers_num: # vqc_layers_num = 3
            break

        for i in range(n):
            if i + 1 != n:
                circuit.cx(pos[i], pos[i + 1])
            elif n > 2:
                circuit.cx(pos[i], pos[0])
    
    return circuit


# In[ ]:




