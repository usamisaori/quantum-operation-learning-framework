#!/usr/bin/env python
# coding: utf-8

# In[1]:


import qiskit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.utils import insert_noise
from qiskit.providers.aer.noise import pauli_error, depolarizing_error


# In[2]:


def createNoiseModel(p, errorType):
    # QuantumError objects
    if errorType == 'bit_flip' or errorType == 'b':
        error = pauli_error([('X', p), ('I', 1 - p)])
    elif errorType == 'phase_flip' or errorType == 'p':
        error = pauli_error([('Z', p), ('I', 1 - p)])
    elif errorType == 'depolarizing' or errorType == 'd':
        error = depolarizing_error(p, num_qubits=1)
        
    ## two-qubits quantumError objects 
    if errorType == 'depolarizing' or errorType == 'd':
        error_2qubits = depolarizing_error(p, num_qubits=2)
    else:
        error_2qubits = error.tensor(error)
        
    # Add errors to noise model
    noise_model = NoiseModel()
    
    noise_model.add_all_qubit_quantum_error(error, ['x', 'y', 'z', 'h', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2qubits, ['cx', 'cz', 'swap'])
    
    return noise_model


# In[3]:


def getNoiseCircuit(circuit, p, errorType):
    noise_model = createNoiseModel( p, errorType)
    noise_circuit = insert_noise(circuit, noise_model)
    
    return noise_circuit


# In[1]:


def getNoiseCircuits(inputCircuits, p, errorType):
    noiseInputCircuits = []
    
    for circuit in inputCircuits:
        noiseInputCircuits.append(getNoiseCircuit(circuit, p, errorType))
    
    return noiseInputCircuits


# In[ ]:




