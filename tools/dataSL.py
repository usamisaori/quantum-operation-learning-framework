#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# ## Save datas

# In[1]:


def save(data, datatype, name, strategy, times):
    # eg. params/losses_sdc/sqt/grover/..._unitary/vqc/..._3
    
    if datatype == 'params':
        saveParams(data, name, strategy, times)
    elif datatype == 'losses':
        saveLosses(data, name, strategy, times)


# In[2]:


def saveParams(params, name, strategy, times):
    path = f"./data/{name}/params/{name}_{strategy}_params_{times}"
    path += '.pkl'
    
    with open(path, 'wb') as file:
        pickle.dump(params, file)
        
def saveLosses(losses, name, strategy, times):
    path = f"./data/{name}/losses/{name}_{strategy}_losses_{times}"
    path += '.pkl'
    
    with open(path, 'wb') as file:
        pickle.dump(losses, file)


# ## Load datas

# In[3]:


def load(datatype, name, strategy, times):
    # eg. params/losses_sdc/sqt/grover/..._unitary/vqc/..._3(_F/D/)
    
    if datatype == 'params':
        return loadParams(name, strategy, times)
    elif datatype == 'losses':
        return loadLosses(name, strategy, times)


# In[4]:


def loadParams(name, strategy, times):
    path = f"./data/{name}/params/{name}_{strategy}_params_{times}"
    path += '.pkl'
    
    with open(path, 'rb') as file:
        params = pickle.load(file)
    
    return params

def loadLosses(name, strategy, times):
    path = f"./data/{name}/losses/{name}_{strategy}_losses_{times}"
    path += '.pkl'
    
    with open(path, 'rb') as file:
        losses = pickle.load(file)
    
    return losses


# ## Plot data

# In[1]:


import numpy as np


# In[5]:


import matplotlib.pyplot as plt 

def plot(data, color='skyblue'):
    y = data
    x = range(1, len(y) + 1)
        
    plt.plot(x, y, color=color, linewidth=2.0, linestyle='--')


# In[1]:


def plotLosses(losses, color="red", fill_color="#CCCCFF", title=None, fontsize=20):
    losses_mean = np.mean(losses, axis=0)
    losses_std = np.std(losses, axis=0)
    
    x = range(1, len(losses_mean) + 1)
    plt.grid()
    plt.plot(x, losses_mean, color=color, linewidth=2.0, linestyle='--', label="Mean value")
    plt.fill_between(x, losses_mean + losses_std, losses_mean - losses_std, 
                     color=fill_color, label="Standard deviation")
    plt.legend(["Mean value", "Standard deviation"], loc="upper right")
    
    if title != None:
        plt.title(title, fontsize=fontsize)


# In[ ]:




