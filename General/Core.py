# Core.py

# Import Modules
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
import seaborn as sns
import sklearn 
import sklearn.metrics as skm
import spacy
from spacy.symbols import ORTH
import os, time, copy
import re, html, json, pickle
import cv2, skimage, skimage.io, skimage.transform
import itertools, collections
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import GPUtil, psutil
from tqdm import tqdm_notebook
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

# This file Core.py contains a collection of simple core functions and classes 
# used in other parts of the library. They are "core" in the sense that they 
# depend only upon the imports from standard libraries given above, but NOT 
# any other files in my library. 

# OUTLINE 
# 1. Variable Type Conversion 
# 2. Regular Utilities
# 3. Torch Utilities
# 4. Data Splitting
# 5. Combining Models and Predictions


# SECTION 1 - VARIABLE TYPE CONVERSION
        
def TEN(x,GPU=True):  
    
    """ This function takes an input x which can be either a numpy array,
    a list of numbers, or a numeric type such as int, float, or np.int32. 
    The input is converted to a torch FloatTensor or LongTensor depending
    on its form. If GPU==True Tensor is placed on GPU, otherwised on CPU.  
    
    NOTE: Numeric types are limited to float, int, np.float32, np.float64, 
          np.int32, np.int64. Corresponding restrictions apply for dtypes 
          of numpy arrays and entries of lists.        
    """
    
    if isinstance(x,list): 
        x = np.array(x)
   
    if isinstance(x,np.ndarray) and x.dtype in [np.float32,np.float64]:
        x = torch.FloatTensor(x)        
    elif isinstance(x,np.ndarray) and x.dtype in [np.int32,np.int64]:
        x = torch.LongTensor(x)
    elif isinstance(x,(float,np.float32,np.float64)):
        x = torch.tensor(x,dtype=torch.float32)
    elif isinstance(x,(int,np.int32,np.int64)):
        x = torch.tensor(x,dtype=torch.int64)    
        
    if GPU==True: x = x.cuda()
    return x
    
def ARR(x):        
    "Converts input x of type torch Tensor (on cpu or gpu) to a numpy array (on cpu)."  
    if x.is_cuda == False: return x.numpy()
    elif x.is_cuda == True: return x.cpu().numpy()
    
def LIST(x,N,Tuple=True,Array=True):
    "Convert input x into a length-N list in a 'natural' way." 
    if isinstance(x,list) and len(x) == N: return x
    elif Tuple and isinstance(x,tuple) and len(x) == N: return list(x)
    elif Array and isinstance(x,np.ndarray) and len(x) == N: return list(x)
    else: return [x]*N 

    
# SECTION 2 - REGULAR UTILITIES     
       
def list_del(L,idxs):
    "Delete items from list L at indices specified in the list idxs."
    L_split = []
    idxs = list(set(idxs))
    idxs.sort()
    idxs = [-1] + idxs + [len(L)]
    for n in range(len(idxs)-1): 
        L_split.append(L[idxs[n]+1:idxs[n+1]])
    return sum(L_split,[])
    
def list_mult(L,c):
    """Multiply each element of a list L by a constant c.
       If input L is a single number instead of list, function just multiplies number by c."""
    if type(L) == list: return [x*c for x in L]
    else: return L*c
    
def outer_mult(A,B):
    """Multiply each element of B by A and return output as a numpy array with length = len(B).
       B is a 1d numpy array, A can be a number (float, int, etc...) or 1d numpy array."""
    return np.array([A*B[i] for i in range(len(B))])

def linear_space(A,B,N):
    """Returns a numpy array of N linearly spaced points from A to B. 
       A and B may both be numbers, or both be numpy arrays of the same length L.
       In latter case, the returned points are linearly spaced vectors from A to B in R^L. """ 
    if isinstance(A,(float,int)): return np.linspace(A,B,N)
    else: return np.array([np.linspace(A[i],B[i],N) for i in range(len(A))]).transpose()  
    
def joint_sort(lists,reverse=False):    
    """ Input: lists = [L1,...,Ln], where the Li's are all lists of same length.  
        Output: Returns [LL1,...,LLn] where LL1 is the sorted version of L1 and, for i>1, 
                LLi is a version Li with elements re-ordered according to same permuation
                used to transform L1 into LL1. """    
    
    L0 = lists[0]
    perm = sorted(range(len(L0)), key=lambda k: L0[k], reverse=reverse)
    sorted_lists = []
    for L in lists: 
        sorted_lists.append( [L[x] for x in perm] )
    return sorted_lists

def correct_foldername(folder_name):
    "Adds a '/' if necessary to end of foldername (because it is easy to forget this)."
    if folder_name[-1] != '/': 
        folder_name = folder_name + '/'
    return folder_name
    

# SECTION 3 - TORCH UTILITIES 
bn_types = (nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)
linconv_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)

def to_cuda(x):
    "Transfer a Tensor or list of Tensors x to the gpu." 
    if isinstance(x,list): return [to_cuda(x[i]) for i in range(len(x))]
    elif x.is_cuda == False: return x.cuda()
    else: return x

def trainable_params(m): 
    "Return list of trainable params in an nn.Module or nn.ModuleList <m>."
    return list(filter(lambda p: p.requires_grad, m.parameters()))

def num_children(m):
    "Get number of children modules in an nn.Module or nn.ModuleList <m>."
    return len(list(m.children()))

def flatten_module(m):
    """Flattens out an nn.Module or nn.ModuleList <m> into a list of its 
    'base components', each of which has no children."""
    return sum(map(flatten_module,m.children()),[]) if num_children(m) > 0 else [m]

def initialize_module(m,init_func,bn_init=False):  
    
    """Applies given init_func to an nn.Module <m>. More precisely:
       * init_func is applied to weights in linear and conv layers of module.
       * biases in linear and conv layers are all set to 0.
       * If bn_init == True, batch norm layers initialized to have weight 1 and bias 0.
         Otherwise, batchnorm layers are left with pytorch defaults.
       NOTE: Typical init_func's: nn.init.kaiming_normal_ or nn.init.kaiming_uniform_.
             The init_func must be an in place operation."""
    
    for l in m.modules(): 
        if isinstance(l, linconv_types):
            init_func(l.weight)
            nn.init.constant_(l.bias,0)
        elif isinstance(l, bn_types) and bn_init == True:
            nn.init.constant_(l.weight,1)
            nn.init.constant_(l.bias,0)

def initialize_modules(L,init_func,bn_init=False):
    """Applies function <initialize_module> to each module m in the list L."""
    for m in L: initialize_module(m,init_func,bn_init)
            
def separate_bn_layers(layer_groups):
    
    """Separates each layer group in a list <layer_groups> into its batchnorm and non-batch norm
    layers, and returns list of split layer groups of form [G1_1,...,G1_N,G2_1,...,G2_N].
    Here G1_i consists of all non-batchnorm layers in layer group i (put into an nn.ModuleList). 
    and  G2_i consists of all batchnorm layers in layer group i (put into an nn.ModuleList). """
   
    reg_groups, bn_groups = [],[]
    for G in layer_groups:
        G_flat = flatten_module(G)
        G1,G2 = [],[]
        for layer in G_flat: 
            if isinstance(layer,bn_types): G2.append(layer)
            else: G1.append(layer)
        reg_groups.append(nn.ModuleList(G1))        
        bn_groups.append(nn.ModuleList(G2))        
    return reg_groups + bn_groups

def make_model_basic(model):
    
    """Takes a model given as a standard nn.Module, and converts it to a 
    model of type used by Learner and Optimizer classes by treating 
    whole model as a single layer group. """
    
    model.layer_groups = [model]
    model.param_groups = separate_bn_layers(model.layer_groups)
    return model

class SaveFeatures():
    "SaveFeatures Class, directly from fastai notebook, for storing layer outputs easily."
    features=None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def close(self): self.hook.remove()


# SECTION 4 - DATA SPLITTING

def SplitTrainVal(datapoints,val_idxs=None,val_frac=0.2):
    
    """ Function to split a collection of datapoints into 2 sets: one for training
    and one for validation. The datapoints to use for validation are chosen randomly
    unless val_idxs is specified.
        
    Arguments:
    datapoints: A collection of datapoints. Can be a list or pd.DataFrame.
                In case of list, each element of list is a datapoint.
                In case of pd.DataFrame, each row of DataFrame is a datapoint.
    val_idxs: If val_idxs != None, then val_idxs should be a list of indices between 0 and len(datapoints)-1,
              which are the indices to use for validation. In this case, val_frac is irrelevant. 
    val_frac: Assuming val_idxs == None, val_frac is the fraction of datapoints to use for validation.
              Example: If val_frac == 0.2, then 20% of datapoints used for validation.
    
    Ouput: 
    Returns 2 objects: train_datapoints, val_datapoints (same format as input datapoints).  
    """
    
    N = len(datapoints)
    if val_idxs is None:
        val_idxs = list( np.random.choice(np.arange(N),int(N*val_frac),replace = False) )
    train_idxs = list(set(np.arange(N)) - set(val_idxs))                       
    
    if type(datapoints) == pd.DataFrame:
        return datapoints.iloc[train_idxs].copy(), datapoints.iloc[val_idxs].copy()
    elif type(datapoints) == list:
        return [datapoints[i] for i in train_idxs], [datapoints[i] for i in val_idxs]


# SECTION 5 - COMBINING MODELS AND PREDICTIONS   

def combine_models(model_list,weights=None):
    
    """Function to combine pytorch models by averaging the values of each parameter.
       Also, averages the values of any running buffers, e.g. for batch norm layers.
       (NOTE: This is useful for SWA averaging, as implemented by Learner class.) 
    
    Arguments:
    model_list: A list of pytorch models to combine (each of class nn.Module, or subclass).
                All models must have exactly same architecture.
    weights: weights to use in averaging paramaters, should sum to 1.
             If weights == None, then equal weight used for all models. 
    """

    n = len(model_list)
    if weights is None: weights = [1/n]*n
    m_avg = copy.deepcopy(model_list[0])
    dict_avg = m_avg.state_dict()
    dicts = [m.state_dict() for m in model_list]    
    
    for name in dict_avg:
        dict_avg[name] = sum(weights[i]*dicts[i][name] for i in range(n))
    
    m_avg.load_state_dict(dict_avg)
    return m_avg
    
def combine_preds(preds,target_type,weights=None):
    
    """Combines multiple sets of predictions for a given dataset by averaging (possibly weighted).
    
    Arguments:
    target_type: Type of predictions for dataset, must be either 'cont','cat','single_label', or 'multi_label'.
    preds: A list of sets of predictions for the dataset. Let N be number of datapoints in dataset. 
           If target_type == 'cont', each preds[i] is a 1d np.array of length N.
           If target_type == 'cat','single_label', or 'multi-lablel', each preds[i] is an N by C np.array, 
           where C = number of categories. 
    weights: A list of same length as preds to use for avering predictions, 
             values in weights should sum to 1. If weights == None, then all sets
             of predictions in preds are weighted equally.
             
    Output: 
    * If target_type == 'cont': 
         returns <combined_preds>, same form as each preds[i].
    * If target_type == 'cat', 'single_label', or 'multi_lablel': 
         returns <combined_preds>, <pred_labels> where:
         (1) <combined_preds> is same form as each preds[i]
         (2) For 'cat' or 'single_label':
             <pred_labels> is a length N numpy integer array giving predicted categories of each input.              
             For 'multi_label': 
             <pred_labels> is an N by C numpy 0-1 array where preds_labels[i,j] = 1 
             if and only if there is greater than 1/2 chance category j present in input i. 
    """
    
    n = len(preds)
    if weights is None: weights = [1/n]*n
    combined_preds = sum(weights[i]*preds[i] for i in range(n))
    if target_type == 'cont': return combined_preds
    elif target_type in ['cat','single_label']: return combined_preds, combined_preds.argmax(axis=1)
    elif target_type == 'multi_label': return combined_preds, combined_preds.round().astype(int)        

