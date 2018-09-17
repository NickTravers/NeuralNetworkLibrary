# CollaborativeFiltering.py

# Problem Description: 
# There are 2 categorical input variables (usually of high cardinality) which are 
# referred to generically as "user" and "item". There is also one continuous output
# variable referred to as "rating". We are given N datapoints of the form (user,item,rating)
# as training data, and we then wish to predict the rating values for other (user,item) pairs 
# we have not seen in training data. 

# Example: Netflix
# Users are people, items are movies, each person may give a rating between 0 and 5
# to a movie after watching it. (A large prize was awarded for best reccomendation system). 

# NOTE: Following the same format as in StructuredData.py we think of the inputs (user,item) 
# as "x values" and the output (rating) as the "y value".  This is used in our variable 
# naming schemes below. 

# Import Modules
from General import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# List of Functions and Classes

# 1. class CollabFilterDataset
# 2. class CollabFilterDataObj
# 3. class CollabFilterNet 

class CollabFilterDataset(Dataset):
    """
    A class for a dataset of collaborative filtering data (can be either train, val, or test). 
    Inherits from class torch.utils.data.Dataset (imported as Dataset). 
    
    Arguments for initialization:
    df: * If for train or validation data, df is a pandas Dataframe containing a
         users column, items column, and ratings column. 
        * If for test data, df is a pandas Dataframe containing a
         users column and items column. 
    user_col: name of the user column in df
    item_col: name of the item column in df
    rating_col: name of the rating column in df (or None if for test data)
    labels: labels = [user_labels,item_labels] where:
            user_labels is a dictionary of form {user_idx_0:0,...,user_idx_(n-1):n-1}
            item_labels is a dictionary of form {item_idx_0:0,...,item_idx_(m-1):m-1}
    
    Attributes: (let N be the number of datapoints) 
    x: An N by 2 numpy integer array. 1st column represents users, 2nd column represents items.
       Both the users and items are replaced by their integer labels in dictionaries user_labels, item_labels.
    y: A 1d numpy array of length N, containing the ratings. 
    y_range: the range of y in observed data, form is [a,b].
    output_type: always = 'cont'
    """
    
    def __init__(self,df,user_col,item_col,rating_col,labels): 
        
        # Rename items and users in df according to labels, and set types for all columns. 
        df[user_col] = df[user_col].astype('category')
        df[item_col] = df[item_col].astype('category')
        df[user_col] = (df[user_col].cat.rename_categories(labels[0])).astype('int64')
        df[item_col] = (df[item_col].cat.rename_categories(labels[1])).astype('int64')
        if rating_col != None: df[rating_col] = df[rating_col].astype('float32') 
        
        # define self.x, self.y, 
        self.x = np.array(df.reindex(columns=[user_col,item_col]))
        if rating_col == None: self.y = np.zeros(len(df),'float32')
        else: self.y = np.array(df[rating_col]) 
        
        # define output_type, y_range
        self.output_type = 'cont'
        self.y_range = [float(np.min(self.y)),float(np.max(self.y))]     
                 
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    
class CollabFilterDataObj(object):
    
    """ Class for a collaborative filtering data object encompassing the datasets and 
    corresponding dataloaders for train, validation, and (optionally) test data, 
    along with a bit of extra information. 
    
    Arguments for initialization:
    train_df: pandas DataFrame of training data (same form as input to class CollabFilterDataset)
    val_df: pandas DataFrame of training data (same form as input to class CollabFilterDataset)
    test_df (optional): pandas DataFrame of test data (same form as input to class CollabFilterDataset)
    user_col: name of the user column in dataframes
    item_col: name of the item column in dataframe
    rating_col: name of the rating column in dataframes
    labels: labels = [user_labels,item_labels] where:
            user_labels is a dictionary of form {user_idx_0:0,...,user_idx_(n-1):n-1}
            item_labels is a dictionary of form {item_idx_0:0,...,item_idx_(m-1):m-1}   
    bs: the batch size to use for dataloaders
    num_workers (optional):
    
    Attributes:
    bs, labels: same as inputs
    train_ds, val_ds, test_ds: datasets of class CollabFilterDataset
    train_dl, val_dl, test_dl: dataloaders of class torch.utils.data.DataLoader (imported as DataLoader)
    """
    
        
    def __init__(self, train_df, val_df, user_col, item_col, rating_col, 
                 labels, bs, num_workers=4, test_df = None):
       
        self.bs = bs 
        self.labels = labels
        
        self.train_ds = CollabFilterDataset(train_df, user_col, item_col, rating_col, labels)
        self.val_ds = CollabFilterDataset(val_df, user_col, item_col, rating_col, labels)
        if type(test_df) == pd.DataFrame:
            self.test_ds = CollabFilterDataset(test_df, user_col, item_col, None, labels)
        
        self.train_dl = DataLoader(self.train_ds, batch_size = bs, num_workers = num_workers, shuffle=True)
        self.val_dl = DataLoader(self.val_ds, batch_size = bs, num_workers = num_workers, shuffle=False)
        if type(test_df) == pd.DataFrame:
            self.test_dl = DataLoader(self.test_ds, batch_size = bs, num_workers = num_workers, shuffle=False)
     
    @classmethod
    def from_csv(cls,train_csv,user_col,item_col,rating_col,bs,val_csv=None,test_csv=None,
                 val_idxs=None,val_frac=0.2,num_workers=4):
        
        """ Method to construct the CollabFilterDataset from csv file/s 
        
        Arguments:
        train_csv: name of csv file of training or combined training+validation data
        val_csv (optional): name of csv file of validation data
        test_csv (optional): name of csv file of test data 
        user_col: name of the user column in csv/s
        item_col: name of the item column in csv/s
        rating_col: name of the rating column in csv/s                
        val_idxs: If val_csv == None and val_idxs != None, then val_idxs should be a list of 
                 row indices in train_csv to use for validation.
        val_frac: If val_csv == None and val_idxs == None, the val_frac is the fraction of 
                 row indices in train csv to pick randomly for validation. 
                 
        Output: 
        returns a CollabFilterDataObj
        """
        
        # construct train_df and get labels for users and items
        train_df = pd.read_csv(train_csv)
        train_df = train_df.reindex(columns=[user_col,item_col,rating_col])
        users = train_df[user_col].unique()
        items = train_df[item_col].unique()
        user_labels = {users[i]:i for i in range(len(users))}
        item_labels = {items[i]:i for i in range(len(items))}     
        labels = [user_labels,item_labels]
        
        # contstruct val_df
        if val_csv != None:
            val_df = pd.read_csv(val_csv)
            val_df = val_df.reindex(columns=[user_col,item_col,rating_col])
        else:
            N = len(train_df)
            if val_idxs == None:    
                val_idxs = list(np.random.choice(np.arange(N),int(N*val_frac),replace = False))
            train_idxs = list(set(np.arange(N)) - set(val_idxs)) 
            train_df, val_df = train_df.iloc[train_idxs], train_df.iloc[val_idxs]
        
        # construct test_df
        if test_csv != None:
            test_df = pd.read_csv(test_csv)
            test_df = test_df.reindex(columns=[user_col,item_col])
        else: 
            test_df = None
               
        # return CollabFilterDataObj
        return cls(train_df, val_df, user_col, item_col, rating_col, 
                   labels, bs, num_workers, test_df)
        
        
class CollabFilterNet(nn.Module):
    
    """ Class for a (very custom) pytorch neural network model 
    to learn from structured data.
    
    Embeddings of same size are done for item and user, and then a dot product
    between the embeddings is computed, adjusted for user bias and item bias,
    and (optionally) passed through a sigmoid to compress into a desired range. 
    
    Arguments for Initialization:
    n_user: number of distinct users
    n_item: number of distinct items
    emb_dim: the dimension of the user and item embeddings
    output_range (optional): range to compress output of model into, e.g. [0,5]
                            (often may know ratings lie in a certain range like 0 to 5)   
    """

    def __init__(self, n_user,n_item,emb_dim,output_range):
        
        super().__init__()
        self.output_range = output_range
        self.user_emb = nn.Embedding(n_user,emb_dim)
        self.item_emb = nn.Embedding(n_item,emb_dim)
        self.user_bias = nn.Embedding(n_user,1)
        self.item_bias = nn.Embedding(n_item,1)
        for emb in [self.user_emb,self.item_emb,self.user_bias,self.item_bias]:
            emb.weight.data.uniform_(-0.05,0.05)
            
    def forward(self,x_batch):
        
        UserEmb, ItemEmb = self.user_emb(x_batch[:,0]), self.item_emb(x_batch[:,1])
        UserBias, ItemBias = self.user_bias(x_batch[:,0]), self.item_bias(x_batch[:,1])
        result = (UserEmb*ItemEmb).sum(dim=1) + UserBias.squeeze() + ItemBias.squeeze()
        
        if self.output_range == None: 
            return result
        else: 
            MIN,MAX = self.output_range[0], self.output_range[1]
            return MIN + (MAX - MIN)*F.sigmoid(result)
    
    @classmethod
    def from_dataobj(cls, data, emb_dim, output_range = 'default'):
        
        n_user,n_item = len(data.labels[0]),len(data.labels[1])
        if output_range == 'default': 
            MIN,MAX = data.train_ds.y_range[0],data.train_ds.y_range[1]
            output_range = [MIN - 0.05*(MAX-MIN), MAX + 0.05*(MAX-MIN)]
        return cls(n_user,n_item,emb_dim,output_range) 
    
    