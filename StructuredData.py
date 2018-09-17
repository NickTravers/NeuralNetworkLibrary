# StructuredData.py

# Problem Description:
# Given input variables X_1, ..., X_m and wish to predict value of an output variable Y.
# Inputs X_i may be categorical or real-valued variables, or some combination of the two types. 
# Output variable Y may be either categorical or real-valued. 
# All variables should have a natural conceptual meaning, and the inputs are 
# generally assumed to be heterogeneous. 

# Example: Car insurance
# X_1 = age of cutomer (may be treated as categorical or continuous, assuming age is known only as an integer) 
# X_2 = gender of customer (categorical)
# X_3 = total amount of claims by customer in past year (continuous)
# X_4 = whether of not customer had an accident in past year (categorical, "yes" or "no") 
# Possible Output 1: Y = total amount of claims by custormer in next year (continuous)
# Possible Output 2: Y = whether or not customer will have accident in next year (categorical)

# In order to make predictions we are given n data points of (input + output) 
# of form (x_1,...,x_m, y) corresponding to realizations of variables (X_1,..., X_m, Y).
# The starting point for doing learning with classes and methods in this file StructuredData.py 
# is the function ProcessDataFrame, which assumes data is in a pandas DataFrame of following form:

#  (index)   age gender   claims_previous_year  accident_previous_year claims_following_year
# 'Person1'  25   'M'     0.00                  'No'                    0.00
# 'Person2'  37   'F'     0.00                  'No'                    0.00
# 'Person3'  46   'F'     452.00                'Yes'                   337.25 
#    .
#    .
#    .
# 'Person_n' 62  'M'     246.30                'Yes'                   925.05
 

# Actually, and more precisely, seperate DataFrames of the above form for train and validation data 
# should be passed into the function ProcessDataFrame. And (optionally) a DataFrame of the above form
# with the output variable column removed can also be passed into the function ProcessDataFrame
# for test data. (If a single DataFrame which contains the output variable column is given, it can 
# be split into seperate train and validation DataFrames using the function SplitDataFrameTrainVal.) 

# NOTE 1: Normally the data will be given in .csv files (or possibly excel files) and will have to 
#         be read in by the user to create a pandas DataFrame. However, often the user may also
#         want to do some feature engineering on the given data. More precisely, the given input variables 
#         in the .csv file may be (X1,...Xm), but the user may want to create from them new input variables
#         (X1_hat, ..., Xn_hat) to use as the inputs. For instance, an input variable might be the date of a 
#         business transaction, but the user may want to extract from the date the corresponding <day_of_week>
#         and <month_of_year> variables, to use explicitly as inputs. 

#         IT IS ASSUMED THAT ALL FEATURE ENGINEERING HAS BEEN DONE PRIOR TO APPLYING ANY OF 
#         THE METHODS IN THIS FILE StructuredData.py. THE INPUT VARIABLES IN THE DATAFRAME" <df> 
#         PASSED TO THE FUNCTION <ProcessDataFrame(df,....)> ARE ASSUMED TO BE THE VARIABLES 
#         WE WILL USE TO DIRECTLY BUILD OUR MODEL. 

# NOTE 2: In many instances when dealing with real data there may be missing values for some of the 
#         input variables for some of the data points. For instance, in the car insurance example 
#         we may not know the amount of claims the previous year for a particular person, because
#         that person had car insurance with a different company the previous year. The ProcessDataFrame
#         function allows for missing values in some of the inputs. (The docstring for that 
#         function explains how missing values are dealt with.)


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

# 1. def SplitDataFrameTrainVal
# 2. def ProcessDataFrame
# 3. class StructuredDataset
# 4. class StructuredDataObj
# 5. class StructuredDataNet 


def SplitDataFrameTrainVal(df,val_idxs=None,val_frac=0.2):
    
    """ Function to split a single DataFrame into 2 DataFrames: one for training 
    and one for validation. The rows to use for validation are chosen randomly
    unless val_idxs is specified.
        
    Arguments:
    df: the DataFrame to split 
    val_idxs: If val_idxs != None, then val_idxs should be a list of indices between 0 and len(df)-1
              which are the indices of the rows of df used for validation. In this case, 
              val_frac is irrelevant. 
    val_frac: Assuming val_idxs == None, val_frac is the fraction of datapoints to use for validation. 
              Example: If val_frac == 0.2, then 20% of datapoints used for validation.
    
    Ouput: 
    Returns 2 objects: train_df, val_df. 
    """
    
    N = len(df)
    if val_idxs == None:
        val_idxs = list( np.random.choice(np.arange(N),int(N*val_frac),replace = False) )
    train_idxs = list(set(np.arange(N)) - set(val_idxs))                       
    train_df, val_df = df.iloc[train_idxs], df.iloc[val_idxs] 
    return train_df, val_df


def ProcessDataFrame(df, cat_vars, cont_vars, output_var, scale_cont, fill_missing = 'median', 
                     category_labels = None, unknown_category = True):
    
    """ Function to pre-process a pandas DataFrame of structured data 
        before training a neural network with it.
   
Arguments:
df: A pandas DataFrame of the form given above in car insurance example.
cat_vars: A list of the column names of categorical variables in <df>.
          These do NOT have to be of type pandas.Categorical in <df>, 
          the function <ProcessDataFrame> will ensure they are treated that way.           
cont_vars: A list of the column names of continuous variables in <df>.
           These may be stored as either float or integer in <df>.
output_var: * If <df> is for test data, set output_var = None. 
            * If <df> is for train or validation data, set ouput_var to be the 
              column name of the output variable in <df>. 
              
(NOTE: output_var should be included in either the list cat_vars or the list cont_vars)
              
scale_cont: * If <scale_cont> == 'No' then function does no rescaling of continuous variables. 
            * If <scale_cont> == 'by_df', then function rescales all continous input variables
              in <df> to have (empirical) mean = 0 and StdDev = 1, in the returned object <xcont_df>. 
            * If <scale_cont> is a dictionary object with entries of form << cont_var_name: [mean,StdDev] >>, 
              then each continous input variable of <df> is rescaled in the returned object <xcont_df>, 
              by subtracting given mean and then dividing by given StdDev.
              
fill_missing: Method to fill in missing values for continous variables.
              Choices are 'mean','median', or c (where c is a constant, given as float or integer). 
              Default is 'median'. 
              
category_labels: * If 0 categorical variables OR running function on a DataFrame of training 
                   data, then leave as the default, <category_labels> = None.
                 * If 1 or more categorical variables and running on a Dataframe of 
                   val or test data, then use <category_labels> = <category_labels>,
                   where the right hand side of the "=" is the output of the function 
                   ProcessDataFrame when run on the training DataFrame. 
                   
unknown_category: If unknown_category == True, adds an extra 'unknown' category to all categorical variables
                  and calls missing entries 'unknown'. 
  
              
Output: 
Returns 5 objects: xcat_df, xcont_df y, scaling_values, category_labels 

xcat_df:  A dataframe whose columns correspond to the columns of <df> with categorical input variables. 
          For each column the following processing steps are applied:
          * If <unknown_category> == True: Existing categories are renamed 1,2,3... and 
            all missing values are replaced by another category 0. 
          * If <unknown_category> == False: Existing categories are renamed 0,1,2...
            (Assumes no missing values). 
             
xcont_df: A dataframe whose columns correspond to the columns of <df> with continuous input variables.
          The following processing steps are done to these columns:
          (1) All missing values of continuous variables from <df> are replaced by either mean, median, 
              or a constant, as specified by <fill_missing>. 
          (2) Continuous variables rescaled (or not) according to the value of <scale_cont>. 
              This rescaling is done after the filling in of any missing values in step (1).

y: * If <df> is for test data (so that there is no <output_var> column) then y = None. 
   * If <df> is for training or validation then y is the <output_var> column of <df> in list form.  
     If <output_var> is categorical, categories also renamed 0,1,2, ... in y.  
    
scaling_values: * If <scale_cont> == 'No', then is equal to None. 
                * If <scale_cont> != 'No', then is a dictionary with 
                  entries of form << cont_var_name: [mean, StdDev]. >>
                               
category_labels: A list with entries which are dictionaries. 
                 The ith dictionary gives the labels for the ith categorical variable. 
                 The order of categorical variables is:
                 First, the input categorical variables in same order as <xcat_df>.
                 Last, the output variables if it is categorical. 
                 
                 Example: Assume ith cagtegory is 'gender' and 'unknown' has label 0,
                 'M' has label 1, and 'F' has label 2 in the returned Dataframe <xcat_df>. 
                 Then the dictionary would be: {'unknown':0, 'M':1, 'F':2}.
                 
                 
NOTE ON RESCALING CONTINUOUS INPUT VARIABLES:

If you want to rescale continuous inputs variables (which normally should) 
then following steps should be done in order: 

(1) Run ProcessDataFrame on <dftrain> with <scale_cont> = 'by_df' and <category_labels> = None.

(2) Run ProcessDataFrame on <dfval> with
    <category_labels> = <category_labels> and <scale_cont> = <scaling_values>,
    where right hand side of each "=" is the output from step (1).    
    
(3) If there is test data, run ProcessDataFrame on <dftest> with
    <category_labels> = <category_labels> and <scale_cont> = <scaling_values>, 
    where right hand side of each "=" is the output from step (1). 
      
   """
    
    # Make lists of categorical and continuous input variables
    xcat_vars,xcont_vars = cat_vars.copy(), cont_vars.copy()
    if output_var in cat_vars: xcat_vars.remove(output_var)
    if output_var in cont_vars: xcont_vars.remove(output_var)
    
    # Ensure proper types of cat and cont variables in df
    for var in cat_vars: df[var] = df[var].astype('category')
    for var in cont_vars: df[var] = df[var].astype('float32')
    
    # Check if need to construct dictionary for scaling_values and list for category_labels. 
    # If so, these variables are initialized as empty dictionary and empty list. 
    need_catlabels = False
    if category_labels == None: 
        category_labels = []
        need_catlabels = True    
    if len(xcont_vars) > 0 and scale_cont == 'by_df': scaling_values = {}        
    elif len(xcont_vars) > 0 and type(scale_cont) == dict: scaling_values = scale_cont    
    else: scaling_values = None
   
    # Output variable y
    if output_var == None: 
        y = None    
    elif output_var in cont_vars: 
        y = list(df[output_var])
    elif output_var in cat_vars:
        if need_catlabels == True:
            y_cats = df[output_var].unique()
            y_cat_labels = {y_cats[i]:i for i in range(len(y_cats))} 
            y = list( df[output_var].cat.rename_categories(y_cat_labels) )
        else:
            y = list( df[output_var].cat.rename_categories(category_labels[-1]) )
    
    # Construct xcat_df. 
    # (Along the way, the list category_labels also built if necessary.)
    if len(xcat_vars) > 0: 
        xcat_df = df.reindex(columns= xcat_vars) 
        for j,var in enumerate(xcat_vars):
            if need_catlabels == True and unknown_category == True:
                var_cats = xcat_df[var].unique()
                Dict = {var_cats[i]:i+1 for i in range(len(var_cats))} 
                Dict['unknown'] = 0
                category_labels.append(Dict)
            elif need_catlabels == True and unknown_category == False:
                var_cats = xcat_df[var].unique()
                Dict = {var_cats[i]:i for i in range(len(var_cats))} 
                category_labels.append(Dict)
            else: 
                Dict = category_labels[j]
        
            if unknown_category == True:
                xcat_df[var] = xcat_df[var].cat.add_categories(['unknown'])
                xcat_df[var] = xcat_df[var].fillna('unknown')
            xcat_df[var] = xcat_df[var].cat.rename_categories(Dict)
            xcat_df[var] = xcat_df[var].astype('int64') 
                    
    else: xcat_df = 'empty'
        
    if need_catlabels == True and output_var in cat_vars: 
        category_labels.append(y_cat_labels)
        
    # Construct xcont_df. 
    # (Along the way, the dictionary scaling_values also built if necessary.) 
    if len(xcont_vars) > 0:
        xcont_df = df.reindex(columns= xcont_vars)
        if fill_missing == 'median': 
            xcont_df = xcont_df.fillna(xcont_df.median())
        elif fill_missing == 'mean': 
            xcont_df = xcont_df.fillna(xcont_df.mean())
        else:
            filler = pd.Series(fill_missing,index = xcont_vars)
            xcont_df = xcont_df.fillna(filler)     
        if scale_cont == 'by_df':
            for var in xcont_vars:
                mean = xcont_df[var].mean()
                std = xcont_df[var].std()
                xcont_df[var] = (xcont_df[var] - mean)/std
                scaling_values[var] = [mean,std]
        elif type(scale_cont) == dict:
            for var in xcont_vars:
                mean = scale_cont[var][0]
                std = scale_cont[var][1]
                xcont_df[var] = (xcont_df[var] - mean)/std
                
    else: xcont_df = 'empty' 
    
    # Return the output
    return xcat_df, xcont_df, y, scaling_values, category_labels


class StructuredDataset(Dataset):
    """
    A class for a dataset of structured data (can be either train, val, or test). 
    Inherits from class torch.utils.data.Dataset (imported as Dataset). 
    
    Arguments for initialization:
    xcat_df, xcont_df, y: these are the outputs of function ProcessDataFrame. 
    output_type: type of the output variable, either 'cont' or 'cat'. 
    
    Attributes:
    output_type: Same as input.
    x_cat, x_cont, y: Same as inputs xcat_df, xcont_df, y but in form of numpy arrays. 
                      If any of inputs is equal to None or 'empty' then corresponding 
                      attribute is an array of zeros. 
    n_cat: number of categorical input variables
    n_cont: number of continuous input variables
    """
    
    def __init__(self,xcat_df,xcont_df,y,output_type): 
        
        # Note: if type(xcat_df) == str, then have xcat_df = 'empty'
        if type(xcat_df) != str: L = len(xcat_df)
        else: L = len(xcont_df) 
        
        self.output_type = output_type
        
        if output_type == 'cat' and y != None: self.y = np.array(y).astype('int')
        elif output_type == 'cont' and y != None: self.y = np.array(y).astype('float32')
        else: self.y = np.zeros(L).astype('float32')
        
        if type(xcat_df) == str: 
            self.n_cat = 0
            self.x_cat = np.zeros(L,'float32')
        else: 
            self.n_cat = xcat_df.shape[1] 
            self.x_cat = np.array(xcat_df)
        
        if type(xcont_df) == str: 
            self.n_cont = 0
            self.x_cont = np.zeros(L,'float32')
        else: 
            self.n_cont = xcont_df.shape[1]
            self.x_cont = np.array(xcont_df)
            
    # Note: len(dataset) returns length of a StructuredDataset object called "dataset"
    def __len__(self):
        return len(self.x_cat)

    # Note: dataset[i] returns ith element in a StructuredDataset object called "dataset". 
    def __getitem__(self, idx):
        return self.x_cat[idx], self.x_cont[idx], self.y[idx]
    
    def y_range(self):
        return [np.min(self.y),np.max(self.y)]
        
    
class StructuredDataObj(object):
    
    """ Class for a structured data object encompassing the datasets and corresponding dataloaders 
    for train, validation, and (optionally) test data, along with a bit of extra information. 
    
    Arguments for initialization:
    train_ds: training dataset of class StructuredDataset 
    val_ds: validation dataset of class StructuredDataset
    test_ds (optional): test dataset of class StructuredDataset 
    category_labels: output of function ProcessDataFrame (applied to original training DataFrame <df>). 
    scaling_values: output of function ProcessDataFrame (applied to original training DataFrame <df>). 
    bs: the batch size to use for dataloaders
    num_workers (optional): ...
    
    Attributes:
    train_ds, val_ds, test_ds, category_labels, scaling_values, bs, num_workers (exact same as inputs).
    train_dl, val_dl, test_dl: dataloaders for corresponding types of data 
                                All of class torch.utils.data.DataLoader (imported as DataLoader)
    """
    
        
    def __init__(self, train_ds, val_ds, category_labels, scaling_values,
                 bs, num_workers=4, test_ds = None):
       
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.category_labels = category_labels
        self.scaling_values = scaling_values
        self.bs = bs
        self.num_workers = num_workers
        
        self.train_dl = DataLoader(train_ds, batch_size = bs, num_workers = num_workers, shuffle=True)
        self.val_dl = DataLoader(val_ds, batch_size = bs, num_workers = num_workers, shuffle=False)
        self.test_dl = DataLoader(test_ds, batch_size = bs, num_workers = num_workers, shuffle=False)
        
    @classmethod
    def from_dataframes(cls, train_df, val_df, cat_vars, cont_vars, output_var, bs,
                        fill_missing = 'median', scale_cont = True, unknown_category = True,
                        num_workers = 4, test_df = None):
        
        if output_var in cat_vars: output_type = 'cat'
        if output_var in cont_vars: output_type = 'cont'
            
        if scale_cont == True:
            
            xcat_df, xcont_df, y, scaling_values, category_labels = \
            ProcessDataFrame(train_df, cat_vars, cont_vars, output_var,'by_df', 
                             fill_missing, None, unknown_category)
            train_ds = StructuredDataset(xcat_df,xcont_df,y,output_type)
            
            xcat_df, xcont_df, y, scaling_values, category_labels = \
            ProcessDataFrame(val_df, cat_vars, cont_vars, output_var, scaling_values, 
                             fill_missing, category_labels, unknown_category)
            val_ds = StructuredDataset(xcat_df,xcont_df,y,output_type)
            
            if type(test_df) == pd.DataFrame:
                xcat_vars,xcont_vars = cat_vars.copy(), cont_vars.copy()
                if output_var in cat_vars: xcat_vars.remove(output_var)
                if output_var in cont_vars: xcont_vars.remove(output_var)
                xcat_df, xcont_df, y, scaling_values, category_labels = \
                ProcessDataFrame(test_df, xcat_vars, xcont_vars, None, scaling_values, 
                                 fill_missing, category_labels, unknown_category)
                test_ds = StructuredDataset(xcat_df,xcont_df,y,output_type)
            else: test_ds = None
            
        if scale_cont == False:
            
            xcat_df, xcont_df, y, scaling_values, category_labels = \
            ProcessDataFrame(train_df, cat_vars, cont_vars, output_var, 'No', 
                             fill_missing, None, unknown_category)
            train_ds = StructuredDataset(xcat_df,xcont_df,y,output_type)
            
            xcat_df, xcont_df, y, scaling_values, category_labels = \
            ProcessDataFrame(val_df, cat_vars, cont_vars, output_var,'No', 
                             fill_missing, category_labels, unknown_category)
            val_ds = StructuredDataset(xcat_df,xcont_df,y,output_type)
            
            if type(test_df) == pd.DataFrame:
                xcat_vars,xcont_vars = cat_vars.copy(), cont_vars.copy()
                if output_var in cat_vars: xcat_vars.remove(output_var)
                if output_var in cont_vars: xcont_vars.remove(output_var)
                xcat_df, xcont_df, y, scaling_values, category_labels = \
                ProcessDataFrame(test_df, xcat_vars, xcont_vars, None,'No', 
                                 fill_missing, category_labels, unknown_category)
                test_ds = StructuredDataset(xcat_df,xcont_df,y,output_type)             
            else: test_ds = None
                
        return cls(train_ds, val_ds, category_labels, scaling_values, bs, num_workers, test_ds)

                
class StructuredDataNet(nn.Module):
    
    """ Class for a pytorch neural network model to learn from structured data.
    
    Embeddings are done for categorical variables, and then output of these embeddings
    combined with values of continuous variables is passed through a (user specified)
    fully connected network of arbitrary depth. 
    
    Arguments for Initialization:
    
    n_cat: The number of categorical input variables.
    n_cont: The number of continuous input variables.
    
    category_labels = output of function ProcessDataFrame (applied to original training DataFrame <df>) 
    
    emb_sizes: A list of tuples of form (c,d) where the ith tuple has:
               c = number of categories of ith input categorical variable.
               d = dimension of the embedding of ith input categorical variable. 
               (Ordering of categorical variables is same as in xcat_df) 
               
               If <emb_sizes> == 'default', following formula is used for all cat variables:
               d = min(ceil(c/2),50).
                   
    fc_layer_sizes: A list specifying the sizes of the fully connected layers. 
                    For example if layer sizes = [50,20,10] then:
                    fc_layer1 has N inputs, 50 outputs
                    fc_layer2 has 50 inputs, 20 outpus
                    fc_layer3 has 20 inputs, 10 outputs
                    10 output activations from the network
                 
                    Here N = (sum of embedding dimesions of all categorical variables) + n_cont.
                 
    dropout_levels: A tuple (emb_drop, cont_drop, other_drop) where:
                    - emb_drop is the dropout prob to use on output of embeddings.
                    - cont_drop is the dropout prob to use for each continuous input variable.
                    - other_drop = [d1=0,d2,d3,...d_n] with n = len(fc_layer_sizes).
                      d_i = dropout prob applied BEFORE passing through ith linear layer of fully connected network.
                      Note: d1 = 0, since dropout seperately applied to embedding and continuous variables,
                            before passing through 1st linear layer.
                            
                    Continuing example above, if dropout_levels = (0.01,0.02,[0,0.3,0.1]) then
                    dropout of 0.01 applied to output of embeddings
                    dropout of 0.02 applied to each continuous input variable 
                            (i.e. each such variable set to 0, with prob 0.02)
                    0 additional dropout before 1st linear layer
                    dropout of 0.3 before 2nd linear layer
                    dropout of 0.1 before 3rd linear layer 
                    
                    Default is dropuout_levels = None, which is interpreted same as (0,0,[0,...,0]).
                          
    output_type: The type of the output variable, either 'cont' or 'cat'. 
   
    output_range: * If output_type == 'cont' and you want to ensure the network only returns values
                    of the output variable in the range [a,b], then set output_range = [a,b].
                  * If output_type == 'cat' OR output_type == 'cont', but you do not want ensure
                    the output variable stays within a prespecified range, then leave as default 
                    output_range = None. 
                          
    use_bn = If use_bn == True, batch norm is used in training mode for fully connected part of network 
                                (not including input to 1st linear layer).
             If use_bn == False, no batch norm is used in training mode. 
             (Batch norm never used in other parts of the network, or in validation or test mode)
    
    """

    def __init__(self, output_type, n_cat, n_cont, category_labels, fc_layer_sizes, 
                 emb_sizes = 'default', output_range=None, dropout_levels = None, use_bn = True):
        
        super().__init__()
        self.n_cat = n_cat
        self.n_cont = n_cont
        
        # set emb_sizes to default values if no emb_sizes specified
        if emb_sizes == 'default': 
            if output_type == 'cont': cat_sizes = [len(Dict) for Dict in category_labels]
            else: cat_sizes = [len(Dict) for Dict in category_labels[0:-1]] 
            emb_sizes = [ (c, min(50, int(np.ceil(c/2)))) for c in cat_sizes ]
        
        # define embeddings            
        self.embeddings = nn.ModuleList( [nn.Embedding(c,d) for c,d in emb_sizes] )
        for emb in self.embeddings: 
            emb_weights = emb.weight.data
            b = 2/(emb_weights.size(1)+1)
            emb_weights.uniform_(-b,b)
        
        # define dropout for continuous and embeddged-categorical inputs
        if dropout_levels == None: dropout_levels = (0,0,None)
        self.emb_drop = nn.Dropout(dropout_levels[0])
        self.cont_drop = nn.Dropout(dropout_levels[1])
        
        #define fully connected part of the network
        total_emb_dim = sum(d for c,d in emb_sizes)
        layer_sizes = [total_emb_dim + n_cont] + fc_layer_sizes 
        if output_type == 'cont' and output_range != None: 
            final_activation_func = 'sigmoidal'
        else: final_activation_func = None
            
        self.fc_net = FullyConnectedNet(layer_sizes, dropout_levels[2], 'relu', 
                                        final_activation_func, output_range, use_bn=[False,use_bn])
                
    def forward(self, xcat_batch, xcont_batch):
        
        if self.n_cat > 0:
            cat_inputs = [emb(xcat_batch[:,i]) for i,emb in enumerate(self.embeddings)]                 
            cat_inputs = torch.cat(cat_inputs,dim=1)
            cat_inputs = self.emb_drop(cat_inputs)
        if self.n_cont > 0:
            cont_inputs = self.cont_drop(xcont_batch)
            
        if self.n_cat == 0: combined_inputs = cont_inputs
        elif self.n_cont == 0: combined_inputs = cat_inputs
        else: combined_inputs = torch.cat([cat_inputs, cont_inputs], dim=1)
        
        return(self.fc_net(combined_inputs))
    
    @classmethod
    def from_dataobj(cls, data, fc_layer_sizes, emb_sizes = 'default', 
                     output_range=None, dropout_levels = None, use_bn = True):
        
        output_type = data.train_ds.output_type
        n_cat = data.train_ds.n_cat
        n_cont = data.train_ds.n_cont
        category_labels = data.category_labels
        
        return cls(output_type, n_cat, n_cont, category_labels, fc_layer_sizes, 
                   emb_sizes, output_range, dropout_levels, use_bn)
    
    