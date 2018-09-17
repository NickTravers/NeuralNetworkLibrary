# General.py

# This file contains general functions and classes that can be used in different 
# deep learning settings (e.g. Structured Data, Image Data, NLP, ... etc.). 

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# List of Functions and Classes

# 1. def VAR
# 2. der ARR
# 3. class FullyConnectedNet
# 4. class Learner


def VAR(x):
    
    """ This function takes an input x which may be of type np.ndarray (on cpu) 
    or either torch.Tensor or torch.LongTensor (on gpu or cpu) and converts 
    it to a torch Variable on gpu. """
    
    if isinstance(x,(torch.FloatTensor,torch.LongTensor)):
        return Variable(x).cuda()
    
    elif isinstance(x,(torch.cuda.FloatTensor,torch.cuda.LongTensor)):
        return Variable(x)
    
    elif type(x) == np.ndarray and x.dtype == ('float64' or float):
        return Variable(torch.Tensor(x)).cuda()
    
    elif type(x) == np.ndarray and x.dtype == ('int64' or int):
        return Variable(torch.LongTensor(x)).cuda()
    
def ARR(x):
    
    """ This function takes an input x which may be of type torch.Tensor, torch.LongTensor
    Variable(torch.Tensor), or Variable(torch.LongTensor), on either cpu or gpu, and 
    converts it to a numpy array (on cpu). """
    
    if type(x) == Variable: return np.array(x.data)
    else: return np.array(x)   
        
class FullyConnectedNet(nn.Module):
    
    """ General class for a multi-level fully connected neural network. 
    
        A user specified nonlinear activation function is applied 
        after each fully connected linear layer. Batch normalization
        and dropout may also be applied.
 
    Arguments:
    layer_sizes: A list specifying the sizes of the layers. 
                 For example if layer sizes = [100,50,20,10] then:
                 100 input activations to the network
                 fc_layer1 has 100 inputs, 50 outputs
                 fc_layer2 has 50 inputs, 20 outpus
                 fc_layer3 has 20 inputs, 10 outputs
                 10 output activations from the network
                 
    dropout_levels: The amount of dropout to use in each fc layer.
                    Continuing example above, if dropout_levels = [0.5,0.3,0.1] then
                    fc_layer1 has 0.5 dropout, fc_layer2 has 0.3 dropout, fc_layer3 has 0.1 dropout.
                   
    inter_layer_activation_func = Activation function to apply after every fully connected linear 
                                  layer except the final one. Choices are 'sigmoid','tanh','relu'. 
                                  Default is 'relu'. 
    
    final_activation_func: Activation function to apply to the output of final fully connected 
                           linear layer, before returning output of network itself. Choices are:
                           None, 'softmax', 'sigmoidal'. If 'sigmoidal' the parameter <output_range>
                           must also be specified.
                          
    output_range: If final_activation_func == 'sigmoidal' and output_range = [a,b] then 
                  outputs of final linear layer are compressed to the range [a,b] using 
                  an appropriately scaled and shifted sigmoid function. 
                  If final_activation_func != 'sigmoidal', leave as the default, output_range = None. 
                          
    use_bn: A list use_bn = [use_bn_input,use_bn_other].
            * If use_bn_input == True, batch norm is used in training mode on the input to the network
              BEFORE passing through 1st linear layer. If use_bn_input == False, then not used. 
            * If use_bn_other == True, batch norm used in training mode BEFORE passing through
              each linear layer after the 1st. If use_bn_other == False, then not used.
            (Batch norm never used in validation or test mode)
            
            If you set use_bn = True, then function behaves same as use_bn = [True,True]
            If you set use_bn = False, then function behaves same as use_bn = [False,False] 
           
    """
        
    def __init__(self,layer_sizes, dropout_levels = None, inter_layer_activation_func = 'relu',
                 final_activation_func = None, output_range=None, use_bn = True):
        
        super().__init__()
        self.inter_layer_activation_func = inter_layer_activation_func
        self.final_activation_func = final_activation_func
        self.output_range = output_range
        if use_bn == True: self.use_bn = [True,True]
        elif use_bn == False: self.use_bn = [False,False]
        else: self.use_bn = use_bn
        
        N = len(layer_sizes) - 1
        if dropout_levels == None: dropout_levels = list(np.zeros(N))
        self.dropoutlayers = nn.ModuleList([nn.Dropout(dropout_levels[i]) for i in range(N)])
        self.batchnormlayers = nn.ModuleList([nn.BatchNorm1d(layer_sizes[i]) for i in range(N)])    
        self.linlayers = nn.ModuleList([nn.Linear(layer_sizes[i],layer_sizes[i+1]) for i in range(N)])
        for lin in self.linlayers: nn.init.kaiming_normal(lin.weight.data)
        
    def forward(self,x): 
        
        N = len(self.linlayers)
        for i,(lin,drop,bn) in enumerate(zip(self.linlayers,self.dropoutlayers,self.batchnormlayers)):
    
            # batchnorm, dropout, and pass through linear layer 
            # (Note: batch norm only works on a minibatch x with len(x) > 1.) 
            if i == 0 and self.use_bn[0] == True and len(x) > 1: x = bn(x) 
            elif i > 0 and self.use_bn[1] == True and len(x) > 1: x = bn(x)
            x = drop(x)
            x = lin(x)
            
            # apply nonlinear activation 
            # (Note: different for final layer, and other layers)
            if i < N-1:
                if self.inter_layer_activation_func == 'relu': x = F.relu(x)
                elif self.inter_layer_activation_func == 'tanh': x = F.tanh(x)
                elif self.inter_layer_activation_func == 'sigmoid': x = F.sigmoid(x)  
            if i == N-1:
                if self.final_activation_func == 'softmax': x = (F.log_softmax(x,dim=1)).exp()
                elif self.final_activation_func == 'sigmoidal':
                    MIN,MAX = float(self.output_range[0]), float(self.output_range[1])
                    x = MIN + (MAX - MIN)*F.sigmoid(x)
        
        return x
    

class Learner(object):
    
    """ General class for a neural network model to learn from data 
    (can be structured data, image data, NLP data, ....etc.). 
    
    The class contains the pytorch model of the neural network, and also the data, 
    the optimizer, and the loss function. 
    
    Arguments for initialization:
    PATH: a path where info related to the Learner object is stored.
    data: a data object of class StructuredDataObj (later ImageClassificationDataObj, TextClassificationDataObj,...) 
    model: a pytorch model of class StructuredDataNet (later ImageClassificationNet, TextClassificationNet,...) 
    optimizer: Optimizer to use, e.g. optim.Adam(model.parameters(),lr=0.01). 
    loss_func: Loss function to use, e.g. nn.MSELoss() or nn.CrossEntropyLoss().  
    
    Attributes:
    PATH, data, model, optimizer, loss_func: all same as inputs.
    lr_sched: The learning rate schedule, a sequence of learning rates used on succesive minibatches.
              Resets to the empty list every time learner.fit(...) or learner.find_lr() is called. 
    loss_sched: A sequence of losses obtained on succesive minibatches in training or when  
                using learner.find_lr(). Sequence resets to the empty list every time 
                learner.fit(...) or learner.find_lr() is called.
    output_type: type of output variable y, either 'cat' or 'cont'.
    """
    
    # Note: For structured data the train and validation dataloaders return batches in form
    #       [xcat_batch, xcont_batch, y_batch], and the test dataloader returns batches in form 
    #       [xcat_batch, xcont_batch]. 
    
    #       For other types of data (e.g. image data, NLP, ...) I think the train and validation 
    #       dataloader will return batches in form [x_batch, y_batch] and the test dataloader
    #       will simply return x_batch.  
    # 
    #       The learner class has been written to be compatible with both structured data and 
    #       other types. This makes some of the methods slightly strange in the first few lines
    #       of code, because they have to determine what type of input they are receiving.    
    
    def __init__(self, PATH, data, model, optimizer='Adam', loss_func='default'):
        
        os.makedirs(PATH + 'models', exist_ok=True)
        
        self.PATH = PATH 
        self.data = data 
        self.model = model.cuda()
        self.output_type = data.train_ds.output_type
        self.lr_sched = []
        self.loss_sched = []
        
        if optimizer == 'Adam': 
            self.optimizer = optim.Adam(model.parameters()) 
        else: self.optimizer = optimizer 
        
        if loss_func == 'default' and self.output_type == 'cont': 
            self.loss_func = nn.MSELoss()
        elif loss_func == 'default' and self.output_type == 'cat': 
            self.loss_func = nn.CrossEntropyLoss()
        else: self.loss_func = loss_func
        
    
    # (1) METHODS - FOR SAVING AND LOADING MODELS
    
    def save(self,filename):
        """ Function to save Learner object. 
        
            Example: If you have a Learner object called <learner>, then learner.save('224') 
            saves a file called '224.pt' containing the learner model-state and optimizer-state 
            into the folder learner.PATH/models. """
            
            # NOTE: '.pt' is (I believe) a pytorch format that works with pickle.
            
        learner_state = {'model_state': self.model.state_dict(),
                         'optimizer_state' : self.optimizer.state_dict()} 
  
        torch.save(learner_state, self.PATH + 'models/' + filename + '.pt')
 
       
    def load(self,filename):
        """ Function to load the model-state and optimizer-state from file for 
            a learner object. The learner must already be created and have 
            a model and optimizer of correct type.
          
            Example: If learner is an object of class Learner then
            learner.load('224') loads a file called '224.pt' in folder
            learner.PATH/models, and sets learner.model.state_dict and 
            learner.optimizer.state_dict to the values given in the file. 
            """
            
            # NOTE: The model-state includes all the weights and biases.
            #       The optimizer-state includes the number of time steps of 
            #       training for Adam. 
        
        filename = self.PATH + 'models/' + filename + '.pt'
        if os.path.isfile(filename):
            learner_state = torch.load(filename)
            self.model.load_state_dict(learner_state['model_state'])
            self.optimizer.load_state_dict(learner_state['optimizer_state'])
        else:
            print("no file found at '{}'".format(filename))
    
    
    # (2) METHODS FOR PLOTTING lr_sched and loss_sched
    
    @staticmethod
    def smooth_timeseries(s,r):
        """simple function to smooth a timeseries sequence using local averaging.
        
        Arguments:
        s: timeseries sequence to smooth (of type list). 
        r: radius of smoothing, positive integer (much smaller than len(s)). 
        
        Output:
        Returns a smoothed time series <s_smooth> of same length as <s> in which, 
        s_mooth[i] = (s[i-r] + s[i-r+1] + ... + s[i+r])/(2r+1).
        (Suitable correction applied for points near boundaries 0 and N-1.)
        
        """
        N = len(s)
        s_smooth = np.zeros(N)
        
        # Compute smoothed values for points within distance r of boundaries
        for i in range(r):
            s_smooth[i] = sum(x for x in s[0:2*i+1])/(2*i+1)
            s_smooth[N-1-i] = sum(x for x in s[N-1-2*i:N+1])/(2*i+1)
            
        # Compute smoothed values for other points
        for i in range(r,N-r):
            s_smooth[i] = sum(x for x in s[i-r:i+r+1])/(2*r+1)
            
        return list(s_smooth)
    
    def plot_loss_sched(self,smoothing_radius='default'):
        
        """ Function to plot the loss schedule, with user defined smoothing.
            Actually plots self.smooth_timeseries(self.loss_sched,smoothing_radius). """
        
        if smoothing_radius == 'default':
            smoothing_radius = max(2,int(len(self.data.train_dl)/100))
        smoothed_loss_sched = self.smooth_timeseries(self.loss_sched,smoothing_radius)
        plt.plot(smoothed_loss_sched)
        plt.xlabel('minibatch')
        plt.ylabel('train loss')
    
    def plot_lr_sched(self):
        plt.plot(self.lr_sched)
        plt.xlabel('minibatch')
        plt.ylabel('learning rate')
    
        
    # (3) CORE METHODS FOR MAKING PREDICTIONS + EVAULATING, AND TRAINING MODEL 
    
    def set_lr(self,new_lr):
        """This function changes the learning rate for self.optimizer to <new_lr>."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def predict1minibatch(self, x_batch):
        """Function to predict the output for a single minibatch of inputs x_batch. 
           x_batch may be a Tensor, a list containing 1 Tensor, or a list 
           containg 2 Tensors (of same length). The output y_pred is a Variable."""
        
        if type(x_batch) == list and len(x_batch) == 2: 
            y_pred = self.model(VAR(x_batch[0]),VAR(x_batch[1]))
        elif type(x_batch) == list and len(x_batch) == 1:
            y_pred = self.model(VAR(x_batch[0])) 
        else:
            y_pred = self.model(VAR(x_batch))
        
        #reshape y_pred for compatibility with loss functions, if necessary
        if y_pred.dim() == 2 and y_pred.shape[1] == 1: 
            y_pred = y_pred.view(len(y_pred))
            
        return y_pred
        
    def predict(self, softmax_cats = True, test=False):
        
        """ Function to make predictions for either the whole validation set or whole test set. 
            
        Arguments:    
        test: If <test> == True predicts for test set, if <test>==False predicts for validation set. 
        softmax_cats: If <self.output_type> = 'cat' and <softmax_cats> == True, then returns as predictions the 
                      softmax function of output activations from the network instead of the output 
                      activations themselves. This is useful because normally we will not apply a softmax function
                      on final activations in the network, for compatibility with nn.CrossEntropyLoss() function.  
                
        Output:
        returns 1 object: predictions 
        
        Let N be the length of the test or validation set that is being used.
        * If self.output_type = 'cont': predictions is a 1d numpy array of length N, 
          predictions[i] = predicted continuous output for ith input. 
        * If self.output_type = 'cat': predictions = [pred_probs,pred_labels] where:  
          - pred_probs is an N by C numpy array, C = number of categories, and 
            pred_probs[i,j] is probabability ith input is of category j
            (or the pre-softmax-ed version of this probability if softmax_cats = false). 
          - pred_labels is a 1d integer numpy array of length N, pred_labels[i] is 
            the most likely category in {0,1,...C-1} for ith input.
        """
   
        self.model.eval()
        predictions = []
        
        if test==True:
            for *x_batch,y_batch in iter(self.data.test_dl):
                y_pred = self.predict1minibatch(x_batch)
                predictions.append(y_pred)        
        elif test==False:
            for *x_batch,y_batch in iter(self.data.val_dl):
                y_pred = self.predict1minibatch(x_batch)
                predictions.append(y_pred)
                
        predictions = torch.cat(predictions)
        
        if self.output_type == 'cont':
            predictions = ARR(predictions) 
        
        if self.output_type == 'cat':
            if softmax_cats == False: 
                pred_probs = ARR(predictions)
            if softmax_cats == True: 
                pred_probs = ARR( (F.log_softmax(predictions,dim=1)).exp() )
            pred_labels = pred_probs.argmax(axis=1)
            predictions = [pred_probs,pred_labels]
            
        return predictions
    
    def evaluate(self, dataset_type, metrics=[]):
        """ This function makes predictions for entire validation or training dataset 
            and computes the following quantities: 
            1. the average loss, always.
            2. the accuracy, if <output_type> == categorical and <datatype> == 'val'.
            3. any other user specified metrics, if <datatype> == 'val'.
            
         Arguments: 
         dataset_type: the type of dataset to use for evaluation, either 'train' or 'val'.
         metrics: a list where each element is a metric such that for pytorch Variables y_pred,y of length N:
                  metric(y_pred,y) = (1/N)*sum(f(y_pred[i],y[i]) for i in range(N)), 
                  for some function f. (Note: type(metric(y_pred,y)) = pytorch Variable). 
         
         """
        
        self.model.eval()
        total_loss = 0
        
        if dataset_type == 'train':
            for *x_batch, y_batch in iter(self.data.train_dl):
                y_pred = self.predict1minibatch(x_batch)
                batch_loss = self.loss_func(y_pred, VAR(y_batch))
                total_loss += (batch_loss*len(y_batch)).data[0]  
            avg_loss = total_loss/len(self.data.train_ds)
            return avg_loss
        
        if dataset_type == 'val':
            num_correct = 0
            metric_values = np.zeros(len(metrics))
            
            for *x_batch, y_batch in iter(self.data.val_dl): 
            
                y_pred = self.predict1minibatch(x_batch)
                batch_loss = self.loss_func(y_pred, VAR(y_batch))
                total_loss += (batch_loss*len(y_batch)).data[0]
                
                if self.output_type == 'cat':
                    pred_labels = y_pred.max(dim=1)[1].data
                    num_correct += (pred_labels==y_batch.cuda()).sum()
                
                for i,m in enumerate(metrics):
                    m_value = m(y_pred,VAR(y_batch)) 
                    metric_values[i] += (m_value*len(y_batch)).data[0]        
            
            avg_loss = total_loss/len(self.data.val_ds)
            accuracy = num_correct/len(self.data.val_ds)
            metric_values = metric_values/len(self.data.val_ds)
            
            if self.output_type == 'cont' and metrics == []: 
                return avg_loss
            elif self.output_type == 'cat' and metrics == []: 
                return avg_loss, accuracy
            elif self.output_type == 'cont' and metrics != []: 
                return avg_loss, metric_values
            elif self.output_type == 'cat' and metrics != []: 
                return avg_loss, accuracy, metric_values
                           
    def train1minibatch(self, x_batch, y_batch, lr_batch):
        """Function to train a model on a single minibatch of inputs. 
           Does only one update of the parameters using the optimizer (e.g. SGD or Adam). 
           Also, returns the average loss for the minibatch. """
        
        # Note: Loss functions are averaged over samples in minibatch, 
        #       and last minibatch may have less than bs elements. 
        #       So, first line below is necessary to make sure samples in last 
        #       minibatch of an epoch contribute equally to how much weights
        #       are updated in training. 
        
        
        lr_batch = lr_batch*(len(y_batch)/self.data.bs)
        self.set_lr(lr_batch)
        self.optimizer.zero_grad()
        y_pred = self.predict1minibatch(x_batch)
        loss = self.loss_func(y_pred, VAR(y_batch))
        loss.backward()
        self.optimizer.step()       
        return loss.data[0]
    
    def train_gen_lr_sched(self,lr_sched, print_output = True, save_name = None, metrics = []):
        
        """ Function to train the model with a general (user specified) learning rate schedule. 
        
        Arguments:
        lr_sched: A list of learning rates to use on succesive minibatches in training.
                  len(lr_sched) should be an integer multiple of len(self.data.train_dl), 
                  so that training is done for some number of complete epochs. 
        print_output: If <print_output> == True, prints train and val loss after every epoch. 
                      Also, prints accuracy if output_type = 'cat', and any other user specified metrics.  
        save_name: If <save_name> != None, then saves version of model with lowest value 
                   of validation loss under the name <save_name>. 
        """
        
        # check if lr_sched is of correct length, and determine number of epochs.
        if len(lr_sched) % len(self.data.train_dl) != 0:
            raise RuntimeError("len(lr_sched) must be integer multiple of len(self.data.train_dl)")    
        else: 
            num_epochs = len(lr_sched)//len(self.data.train_dl)
        
        # save initial version of model if necessary, and
        # set min_loss = val_loss (before any training). 
        if save_name != None: self.save(save_name)
        if self.output_type == 'cont': val_loss = self.evaluate('val')
        if self.output_type == 'cat': val_loss =  self.evaluate('val')[0]
        min_loss = val_loss
        
        # print out the column names for output below
        if print_output == True:
            col_names = ['train_loss', 'val_loss ']
            if self.output_type == 'cat':col_names.append('accuracy')
            if metrics != []: col_names.append('metrics') 
            print("epoch".ljust(8) + "".join(col_name.ljust(12) for col_name in col_names) + "\n")
        
        # train using the lr_sched
        i = -1 
        
        for n in range(num_epochs):
            start_time = time.time()
      
            #training pass for the epoch
            self.model.train()
            for *x_batch, y_batch in iter(self.data.train_dl):
                i = i + 1 
                self.lr_sched.append(lr_sched[i])
                loss = self.train1minibatch(x_batch, y_batch, lr_sched[i])
                self.loss_sched.append(loss)
            
            # evaluate updated model on training data
            train_loss = self.evaluate('train')
            
            # evaluate updated model on validation data 
            if self.output_type == 'cont' and metrics == []: 
                val_loss = self.evaluate('val')
            elif self.output_type == 'cat' and metrics == []: 
                val_loss, accuracy = self.evaluate('val')
            elif self.output_type == 'cont' and metrics != []: 
                val_loss, metric_values = self.evaluate('val',metrics)
            elif self.output_type == 'cat' and metrics != []: 
                val_loss, accuracy, metric_values = self.evaluate('val',metrics)
            
            end_time = time.time()
            
            # print results for the epoch, if desired.            
            if print_output == True: 
                
                mins, secs = divmod(end_time - start_time, 60)
                run_time_string = ("  epoch run time: %d min, %.2f sec" % (mins, secs))
                
                if self.output_type == 'cont' and metrics == []:
                    nums_to_print = [train_loss, val_loss]
                elif self.output_type == 'cat' and metrics == []:
                    nums_to_print = [train_loss, val_loss, accuracy] 
                elif self.output_type == 'cont' and metrics != []:
                    nums_to_print = [train_loss, val_loss] + [mv for mv in metric_values] 
                elif self.output_type == 'cat' and metrics != []:
                    nums_to_print = [train_loss, val_loss, accuracy] + [mv for mv in metric_values]
                
                col_width = 12
                for j in range(len(nums_to_print)):
                    nums_to_print[j] = '{:.5f}'.format(nums_to_print[j])
                print(str(n).ljust(8) + "".join(num.ljust(12) for num in nums_to_print) + run_time_string)
                    
            # update min_loss and save model if necessary.
            if val_loss < min_loss:
                min_loss = val_loss
                if save_name != None: self.save(save_name)
            
            # break the loop if val_loss is too high
            if val_loss > 20*min_loss: 
                print('val_loss increased too much, stopping training early')
                break
            
   
    def fit(self, lr, num_epochs = None, num_cycles = None, 
            base_cycle_length = 1, cycle_mult = 1, save_name = None, metrics = []):
        
        """
        Method to train the learner using a fixed learning rate or cosine annealing.
        
        Arguments and Training:
        num_epochs and num_cycles: Both are set to None by default, 
                                   exactly 1 should be set to a positive integer n. 
                                   
        If num_epochs == n: trains the pytorch model, self.model, for n epochs 
                            at specified constant learning rate = <lr>. 
                            
        If num_cycles == n: trains the pytorch model, self.model, for n cycles where cosine annealing 
                            is used for the learning rate in each cycle. The parameter 
                            <lr> specifies the base learning rate to use for cosine annealing. 
                            <base_cycle_length> is an integer, which is the number of epochs for first cycle. 
                            <cycle_mult> is an integer, which is a multplicative factor by which to increase
                            the length of each succesive cycle. 
                            
                            Example: <num_cycles> = 3, <base_cycle_length> = 2, <cycle mult> =4
                            1st cycle cosine annealing: length 2 epochs 
                            2nd cycle cosine annealing: length 2*4 = 8 epochs
                            3rd cycle cosine annealing: length 8*4 = 32 epochs
        
        save_name: If <save_name> != None, then saves version of model with lowest value 
                   of validation loss during training under the name <save_name>.
        """
        
        self.lr_sched = []
        self.loss_sched = []
        
        if num_epochs != None:
            lr_sched = [lr for i in range(num_epochs*len(self.data.train_dl))]
            
        if num_cycles != None:
            lr_sched = []
            cycle_length = base_cycle_length
            for i in range(num_cycles):
                if i > 0: cycle_length = cycle_length*cycle_mult
                N = len(self.data.train_dl)*cycle_length
                cycle_lr_sched = (np.cos((np.linspace(0,np.pi,N+1))[0:N]) + 1) * 0.5 * lr
                lr_sched = lr_sched + list(cycle_lr_sched)
                        
        self.train_gen_lr_sched(lr_sched, save_name = save_name, metrics=metrics)
     
    
    def find_lr(self, min_lr = 1e-5, max_lr=1.0, smoothing_radius='default'):
        
        """ Function to help find the best learning rate (between min_lr and max_lr).
            
            Learning rate is increased by a small multiplicative factor on each 
            succesive minibatch for 1 epoch, and the training loss over time is plotted.
            (Actually, a smoothed version of the training loss over time 
             is plotted with given <smoothing_radius>).  
        
        """
        
        # Define smoothing_radius by default formula if none given
        if smoothing_radius == 'default':
            smoothing_radius = max(2,int(len(self.data.train_dl)/50))
        
        # Save the current form of model to reset to at the end 
        self.save('temp')
        
        # Reset self.loss_sched and self.lr_sched 
        # Then train for 1 epoch increasing learning rate on each minibatch. 
        self.loss_sched = [] 
        self.lr_sched = []
        N = len(self.data.train_dl)
        lr_sched = list(10.0**(np.linspace(np.log10(min_lr),np.log10(max_lr),N)))
        self.train_gen_lr_sched(lr_sched,print_output = False, save_name = None)
        smoothed_loss_sched = self.smooth_timeseries(self.loss_sched,smoothing_radius)
        
        # plot the results
        figuresize=(12,6)
        fig = plt.figure(figsize=figuresize)
 
        sp = fig.add_subplot(1,2,1)
        plt.plot(self.lr_sched)
        sp.set(xlabel='minibatch', ylabel='learning rate')
        
        sp = fig.add_subplot(1,2,2) 
        plt.plot(self.lr_sched,smoothed_loss_sched)
        sp.set_xscale('log')
        sp.set(xlabel='learning rate (log scale)', ylabel='train loss')
        
        # reset the model to original state
        self.load('temp') 
                
       
