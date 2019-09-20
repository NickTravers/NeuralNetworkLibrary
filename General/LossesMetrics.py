# LossesMetrics.py
from .Core import *

# This file LossesMetrics.py contains a collection of loss functions and other metrics, 
# which are not part of the standard built-in pytorch loss functions. For consistency with 
# the built-in pytorch loss functions, all loss functions and metrics in this file are written 
# as python classes, rather than python functions, even if they do not need any arguments for 
# initilization. Inputs and outputs of all these loss functions and metrics are torch Tensors. 

# LIST OF LOSS FUNCTIONS AND METRICS:
# 1. MSPE_loss
# 2. logMSE_loss
# 3. expMSPE_loss
# 4. fbeta_loss
# 5. kPrecision
# 6. AUC

class MSPE_loss(object):
    """Class for the mean-square-percentage-error loss defined by:
       MSPE = (1/N) * sum_{i=1}^N ((yhat_i - y_i)/y_i)^2 """    
    
    def __call__(self,preds,target):
        return pow((preds-target)/target, 2).mean()

class logMSE_loss(object):
    """ Class for the log-Mean-Square-Error loss defined by: 
        log_MSE = (1/N) * sum_{i=1}^N (log(yhat_i) - log(y_i))^2        
        
        NOTE: This is a more numerically stable approximation of the MSPE_loss. """
         
    def __call__(self,preds,target):
        return pow(torch.log(preds) - torch.log(target), 2).mean()
    
class expMSPE_loss(object):    
    """ Class for the exponential-Mean-Square-Percentage-Error loss. 
        This is the MSPE_loss of the exponential of the predictions:
        exp_MSPE = (1/N) * sum_{i=1}^N ((zhat_i - z_i)/z_i)^2, 
        where z_i = e^y_i, zhat_i = e^yhat_i. """
     
    def __call__(self,preds,target):
        exp_pred, exp_targ = torch.exp(preds), torch.exp(target)
        return pow((exp_pred-exp_targ)/exp_targ, 2).mean()
    
class fbeta_loss(object):
    
    """ Class for the fbeta loss function for multi-label classification. 
    See https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/overview/evaluation ,
    for loss definition. My code modified from https://www.kaggle.com/igormq/f-beta-score-for-pytorch       
    
    Arguments:
    beta: The parameter beta in definition of f_beta loss. 
    eps: Small number for numerical stability, default 1e-9 is always fine.
    use_thresh, threshold: If use_thresh == True, we say model predicts that input i contains category j, 
                           if and only if sigmoid(y_pred[i][j]) > threshold. """
    
    def __init__(self,beta,threshold=0.5,use_thresh=True,eps=1e-9):
        self.beta, self.eps = beta, eps
        self.threshold, self.use_thresh = threshold, use_thresh
    
    def __call__(self,y_pred,y_true):       
        """ 
        Arguments: (N is bs, C is number of categories)
        y_pred: N by C torch Tensor.
                If use_thresh == True: sigmoid(y_pred[i][j]) = Prob(input i contains category j)
                If use_thresh == False: y_pred[i][j] = 1 or 0, binary prediction that input i contains category j.
        y_true: N by C 0-1 valued torch Tensor, with y_true[i][j] = 1 if and only if 
                input i contains category j. 
        """
        
        beta2 = self.beta**2
        if self.use_thresh==True: 
            y_pred = torch.ge(y_pred.sigmoid(),self.threshold).float()
        else: y_pred = y_pred.float()
        y_true = y_true.float()
        tp = (y_pred * y_true).sum(dim=1)           # tp = true positives
        p = tp.div(y_pred.sum(dim=1).add(self.eps)) # p = precision 
        r = tp.div(y_true.sum(dim=1).add(self.eps)) # r = recall 
        return torch.mean( (1+beta2)*(p*r)/(beta2*p + r + self.eps) )
        
class kPrecision(object):
    """precision@k metric, for single-label classification."""
    
    def __init__(self,k):
        self.k = k
       
    def __call__(self,preds,target,weights=None):
        """
        Arguments: (N is either bs or size of dataset)
        preds: torch Tensor of dimensions (N x num_classes), preds[i,j] = prob(element i of class j).
        target: torch LongTensor of length N, ith entry is integer class label of ith element.
        weights: list of weights to apply to predictions (or None).
        """
        N = len(preds)
        if weights == None: weights = [1]*N
        batch_precision = 0
        
        for i in range(N):
            sorted_values, sorted_labels = preds[i].sort(descending=True)
            sorted_labels = sorted_labels[:self.k]
            true_label = target[i]
            
            precision = 0
            for j in range(self.k): 
                if sorted_labels[j] == true_label: precision = 1/(j+1)   
            batch_precision += precision*weights[i]
            
        return TEN(batch_precision/sum(weights))
    

class AUC(object):
    """Class AUC is "Area under the Curve" for Receiver Operating Characteric (ROC) curve. 
       Used for binary classification where 1 represents positive class, 0 negative class. """
    
    def __call__(self,preds,target):
        """ 
        Arguments: (N is the size of dataset)
        preds: torch Tensor of dimensions (N x 2).
               softmax(pred[i]) = [prob(element i of neg class), prob(element i of pos class)]
        target: torch LongTensor of length N, ith entry is 1 if ith element is of pos class, and 0 otherwise. 
        """
            
        preds = torch.exp(F.log_softmax(preds,dim=1))
        y_pred, y_true = ARR(preds[:,1]), ARR(target)
        return TEN(skm.roc_auc_score(y_true,y_pred))

        
            