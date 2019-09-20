# Optimizer.py
from .Core import *

# This file Optimizer.py contains the 'Optimizer' class, which is a wrapper
# around the pytorch optim.Optimizer class. It incorporates gradient 
# clipping and a corrected form of weight decay for use with Adam. It also 
# has support for using differential learning rates for different layer groups.

def get_param_dict(momentum=None,betas=None):
    "Simple function to construct a dictionary of parameters for an optimizer."
    param_dict = {}
    if momentum:param_dict['momentum'] = momentum
    if betas:param_dict['betas'] = betas
    return param_dict

class Optimizer(object):
    
    """Wrapper around a pytorch optim.Optimizer. Incorporates gradient clipping and correct 
    form of weight decay as in paper "Fixing Weight Decay Regularization in Adam". 
    
    Arguments for Initialization:
    opt_func: pytorch optimizer class e.g. (optim.SGD, optim.Adam, Adam2)
    model: model to be trained (of class ImageClassificationNet, ObjectDetectionNet, StructuredDataNet, or CollabFilterNet)
    wd: list of length NL=number of layer_groups in model, to use for L2 weight decay constants in each layer group, 
        or a single number to use for all layer groups, or None.
    bn_wd: If bn_wd = True, weight decay is used in batch norm layers. If bn_wd = False, is not used.
    clip: Value for clipping norm of gradient of model.parameters(), or None.  
    
    Attributes:
    model, opt_func, wd, bn_wd, clip: same as inputs
    NL: number of layer groups in model
    lr: list of length NL to use for learning rates in each layer group. 
    opt: the pytorch optimizer of class optim.Optimizer
    """
    
    def __init__(self,opt_func,model,wd=None,bn_wd=True,clip=None): 
        self.model, self.opt_func, self.NL = model, opt_func, len(model.layer_groups)
        self.lr, self.wd, self.bn_wd, self.clip = [0]*self.NL, wd, bn_wd, clip
        self.opt = opt_func([{'params':trainable_params(pg), 'lr':0} for pg in model.param_groups])
         
    def set_params(self,lr,wd=None,bn_wd=True,clip=None,**kwargs):
        """Set parameters of optimizer (e.g. lr, wd, bn_wd, clip, momentum). Each param in 
        <kwargs>, as well as <lr> and <wd>, may each be a single value (to use for all layer
        groups) or a list of values or np.array with length equal to NL.""" 
        lr = LIST(lr,self.NL)
        if wd: wd = LIST(wd,self.NL)
        for par in kwargs: kwargs[par] = LIST(kwargs[par],self.NL,Tuple=False)
        self.lr, self.wd, self.bn_wd, self.clip = lr, wd, bn_wd, clip
        kwargs['lr'] = lr
        for par in kwargs:            
            for i,pg in enumerate(self.opt.param_groups): 
                pg[par] = kwargs[par][i%self.NL]
    
    def grad_clip(self):
        """Clip gradient of all model parameters combined (i.e. not by layers). """
        if self.clip: torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clip)
             
    def step(self):
        """Implements one step of optimizer including gradient clippling and weight decay."""
        if self.wd:
            reg_groups = self.opt.param_groups[:self.NL] 
            bn_groups = self.opt.param_groups[self.NL:]
            with torch.no_grad():
                for lr,wd,pg1,pg2 in zip(self.lr,self.wd,reg_groups,bn_groups):
                    for X in pg1['params']: X.mul_(1-wd*lr)
                    if self.bn_wd: 
                        for X in pg2['params']: X.mul_(1-wd*lr)
        
        self.grad_clip()
        self.opt.step()
        
    def print_summary(self,print_param_groups=True):
        """Function to print out summary info about an optimizer. """ 
        print('optimizer.model = ',self.model)
        print('optimizer.opt = ',self.opt)
        if print_param_groups == True:
            for pg in self.opt.param_groups: 
                print(pg)
                print('')
        print('optimizer.NL = ', self.NL)
        print('optimizer.lr = ', self.lr)
        print('optimizer.wd = ', self.wd)
        print('optimizer.clip = ', self.clip)
        print('optimizer.bn_wd = ', self.bn_wd) 
        
    def print_params_grads(self):
        """Function to print out all parameters in a model and their gradients, 
        by param_group. Useful for diagnosing problems with optimization."""
        for j,pg in enumerate(self.model.param_groups):
            print('PG',j,'=',pg)
            print('')
            for i,X in enumerate(pg.parameters()): 
                print('parameter',i,'=',X)
                print('parameter',i,'grad =', X.grad)
                if X.grad is None: print('')
            print('')
