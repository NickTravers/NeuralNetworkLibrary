# Layers.py
from .Core import *

# This file Layers.py contains a collection of pytorch layers (i.e. modules),
# that are used as building blocks in creating models. Most layers are quite
# simple, but there is also 1 class for a general fully connected linear
# network. Such fully connected nets are used as the "head" component of 
# larger networks in image classification and structured data applications. 

# LIST OF FUNCTIONS AND CLASSES:
# 1. class Flatten
# 2. class Flatten1d
# 3. class Linear
# 4. class Conv2d
# 5. def get_embbedding
# 6. class EmbeddingDrop
# 7. class AdaptiveConcatPool2d
# 8. class FullyConnectedNet   

class Flatten(nn.Module):
    "Flattens an input batch x of shape (bs x n1 x n2 ... x nk) to shape (bs x n), for appropriate n."
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Flatten1d(nn.Module):
    "Flattens an input batch x of shape (bs x 1) to a 1d Tensor of length bs."
    def forward(self,x):
        return x.view(-1)

class Linear(nn.Module):
    "Applies nn.Linear, preceeded (optionally) by dropout, and followed by relu and then (optionally) bn."
    def __init__(self, nin, nout, bn=True, drop=0):
        super().__init__()
        self.lin = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1d(nout) if bn else None
        self.drop = nn.Dropout(drop) if drop else None
    def forward(self,x):
        if self.drop: x = self.drop(x)
        x = F.relu(self.lin(x))
        if self.bn: x = self.bn(x)
        return x

class Conv2d(nn.Module):
    "Applies nn.Conv2d, preceeded (optionally) by dropout, and followed by relu and then (optionally) bn."
    def __init__(self, nin, nout, ks=3, stride=1, pad=1, bn=True, drop=0):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, ks, stride, pad)
        self.bn = nn.BatchNorm2d(nout) if bn else None
        self.drop = nn.Dropout(drop) if drop else None     
    def forward(self,x):
        if self.drop: x = self.drop(x)
        x = F.relu(self.conv(x))
        if self.bn: x = self.bn(x)
        return x 
    
def get_embedding(num_cats,emb_dim,std=0.01,max_norm=None):
    """Returns an embedding layer, pre-initialized so that weight matrix is
       independent truncated normal random variables with mean 0 and given std. """ 
    emb = nn.Embedding(num_cats,emb_dim,max_norm=max_norm)
    with torch.no_grad(): (emb.weight).normal_().fmod_(2).mul_(std)
    return emb

class EmbeddingDrop(nn.Module):
    """Effectively applies a dropout layer followed by an embedding layer. 
       Embedding is pre-initialized, so that weight matrix is independent 
       truncated normal random variables with mean 0 and given std. """
          
    def __init__(self,num_cats,emb_dim,drop,std,max_norm):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.emb = nn.Embedding(num_cats,emb_dim,max_norm=max_norm)
        with torch.no_grad(): (self.emb.weight).normal_().fmod_(2).mul_(std)
    
    def forward(self,x):
        ones = torch.ones(len(x)).cuda()
        return self.emb(x) * self.drop(ones).unsqueeze(1)   
       
class AdaptiveConcatPool2d(nn.Module):
    """Class for an adapitve concat pool. 'Adaptive' means adaptive to size of input,
       "concat" means concatenates results of a max pool and an average pool."""
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)    

class FullyConnectedNet(nn.Module):
    
    """ Class for a multi-layer fully connected neural network. 
    
        Relu activation is applied after each fully connected linear layer,
        except the last one. Batch normalization and dropout may also be used. 
 
    Arguments:
    layer_sizes: A list specifying the sizes of the layers. 
                 For example, if layer sizes = [100,50,20,10] then:
                 100 input activations to the network
                 fc_layer1 has 100 inputs, 50 outputs
                 fc_layer2 has 50 inputs, 20 outpus
                 fc_layer3 has 20 inputs, 10 outputs
                 10 output activations from the network
                 
    drops: The amount of dropout to use before each fully connected layer.
           Continuing example above, if drops = [0.5,0.3,0.1] then
           0.5 dropout before fc_layer1, 0.3 dropout before fc_layer2, and 
           0.1 dropout before fc_layer3.
    
    final_activ: Activation function to apply to the output of final fully connected 
                 linear layer, before returning output of network itself. Choices are:
                 None, 'softmax', 'sigmoidal'. If 'sigmoidal' the parameter <output_range>
                 must also be specified.
                          
    output_range: If final_activation_func == 'sigmoidal' and output_range = [a,b] then 
                  outputs of final linear layer are compressed to the range [a,b] using 
                  an appropriately scaled and shifted sigmoid function. 
                  If final_activation_func != 'sigmoidal', leave as the default, output_range = None. 
                          
   bn: If True, bn is applied after the Relu activation following each non-final linear layer.
   
   pre_bn: If True, bn is applied to the input to network before passing through 1st linear layer.
   """
    
    def __init__(self, layer_sizes, drops=None, final_activ=None, output_range=None, bn=True, pre_bn=True):
        
        super().__init__()
        N = len(layer_sizes) - 1
        if drops is None: drops = [0]*N         
        self.final_activ = final_activ
        self.output_range = output_range
        
        self.pre_bn = nn.BatchNorm1d(layer_sizes[0]) if pre_bn else None
        self.lins = nn.ModuleList([ Linear(layer_sizes[i],layer_sizes[i+1],bn,drops[i]) for i in range(N-1) ])
        self.final_drop = nn.Dropout(drops[N-1])
        self.final_lin = nn.Linear(layer_sizes[N-1],layer_sizes[N])
        initialize_modules([self.lins,self.final_lin],nn.init.kaiming_normal_,False)

    def forward(self,x):
        
        if self.pre_bn: 
            x = self.pre_bn(x)
        for lin in self.lins: 
            x = lin(x)                
        x = self.final_drop(x)
        x = self.final_lin(x)        
        
        if self.final_activ == 'softmax': 
            x = (F.log_softmax(x,dim=1)).exp()
        elif self.final_activ == 'sigmoidal':
            MIN,MAX = float(self.output_range[0]), float(self.output_range[1])
            x = MIN + (MAX - MIN)*x.sigmoid()
            
        return x
    