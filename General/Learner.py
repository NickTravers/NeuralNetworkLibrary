# Learner.py
from .Core import *
from .Optimizer import *
from .LossesMetrics import *

# This file Learner.py contains the class 'Learner', which is the main class in the 
# entire library. This class combines the pytorch model of a neural network together
# with the data, the optimizer, and the loss function. The class also has a large 
# number of methods for training and evaluating models, saving and loading models,
# freezing and unfreezing layers, and plotting various quantities of interest. 


# PRELIMINARIES (used in class Learner)

# Hard-Coded Variables and Dictionaries
end_metrics = {'auc':AUC}
SGD_Mom = partial(optim.SGD, momentum = 0.9)
Adam2 = partial(optim.Adam, betas=(0.9,0.99))
opt_dict = {'default':SGD_Mom, 'SGD_Mom':SGD_Mom, 'SGD':optim.SGD, 'Adam':optim.Adam, 'Adam2':Adam2} 
loss_func_dict = {'cont':nn.MSELoss(), 'cat':nn.CrossEntropyLoss(), 'single_label':nn.CrossEntropyLoss(),
                  'multi_label':nn.BCEWithLogitsLoss()} 
# Progress Bars
PBar = partial(tqdm_notebook, miniters=0)
PBarPredict = partial(tqdm_notebook, miniters=0, desc='Predicting')
PBarTrain = partial(tqdm_notebook, miniters=0, desc='Training')
PBarEvalTrain = partial(tqdm_notebook, miniters=0, desc='Eval-Train')
PBarEvalVal = partial(tqdm_notebook, miniters=0, desc='Eval-Val')
PBarTTA = partial(tqdm_notebook, miniters=0, desc='TTA')

# Plotting Confusion Matrix
# Taken directly from https://github.com/scikit-learn/scikitlearn/blob/master/examples/model_selection/plot_confusion_matrix.py
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues): 
    """ This function prints and plots the confusion matrix.
        Normalization can be applied by setting 'normalize=True'. """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()  
    

# CLASS LEARNER
class Learner(object):
    
    """ General class for a neural network model to learn from data. 
        Can be structured data, collab filter data, or image data.
    
    The class contains the pytorch model of the neural network, and also 
    the data, the optimizer, and the loss function. 
    
    Arguments for initialization:
    PATH: a path where info related to the Learner object is stored.
    data: a data object of class StructuredDataObj, CollabFilterDataObj, or ImageDataObj. 
    model: a pytorch model of class StructuredDataNet, CollabFilterNet, ImageClassificationNet, or ObjectDetectionNet.
    optimizer: an optimizer of class Optimizer. Can also specify optimizer as a string by its name in opt_dict. 
    loss_func: Loss function to use, e.g. nn.MSELoss() or nn.CrossEntropyLoss().
               If loss_func == 'default', loss function is chosen based on data.target_type using loss_func_dict. 
    use_moving_avg: If use_moving_avg == True, learner does NOT do a full evaluation on training data 
                    after each epoch of training. Instead simply returns most recent value of 
                    the attribute <moving_avg_loss> as epoch train loss. This speeds up training. 
                   
    Attributes:
    PATH, data, model, optimizer, loss_func, use_moving_avg: all same as inputs.
    target_type: Type of output variable y ('cont','cat','single_label','multi_label', or 'bbox'). 
    loss_sched: A sequence of losses obtained on succesive minibatches during training, 
                or when using learner.find_lr().
    lr_sched: The learning rate schedule, a sequence of learning rates used on succesive minibatches.
    mom_sched: The momentum schedule, a sequence of momentum values used on succesive minibatches.
    betas_sched: The sequence of values of betas used on succesive minibatches with Adam or Adam2. 
    moving_avg_loss: Exponentially weighted moving average of the training loss on successive minibatches.
    bn_frozen: Specifies which bn layers in model are frozen (either 'all','non_head', or None).
                    
    NOTE: loss_sched, lr_sched, mom_sched, and betas_sched all reset to the empty list every time 
          learner.fit(), learner.fit_cycles(), learner.fit_one_cycle() or learner.find_lr() is called. 
          moving_avg_loss also resets to 0 every time one of these functions is called. 
          However, mom_sched and betas_sched are not recorded when using learner.find_lr().         
    """  
    
    def __init__(self, PATH, data, model, optimizer='default', loss_func='default', use_moving_avg=True):
        
        PATH = correct_foldername(PATH)
        os.makedirs(PATH + 'models', exist_ok=True)
        self.PATH, self.data, self.model = PATH, data, model.cuda()
        self.loss_sched, self.lr_sched, self.mom_sched, self.betas_sched = [],[],[],[]
        self.moving_avg_loss, self.use_moving_avg = 0, use_moving_avg
        self.target_type = data.target_type 
        if loss_func == 'default': self.loss_func = loss_func_dict[self.target_type]
        else: self.loss_func = loss_func 
        if isinstance(optimizer,str): self.optimizer = Optimizer(opt_dict[optimizer],self.model)
        else: self.optimizer = optimizer
        self.bn_frozen = None
    
    # (1) METHODS FOR SAVING AND LOADING MODELS
    
    def save(self,filename,save_optimizer=False):
        """ Function to save Learner object. 
        
        Example: If you have a Learner object called <learner>, then learner.save('224') 
        saves a file called '224.pt' containing the learner model-state (and also 
        optimizer-state, if save_optimizer == True) into the folder learner.PATH/models. 
        """
        
        if save_optimizer == False:
            learner_state = {'model_state': self.model.state_dict()}
        if save_optimizer == True:
            learner_state = {'model_state': self.model.state_dict(),
                             'optimizer_state' : self.optimizer.opt.state_dict()}
            
        torch.save(learner_state, self.PATH + 'models/' + filename + '.pt') 
       
    def load(self,filename,saved_optimizer=False):
        """ Function to load the model-state (and also optimizer-state if it is saved)
            from file for a Learner object. The instance <learner> must already be  
            created and have a model and optimizer of correct type.
          
        Example: If <learner> is an object of class Learner then
        learner.load('224') loads a file called '224.pt' in folder
        learner.PATH/models, and sets learner.model.state_dict (and also
        learner.optimizer.opt.state_dict, if saved_optimizer == True) 
        to the values given in the file. 
        """
        
        filename = self.PATH + 'models/' + filename + '.pt'
        if os.path.isfile(filename):
            learner_state = torch.load(filename)
            self.model.load_state_dict(learner_state['model_state'])
            if saved_optimizer: self.optimizer.opt.load_state_dict(learner_state['optimizer_state'])
        else:
            print("no file found at '{}'".format(filename))
    
    
    # (2) METHODS FOR PLOTTING loss_sched, lr_sched, mom_sched, betas_sched
    
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
            smoothing_radius = max(5,int(len(self.data.train_dl)/50))
        smoothed_loss_sched = self.smooth_timeseries(self.loss_sched,smoothing_radius)
        plt.plot(smoothed_loss_sched)
        plt.xlabel('minibatch')
        plt.ylabel('train loss')
    
    def plot_lr_sched(self):
        plt.plot(self.lr_sched)
        plt.xlabel('minibatch')
        plt.ylabel('learning rate')
        
    def plot_mom_sched(self):
        plt.plot(self.mom_sched)
        plt.xlabel('minibatch')
        plt.ylabel('momentum')
        
    def plot_beta_sched(self):
        beta1_sched = [beta1 for (beta1,beta2) in self.betas_sched]
        plt.plot(beta1_sched)
        plt.xlabel('minibatch')
        plt.ylabel('beta_1')
        
    def plot_lr_and_loss_sched(self,smoothing_radius='default'):
        
        if smoothing_radius == 'default':
            smoothing_radius = max(5,int(len(self.data.train_dl)/50))
        smoothed_loss_sched = self.smooth_timeseries(self.loss_sched,smoothing_radius)
        
        fig = plt.figure(figsize=(12,6))        
        
        sp = fig.add_subplot(1,2,1)
        plt.plot(self.lr_sched)
        sp.set(xlabel='minibatch', ylabel='learning rate')         
        
        sp = fig.add_subplot(1,2,2) 
        plt.plot(smoothed_loss_sched)
        sp.set(xlabel='minibatch', ylabel='loss')
        
    
    # (3) METHODS FOR FREEZING AND UNFREEZING LAYER GROUPS
    
    # NOTE: All freezing/unfreezing functions automatically reset self.optimizer to match 
    #       only unfrozen layers. This clears values of lr, wd, bn_wd, clip, momentum,...etc. 
    #       from self.optimizer and resets to defaults. 
        
    def freeze(self):
        "Function to freeze all layer groups except those in the <head> submodule of self.model."
        for par in self.model.parameters(): par.requires_grad = False      
        for par in self.model.head.parameters(): par.requires_grad = True  
        self.optimizer = Optimizer(self.optimizer.opt_func,self.model)       
        
    def unfreeze(self):
        "Function to unfreeze all layer groups in self.model."
        for par in self.model.parameters(): par.requires_grad = True    
        self.optimizer = Optimizer(self.optimizer.opt_func,self.model)
        
    def bn_freeze(self,freeze_type='non_head'):
        """Freeze batchnorm layers in a model. 
           If freeze_type == 'all' freezes all bn layers.
           If freeze_type == 'non_head' freezes all bn layers, except those in head submodule of model.
           """
        
        for m in self.model.modules():
            if isinstance(m,bn_types):
                for par in m.parameters(): par.requires_grad = False
        
        if freeze_type == 'non_head':
            for m in self.model.head.modules():
                if isinstance(m,bn_types):
                    for par in m.parameters(): par.requires_grad = True
                
        self.optimizer = Optimizer(self.optimizer.opt_func,self.model)
        self.bn_frozen = freeze_type
        
    def bn_unfreeze(self):
        """Unfreeze all batchnorm layers in a model."""
        for m in self.model.modules():
            if isinstance(m,bn_types):
                for par in m.parameters(): par.requires_grad = True
        self.optimizer = Optimizer(self.optimizer.opt_func,self.model)
        self.bn_frozen = None
        
             
    # (4) METHODS FOR MAKING PREDICTIONS AND EVAULATING MODELS
      
    def predict1minibatch(self, x_batch):
        """Function to predict the output for a single minibatch of inputs x_batch. 
           Input x_batch is a cuda Tensor or list of cuda Tensors (each of length bs). 
           Output y_pred is a cuda Tensor or list of cuda Tensors (each of length bs).""" 
        
        if isinstance(x_batch,list): y_pred = self.model(*x_batch) 
        else: y_pred = self.model(x_batch)    
        return y_pred    
                   
    def predict(self, dl, correct_probs=True, thresh=0.05, max_overlap=0.5, 
                rel_thresh=None, top_k=1000, max_boxes=20, dup=None, inc=None):
        
        """ Function to make predictions for an entire dataset, specified by a dataloader.   
            
        Arguments:    
        
        dl: The pytorch DataLoader to use for making predictions on. 
            If dl = 'val' or 'test' then self.data.val_dl or self.data.test_dl are used. 
            But, may sometimes want to use other dataloaders, e.g. for TTA in image classification.
            However, for bbox object detection the dataloader must be either 'val' or 'test'.
        
        correct_probs: 
        Effects only target_type = 'cat', 'single_label', and 'multi_label'. 
        * If <self.target_type> == 'cat' or 'single_label' and correct_probs == True, then returns as 
          predictions the softmax function of output activations from network instead of activations themselves.
        * If <self.target_type> == 'multi_label' and correct_probs == True, then returns as predictions 
          the pointwise sigmoid function of activations from the network instead of activations themselves                       
        **NOTE**: These corrections are useful because normally don't apply softmax/sigmoid to final activations in 
                  the network, for compatibility with nn.CrossEntropyLoss() and nn.BCEWithLogitsLoss functions. 
          
        thresh, max_overlap, rel_thresh, top_k, max_boxes, dup, inc: 
        Effect only target_type = 'bbox'. Parameters for self.model.BBoxPredictor.
                
        Output:
        returns 1 object: predictions 
        
        Let N be the length of the dataset set that is being used for making predictions,  
        and in applicable cases also let C be the number of categories. 
        * If self.target_type == 'cont': predictions is a 1d numpy array of length N, 
          predictions[i] = predicted continuous output for ith input. 
        * If self.target_type == 'cat' or 'single_label': predictions = [pred_probs,pred_labels] where:  
          - pred_probs is an N by C numpy array with
            pred_probs[i,j] = probabability ith input is of category j
            (or the pre-softmax-ed version of this probability if correct_probs = False). 
          - pred_labels is a 1d integer numpy array of length N, pred_labels[i] is 
            the most likely category in {0,1,...,C-1} for ith input.
        * If self.target_type == 'multi_label': predictions = [pred_probs,pred_labels] where:
          - pred_probs is an N by C numpy array with
            pred_probs[i,j] = probabibility ith input contains an element of category j. 
            (or the pre-sigmoid-ed version of this probability if correct_probs = False).
          - pred_labels = is a 0-1 valued N by C numpy array such that 
            pred_labels[i,j] = 1 if and only if pred_probs[i,j] > 1/2.
        * If self.target_type == 'bbox': predictions is a list of length N with elements of 
          the form [pred_boxes, pred_classes, conf_scores] where:
          - pred_boxes is a list of predicted bounding boxes for a single image in min-max form
          - pred_classes is a list of predicted classes for each of the predicted boxes
          - conf_scores is a list of confidence scores that each predicted box is 
            of corresponding predicted class
               
        """    
        
        if self.target_type == 'bbox' and dl not in ['val','test']:
            raise ValueError("Must use 'val' or 'test' dataloader for target_type = bbox.")
        
        if dl == 'val': dl = self.data.val_dl
        elif dl == 'test': dl = self.data.test_dl
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for j,(x_batch,y_batch) in enumerate(PBarPredict(dl)):
            
                x_batch = to_cuda(x_batch)
                y_pred = self.predict1minibatch(x_batch)
            
                if self.target_type == 'cont':
                    predictions.append(ARR(y_pred))
                
                elif self.target_type in ['cat','single_label']:
                    if correct_probs == False: pred_probs = ARR(y_pred)
                    else: pred_probs = ARR((F.log_softmax(y_pred,dim=1)).exp())
                    pred_labels = pred_probs.argmax(axis=1)
                    predictions.append([pred_probs,pred_labels])
            
                elif self.target_type == 'multi_label':
                    if correct_probs == False: pred_probs = ARR(y_pred) 
                    else: pred_probs = ARR(y_pred.sigmoid())
                    true_pred_probs = ARR(y_pred.sigmoid()) 
                    pred_labels = np.around(true_pred_probs).astype(int)
                    predictions.append([pred_probs,pred_labels])
             
                elif self.target_type == 'bbox':
                    anchors, reg, clas = y_pred
                    PredBoxes, PredClasses, ConfScores = \
                    self.model.BBoxPredictor(x_batch, reg, clas, anchors, thresh, max_overlap, 
                                             rel_thresh, top_k, max_boxes, dup, inc)
                    PredBoxes, PredClasses, ConfScores = PredBoxes[0], PredClasses[0], ConfScores[0] #bs=1, for 'val' & 'test'
                    if dl==self.data.val_dl: scale = self.data.val_ds.images[j]['scale']
                    elif dl==self.data.test_dl: scale = self.data.test_ds.images[j]['scale']
                    PredBoxes = list_mult(PredBoxes,1/scale) # correct for scaling of images in training
                    predictions.append([PredBoxes,PredClasses,ConfScores])            
            
        torch.cuda.empty_cache() 
        
        # concatenate prediction lists (if necessary)
        if self.target_type == 'cont':
            predictions = np.concatenate([x for x in predictions])
            
        elif self.target_type in ['cat','single_label','multi_label']:
            predictions = [np.concatenate([predictions[i][0] for i in range(len(predictions))]),
                           np.concatenate([predictions[i][1] for i in range(len(predictions))])]
            
        return predictions
    
    def evaluate(self, dataset_type, metrics=[]):
        """ This function makes predictions for entire validation or training dataset 
            and computes the following quantities: 
            1. the average loss, always.
            2. the accuracy, if <target_type> is 'cat','single_label', or 'multi_label' and <dataset_type> is 'val'.
            3. any other user specified metrics, if <dataset_type> is 'val'.
            
         Arguments: 
         dataset_type: the type of dataset to use for evaluation, either 'train' or 'val'.
         metrics: A list of metrics to use. Each element m in metrics must be either:
                  1. A metric in <end_metrics>
                  2. A metric such that, for Tensors y_pred and y of length N,
                  m(y_pred,y) = (1/N)*sum(f(y_pred[i],y[i]) for i in range(N)), for some function f. 
         """

        # preliminaries
        EndMetrics = False
        for m in metrics:
            if m in end_metrics:
                EndMetrics = True
        
        self.model.eval()
        total_loss = 0
        
        # evaluation procedure for train dataset
        if dataset_type == 'train':
            
            with torch.no_grad():
                for x_batch, y_batch in PBarEvalTrain(self.data.train_dl):
                    bs = len(y_batch) if type(y_batch) != list else len(y_batch[0])
                    x_batch, y_batch = to_cuda(x_batch), to_cuda(y_batch)
                    y_pred = self.predict1minibatch(x_batch)
                    batch_loss = self.loss_func(y_pred,y_batch).item()
                    total_loss += bs*batch_loss        
            
            avg_loss = total_loss/len(self.data.train_ds)
            return avg_loss
        
        # evaluation procedure for val dataset
        if dataset_type == 'val':
            
            num_correct, Y, YPRED = 0, [], []
            metric_values = np.zeros(len(metrics))
            
            with torch.no_grad():
                for x_batch, y_batch in PBarEvalVal(self.data.val_dl):
                
                    bs = len(y_batch) if type(y_batch) != list else len(y_batch[0])
                    x_batch, y_batch = to_cuda(x_batch), to_cuda(y_batch)
                    y_pred = self.predict1minibatch(x_batch)
                    batch_loss = self.loss_func(y_pred,y_batch).item()
                    total_loss += bs*batch_loss

                    if EndMetrics == True:
                        YPRED.append(y_pred)
                        Y.append(y_batch)
                   
                    for i,m in enumerate(metrics):
                        if m in end_metrics: continue
                        m_value = m(y_pred,y_batch).item()
                        metric_values[i] += bs*m_value 
                
                    if self.target_type in ['cat','single_label']:
                        pred_labels = y_pred.max(dim=1)[1]
                        num_correct += (pred_labels==y_batch).sum().item()  
                    elif self.target_type == 'multi_label':
                        pred_labels = y_pred.sigmoid().round()
                        num_correct += (pred_labels==y_batch).sum().item()
            
            if EndMetrics == True:
                YPRED = torch.cat(YPRED)
                Y = torch.cat(Y)                
                for i,m in enumerate(metrics):
                    if m in end_metrics:
                        met = end_metrics[m]()
                        metric_values[i] = len(self.data.val_ds)*met(YPRED,Y).item()
            
            avg_loss = total_loss/len(self.data.val_ds)
            metric_values = metric_values/len(self.data.val_ds)
            if self.target_type in ['cat','single_label']: 
                accuracy = num_correct/len(self.data.val_ds)
            elif self.target_type == 'multi_label': 
                accuracy = num_correct/(len(self.data.val_ds)*len(self.data.categories))
            
            results = [avg_loss]
            if self.target_type in ['cat','single_label','multi_label']: results.append(accuracy)
            if metrics: results.append(metric_values)
            return results
            
    
    # (5) METHODS FOR TRAINING MODELS
    
    def train1minibatch(self, x_batch, y_batch, lr_batch, mom_batch=None, betas_batch=None):
        """Function to train a model on a single minibatch of inputs. 
           Does only one update of the parameters using the optimizer (e.g. SGD or Adam). 
           Also, returns the average loss for the minibatch. 
           Inputs x_batch and y_batch are each cuda Tensors or lists of cuda Tensors."""
        
        # Note: Loss functions are averaged over samples in minibatch, 
        #       and last minibatch may have less than self.data.bs elements. 
        #       So, the 'if bs < self.data.bs' statement below is necessary to 
        #       ensure samples in last minibatch of an epoch contribute equally 
        #       to how much weights are updated in training. 
        
        # update optimizer parameters for minibatch and zero gradient        
        bs = len(y_batch) if type(y_batch) != list else len(y_batch[0])
        if bs < self.data.bs: 
            lr_batch = list_mult(lr_batch, bs/self.data.bs)                              
        param_dict = get_param_dict(mom_batch,betas_batch)
        wd, bn_wd, clip = self.optimizer.wd, self.optimizer.bn_wd, self.optimizer.clip
        self.optimizer.set_params(lr_batch,wd,bn_wd,clip,**param_dict)
        self.optimizer.opt.zero_grad()
        
        # step the optimizer and return loss
        y_pred = self.predict1minibatch(x_batch)
        loss = self.loss_func(y_pred,y_batch)
        loss.backward()
        self.optimizer.step()       
        return loss.item()
    
    @staticmethod
    def display_training_results(col_names,values,run_times):
        """Helper function for train_gen_sched, used to print results of training in each epoch."""
        num_epochs = len(values)
        num_values = len(values[0]) if num_epochs > 0 else 0
        print("epoch".ljust(8) + "".join(col_name.ljust(12) for col_name in col_names) + '\n')
        for n in range(num_epochs):
            epoch_vals = ['{:.5f}'.format(values[n][j]) for j in range(num_values)] 
            print(str(n).ljust(8) + "".join(val.ljust(12) for val in epoch_vals) + run_times[n])
    
    def train_gen_sched(self, lr_sched, mom_sched, betas_sched, metrics=[], print_batch=False, 
                        save_name=None, save_method='best', swa_freq=None):
        
        """ Function to train the model with general user specified schedules for the 
            learning rate and also the momentum or betas values of the optimizer. 
        
        Arguments:
        lr_sched: A list of learning rates to use on succesive minibatches in training.          
        mom_sched: A list of momentum values to use on succesive minibatches in training (or None).          
        betas_sched: A list of betas values to use on succesive minibatches in training (or None). 
        metrics: List of metrics to evaluate on validation set after each epoch of training. 
        print_batch: If False, does NOT print loss and metrics of minibatches.
                     If True, prints loss and metrics of every minibatch.
                     If integer n, prints loss and metrics every n minibatches. 
                     NOTE: print_batch, must be set to False if using a metric in end_metrics. 
        save_method: None, 'best', or 'all' specifies which version(s) of model to save during training.
        save_name: * If <save_method> == 'best', saves version of model with lowest value 
                     of validation loss under the name <save_name> .
                   * If <save_method> == 'all', saves version of model after each epoch n
                     under name <save_name> + '_n'.
                   * If <save_name>  == None: then save_method automatically reset to None.
        swa_freq: Frequency, in terms of number of epochs, to use for 
                  self-weighted-averaging of model parameters (or None). 
                  See paper "Averaging Weights Leads to Wider Optima and Better Generalization"       
        """
        
        if save_name == None: save_method = None
        
        # check if given lr_sched is of correct length, and determine number of epochs.
        if len(lr_sched) % len(self.data.train_dl) != 0:
            raise ValueError("len(lr_sched) must be an integer multiple of len(learner.data.train_dl).")
        num_epochs = len(lr_sched)//len(self.data.train_dl)
        
        # clear self schedules, reset self.moving_avg_loss to 0.
        self.loss_sched, self.lr_sched, self.mom_sched, self.betas_sched = [],[],[],[]
        self.moving_avg_loss = 0
                           
        # set min_loss = val_loss (before training) and save initial version of model (if necessary). 
        min_loss =  self.evaluate('val')[0]
        clear_output() # clear progress bar 
        if save_name: self.save(save_name)
        
        # print out column names for output below
        values, run_times, col_names = [], [], ['train_loss', 'val_loss']
        if self.target_type in ['cat','single_label','multi_label']: col_names.append('accuracy')
        if metrics != []: col_names.append('metrics')
        self.display_training_results(col_names,values,run_times)
            
        # if using swa, set initial swa_model to self.model and start swa_count at 1.
        if swa_freq: 
            swa_model = copy.deepcopy(self.model)
            swa_count = 1
            
        # train using the lr_sched (and possibly mom_sched or betas_sched)       
        for n in range(num_epochs):            
            start_time = time.time()
            
            #training pass for the epoch
            self.model.train() 
            
            if self.bn_frozen in ['all','non_head']:
                for m in self.model.modules():
                    if isinstance(m,bn_types): m.training = False
            if self.bn_frozen == 'non_head':
                for m in self.model.head.modules():
                    if isinstance(m,bn_types): m.training = True
                        
            for j, (x_batch, y_batch) in enumerate(PBarTrain(self.data.train_dl)):
                
                start_time_batch = time.time()
                x_batch, y_batch = to_cuda(x_batch), to_cuda(y_batch)              
                i = n*len(self.data.train_dl) + j
                self.lr_sched.append(lr_sched[i])     
                if mom_sched:
                    self.mom_sched.append(mom_sched[i])
                    loss = self.train1minibatch(x_batch, y_batch, lr_sched[i], mom_batch=mom_sched[i])
                elif betas_sched:
                    self.betas_sched.append(betas_sched[i])
                    loss = self.train1minibatch(x_batch, y_batch, lr_sched[i], betas_batch=betas_sched[i])
                else: loss = self.train1minibatch(x_batch, y_batch, lr_sched[i])
                self.loss_sched.append(loss)
                self.moving_avg_loss = self.moving_avg_loss*0.98 + loss*0.02
                debiased_moving_avg_loss = self.moving_avg_loss/(1 - 0.98**(i+1))
                end_time_batch = time.time()
                
                #print out the results for the batch if necessary
                if (print_batch==True) or (type(print_batch)==int and (j%print_batch)==0):
                    
                    if j == 0:
                        colnames = ['avg_loss','batch_loss']
                        if len(metrics) > 0: colnames += ['batch_metrics']
                        print("batch".ljust(8) + "".join(colname.ljust(12) for colname in colnames))
                    
                    batch_vals = [debiased_moving_avg_loss, loss]
                    if len(metrics) > 0:
                        with torch.no_grad():
                            y_pred = self.predict1minibatch(x_batch)  
                            for m in metrics: batch_vals.append(m(y_pred,y_batch).item())
                    batch_vals = ['{:.5f}'.format(batch_vals[l]) for l in range(2+len(metrics))]
                    s = str(j).ljust(8) + "".join(val.ljust(12) for val in batch_vals)
                    print(s + ("batch run time: %.2f" % (end_time_batch - start_time_batch)))
                    
            # evaluate updated model on training data, or use debiased moving_avg_loss as train_loss
            if self.use_moving_avg == True:
                train_loss = debiased_moving_avg_loss 
            else: train_loss = self.evaluate('train')
                  
            # evaluate updated model on validation data
            if self.target_type in ['cat','single_label','multi_label'] and metrics == []: 
                val_loss, accuracy = self.evaluate('val')
                values.append([train_loss, val_loss, accuracy])
            elif metrics == []: 
                val_loss = self.evaluate('val')[0]
                values.append([train_loss, val_loss])
            elif self.target_type in ['cat','single_label','multi_label'] and metrics != []: 
                val_loss, accuracy, metric_values = self.evaluate('val',metrics)
                values.append([train_loss, val_loss, accuracy] + [mv for mv in metric_values])
            elif metrics != []: 
                val_loss, metric_values = self.evaluate('val',metrics)
                values.append([train_loss, val_loss] + [mv for mv in metric_values])
            
            end_time = time.time()
            mins, secs = divmod(end_time - start_time, 60)
            run_times.append("  epoch run time: %d min, %.2f sec" % (mins, secs))
            
            # print results for the epoch 
            # (must clear display first, because of problem removing progress bars)           
            clear_output() 
            self.display_training_results(col_names,values,run_times)
                    
            # update min_loss and save model if necessary.
            if val_loss < min_loss:
                min_loss = val_loss
                if save_method == 'best': self.save(save_name)
            if save_method == 'all':
                self.save(save_name + '_' + str(n))
                
            # do swa updates if necessary
            if swa_freq and (n+1)%swa_freq == 0:
                weights=[swa_count/(swa_count+1), 1/(swa_count+1)]
                swa_model = combine_models([swa_model,self.model],weights)
                swa_count += 1
      
            # break the loop if val_loss is too high
            if val_loss > 20*min_loss:
                print('val_loss increased too much, stopping training early')
                break
                
        # set final model to the swa model, if using swa
        if swa_freq: self.model = swa_model
        
    def init_optimizer(self,wd=None,bn_wd=None,clip=None):
        """Initialize optimizer with specified values of wd, bn_wd, clip. 
           These are not modified during a training period (i.e. 1 call 
           to fit, fit_cycles, or fit_one_cycle). If any values are 
           not specified, then last ones used by the optimizer are taken."""
        WD = wd if wd else self.optimizer.wd
        BN_WD = bn_wd if (bn_wd is not None) else self.optimizer.bn_wd
        CLIP = clip if clip else self.optimizer.clip
        self.optimizer.set_params(lr=0,wd=WD,bn_wd=BN_WD,clip=CLIP)
    
    @staticmethod 
    def get_sched(sched_type,N,start_val,end_val):
        
        """Returns a list of N points according to a given sched_type.
        
          For purposes of description, let us call the output y = y[0],...,y[N-1].
        * If sched_type == 'linear': 
          Points y[i] are linearly spaced between start_val and end_val.
        * If sched_type == 'cos': 
          Points y[i] follow the curve y = end_val + (start_val - end_val)*(1/2)*(cos(x)+1), 
          for x[i] linearly spaced in [0,pi]. 
        * If sched_type == 'exp':
          Points y[i] follow the curve y = e^x, for x[i] linearly spaced in [a,b], 
          where a,b are chosen such y[0] = start_val, y[N-1] = end_val.
        * If sched_type == 'poly':
          Points y[i] follow the curve y = start_val*x^p, for x[i] = i+1 (i=0,...,N-1),
          where p is chosen such that y[0] = start_val, y[N-1] = end_val. 
          
        NOTE: start_val and end_val may both be arrays of the same length L 
              (or lists, which are converted to arrays) instead of single numbers. 
              In this case, the list of points which is returned is also list of 
              length L arrays, where x[i],y[i],x,y in formulas above are thought of as 
              length L vectors.   
        """
        
        if type(start_val) == list: start_val = np.array(start_val)
        if type(end_val) == list: end_value = np.array(end_val)
        
        if sched_type == 'linear':
            return list(linear_space(start_val,end_val,N))
        elif sched_type == 'cos':
            s = 0.5*(np.cos(np.linspace(0,np.pi,N)) + 1)
            return list(end_val + outer_mult(start_val-end_val,s))        
        elif sched_type == 'exp':
            a,b = np.log(start_val),np.log(end_val)
            return list(np.exp(linear_space(a,b,N)))
        elif sched_type == 'poly':
            p = np.log(end_val/start_val)/np.log(N)
            return [start_val*i**p for i in range(1,N+1)] 
        
    def fit(self, lr, num_epochs, wd=None, bn_wd=None, clip=None, momentum=None, betas=None, 
            metrics=[], print_batch=False, save_name=None, save_method='best', swa_freq=None):       
        """Method to train learner with constant lr, and constant momentum or 
           beta values, for given number of epochs. """ 
        
        if (type(lr) == list) and (len(lr) != len(self.model.layer_groups)):
            raise ValueError("If <lr> is a list, must have len(lr) = len(learner.model.layer_groups).")
            
        self.init_optimizer(wd,bn_wd,clip)
        N = num_epochs*len(self.data.train_dl)
        lr_sched = [lr]*N
        mom_sched = [momentum]*N if momentum else None
        betas_sched = [betas]*N if betas else None 
        self.train_gen_sched(lr_sched, mom_sched, betas_sched, metrics, 
                             print_batch, save_name, save_method, swa_freq)
          
    def fit_cycles(self, lr_start, lr_end, num_cycles, cycle_type='cos', base_length=1, cycle_mult=1, 
                   wd=None, bn_wd=None, clip=None, momentum=None, betas=None, metrics=[], 
                   print_batch=False, save_name=None, save_method='best', swa_freq=None): 
        """Method to train learner using a given type of lr annealing schedule,
           with restarts, for a given number of cycles. Momentum and beta values 
           are held constant throughout training. """
        
        
        if (type(lr_start) == list) and (len(lr_start) != len(self.model.layer_groups)):
            raise ValueError("If <lr_start> is a list, must have len(lr_start) = len(learner.model.layer_groups).")
        if (type(lr_end) == list) and (len(lr_end) != len(self.model.layer_groups)):
            raise ValueError("If <lr_end> is a list, must have len(lr_end) = len(learner.model.layer_groups).")
        
        self.init_optimizer(wd,bn_wd,clip) 
        
        lr_sched = []
        mom_sched = [] if momentum else None
        betas_sched = [] if betas else None
        
        cycle_length = base_length
        for i in range(num_cycles):
            if i > 0: cycle_length = cycle_length*cycle_mult
            N = len(self.data.train_dl)*cycle_length
            lr_sched += self.get_sched(cycle_type,N,lr_start,lr_end)
            if momentum: mom_sched += [momentum]*N
            if betas: betas_sched += [betas]*N
                
        self.train_gen_sched(lr_sched, mom_sched, betas_sched, metrics, 
                             print_batch, save_name, save_method, swa_freq)
        
    def fit_one_cycle(self, lr_max, num_epochs, div_fac=25, start_pct=0.3, wd=None, bn_wd=None, 
                      clip=None, mom_min=0.85, mom_max=0.95, beta_min=0.85, beta_max=0.95, 
                      metrics=[], print_batch=False, save_name=None, save_method='best'):
        """Method to train learner using the one-cycle method for a given number of epochs. """
        
        if (type(lr_max) == list) and (len(lr_max) != len(self.model.layer_groups)):
            raise ValueError("If <lr_max> is a list, must have len(lr_max) = len(learner.model.layer_groups).")
        
        if type(lr_max) == list: lr_max = np.array(lr_max)
        self.init_optimizer(wd,bn_wd,clip)
        
        # define lr_sched
        N = num_epochs*len(self.data.train_dl) 
        N1,N2 = int(N*start_pct), N - int(N*start_pct)
        lr_min = lr_max/div_fac
        lr_sched = self.get_sched('linear',N1,lr_min,lr_max) + self.get_sched('cos',N2,lr_max,lr_min/1e4)
        
        # define mom_sched or betas_sched (if applicable)
        mom_sched, betas_sched = None, None
        if 'momentum' in self.optimizer.opt.param_groups[0]:
            mom_sched = self.get_sched('linear',N1,mom_max,mom_min) + self.get_sched('cos',N2,mom_min,mom_max)
        if 'betas' in self.optimizer.opt.param_groups[0]:
            betas_sched = self.get_sched('linear',N1,beta_max,beta_min) + self.get_sched('cos',N2,beta_min,beta_max)
            betas_sched = [(float(beta_val),0.99) for beta_val in betas_sched]
        
        # train with given schedules
        self.train_gen_sched(lr_sched, mom_sched, betas_sched, metrics, print_batch, save_name, save_method)
    
    def find_lr(self, lr_min=1e-5, lr_max=1.0, wd=None, bn_wd=None, clip=None, momentum=None,
                betas=None, length='1epoch', break_fac = 3, sched_type='exp', smoothing_radius='default',
                plot_start_batch=0):
        
        """ Function to help find the best learning rate between lr_min and lr_max.
        Increases lr rate on each minibatch, either linearly or exponetntially,
        then plots loss_sched and lr_sched.  
            
        Arguments:
        lr_min,lr_max: min and max learning rates.
        sched_type: 'linear' or 'exp' defines how lr is increased on succesive minibatches.
        length: '1epoch' or an integer for number minibatches of training to do.
        smoothing_radius: smoothing radius to use when plotting loss_sched.                     
        wd,bn_wd,clip,betas,momentum: parameters for optimizer, constant throughout all minibatches.
        break_fac: breaks training early if debiased moving_avg_loss increases by a factor of break_fac 
                   from its initial value. If break_fac == None, never breaks training early.
        plot_start_batch: Begins plots with batch number plot_start_batch on x-axis.
                          Default is 0, but if loss starts out large and very quickly gets quite small, 
                          it may be useful to start plots later for better visualization of the later changes.            
        """
        
        if (type(lr_max) == list) and (len(lr_max) != len(self.model.layer_groups)):
            raise ValueError("If <lr_max> is a list, must have len(lr_max) = len(learner.model.layer_groups).")
        if (type(lr_min) == list) and (len(lr_min) != len(self.model.layer_groups)):
            raise ValueError("If <lr_min> is a list, must have len(lr_min) = len(learner.model.layer_groups).")    
        
        # Save current form of model to reset to at the end 
        self.save('temp',save_optimizer=True)
        
        # Preliminaries for training
        self.moving_avg_loss = 0
        self.loss_sched, self.lr_sched, self.mom_sched, self.betas_sched = [],[],[],[]
        self.init_optimizer(wd,bn_wd,clip)
        
        self.model.train()
        if self.bn_frozen in ['all','non_head']:
            for m in self.model.modules():
                if isinstance(m,bn_types): m.training = False
        if self.bn_frozen == 'non_head':
            for m in self.model.head.modules():
                if isinstance(m,bn_types): m.training = True    
        
        # Train for number of minibatches specified by <length>.
        # At each iteration increase lr either exponentially or linearly,
        # as specified by sched_type.
        N = len(self.data.train_dl) if length == '1epoch' else length         
        num_epochs = int(np.ceil(N/len(self.data.train_dl))) 
        lr_sched = self.get_sched(sched_type,N,lr_min,lr_max)        
        for n in range(num_epochs):
            for j, (x_batch, y_batch) in enumerate(PBarTrain(self.data.train_dl)):
                i = n*len(self.data.train_dl) + j
                x_batch, y_batch = to_cuda(x_batch), to_cuda(y_batch)
                if momentum: loss = self.train1minibatch(x_batch, y_batch, lr_sched[i], mom_batch=momentum)
                elif betas: loss = self.train1minibatch(x_batch, y_batch, lr_sched[i], betas_batch=betas)
                else: loss = self.train1minibatch(x_batch, y_batch, lr_sched[i])
                self.loss_sched.append(loss)
                self.lr_sched.append(lr_sched[i])
                self.moving_avg_loss = self.moving_avg_loss*0.98 + loss*0.02
                debiased_moving_avg = self.moving_avg_loss/(1 - 0.98**(i+1))
                if i == 0: initial_loss = debiased_moving_avg
                if break_fac and debiased_moving_avg > break_fac*initial_loss: break
                if i == N-1: break
        
        # Plot the results
        fig = plt.figure(figsize=(12,6))
        if smoothing_radius == 'default': smoothing_radius = int(max(5,N/50))
        smoothed_loss_sched = self.smooth_timeseries(self.loss_sched,smoothing_radius)
 
        sp = fig.add_subplot(1,2,1)
        plt.plot(range(plot_start_batch,len(self.lr_sched)),self.lr_sched[plot_start_batch:])
        sp.set(xlabel='minibatch', ylabel='learning rate')
        
        sp = fig.add_subplot(1,2,2) 
        plt.plot(self.lr_sched[plot_start_batch:],smoothed_loss_sched[plot_start_batch:])
        if sched_type == 'linear':
            sp.set(xlabel='learning rate', ylabel='train loss')
        elif sched_type == 'exp':
            sp.set_xscale('log')
            sp.set(xlabel='learning rate (log scale)', ylabel='train loss')
            
        # Reset model to original state
        self.load('temp',saved_optimizer=True)
        