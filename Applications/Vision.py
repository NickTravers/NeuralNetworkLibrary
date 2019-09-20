# Vision.py
from General.Core import *
from General.Layers import *
from General.Learner import *
from General.LossesMetrics import *
from General.Optimizer import *

import torchvision.models as models
from .VisionModels import vmods
from .pycocotools.coco import COCO
from .pycocotools.cocoeval import COCOeval

# This file Vision.py contains a collection of functions and classes 
# for use in various problems related to computer vision. Specifically, 
# image classification, multi-label image classification, and bounding box
# object detection. 

# It is also intended to possibly add in functionality for image segmentation at a later 
# time, and thus some of the functions (and their comments) allow this as a possibility. 
# However, full functionality for image segmentation is not yet incorporated. 

# OUTLINE:
# Section (1) - Utility Functions 
# Section (2) - Image Display
# Section (3) - Image Transforms
# Section (4) - Datasets and DatabObj
# Section (5) - ImageClassificationNet and ImageClassificationEnsembleNet
# Section (6) - ObjectDectionNet and Object Dectection Losses + Metrics
# Section (7) - ImageLearner

# NOTE: 
# Following notation and conventions are used in docstrings 
# and functions throughout this file:

# 1. Images are numpy arrays of floats in range [0,1], shape is (height,width,num_channels).
# 2. Unless otherwise specified num_channels=3, RGB.
# 3. A "min-max bounding box" is box of form = np.array([xmin,ymin,xmax,ymax]) or corresponding torch Tensor.
# 4. A "height-width bounding box" is box of form = np.array([xmin,ymin,width,height]) or corresponding torch Tensor.
# 5. Unless otherwise specified, bounding boxes are assumed to be in min-max form.
# 6. The variable called <bboxes> normally has form [(b_1,c_1),...,(b_n,c_n)] where:
#    each b_i is a min-max bounding box, each c_i is an integer category label.
# 7. The variable called <categories> always is a dictionary of form:
#    {0:'cat',1:'chair,2:'tree'} mapping integer category labels to category names.

# Hard Coded Variables
imagenet_stats = [np.array([0.485, 0.456, 0.406]),np.array([0.229, 0.224, 0.225])] # for all other pretrained models
alternate_stats = [np.array([0.5,0.5,0.5]),np.array([0.5,0.5,0.5])]  #for inceptionv4, inceptionresnetv2, nasnetalarge
Pascal_thresholds = [0.5]
COCO_thresholds = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

# SECTION (1) - UTILITY FUNCTIONS

# (1.1) General Utilities
def open_image(img_name):
    
    """Opens an image with openCV, <img_name> is full path of the image file.
     Returns image in RGB format as np.array of floats in range [0,1], shape is (height,width,3)."""
    
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYCOLOR
    img = cv2.imread(img_name,flags).astype(np.float32)/255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_resized(img_folder,new_folder,sz):
    
    """Function to save resized versions of images in one folder to another folder. 
       Useful if original images are very large, makes training on smaller images faster. 
    
    Arguments:
    img_folder: full path of folder where images are stored (no other files should be in this folder)
    new_folder: new folder where resized copies should be stored (should be distinct from img_folder)
    sz: * If sz is an integer, images are resized to have min(height,width) = sz, 
          with same aspect ratio as original image.
        * If sz is a tuple, sz = (h,w), images are resized to (height,width) = (h,w).
    """        
    
    img_folder = correct_foldername(img_folder)
    new_folder = correct_foldername(new_folder)
    os.makedirs(new_folder, exist_ok=True)
    filenames = os.listdir(img_folder)
    
    for fn in PBar(filenames):
        if fn[:2] == '._': continue
        img = cv2.imread(img_folder + fn)
        if type(sz) == int:
            rows,cols = img.shape[0], img.shape[1]
            if rows <= cols: img = cv2.resize(img, (int(cols*sz/rows),sz), interpolation=cv2.INTER_LINEAR)
            if rows > cols: img = cv2.resize(img, (sz,int(rows*sz/cols)), interpolation=cv2.INTER_LINEAR) 
        else: 
            img = cv2.resize(img, (sz[1],sz[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(new_folder + fn,img) 

def get_stats(img_folder,img_names=None):
    
    """Function to compute the basic stats (means and stds) over images in a given folder. 
    
    Arguments:
    img_folder: Full path of folder where images are stored.
    img_names: A list of names of images within given folder to compute stats over. 
               If img_names == None, computes stats over all names in img_folder. 
               
    Output: Returns [np.array([red_mean,green_mean,blue_mean]),np.array([red_std,green_std,blue_std])]
    """
    
    img_folder = correct_foldername(img_folder)
    if img_names is None: img_names = os.listdir(img_folder)
    N = len(img_names)
    
    means,variances = np.zeros(3),np.zeros(3)
    for name in PBar(img_names):
        img = open_image(img_folder + name)
        means += img.mean(axis=(0,1))
        variances += (img**2).mean(axis=(0,1))
    means = means/N
    variances = variances/N - means**2
    stds = np.sqrt(variances)
    
    return [means,stds]

def get_cat_counts(csv_file,skip_first=True,plot_hist=True):
    
    """
    Get counts of how many images are of each category for single-label classification, 
    and (optionally) plot a histogram of counts. csv_file must be in form below. 
    If csv does not have a header row with column names, then set skip_first=False. 
    
    CSV FORMAT:
    
    img_name, class   <--- Header Row 
    img1.jpeg, dog
    img2.jpeg, table
    img3.jpeg, tree
    ... etc. 
    
    """
    
    if skip_first == False: 
        df = pd.read_csv(csv_file, names = ['img_name','category'])
    if skip_first == True: 
        df = pd.read_csv(csv_file, names = ['img_name','category'], skiprows=1)
    
    cat_counts = df['category'].value_counts()
    cat_counts = pd.DataFrame({'category':list(cat_counts.index),'count':list(cat_counts)})
    
    if plot_hist: 
        plt.hist(cat_counts['count'])
        plt.xlabel('number images in category')
        plt.ylabel('frequency')
        
    return cat_counts
    
def plot_imgsize_histograms(foldername,ranges=None,num_bins=10):
    
    """ Plots histograms of row sizes, col sizes, and aspect ratios of images in specified folder.
    
    Arguments:
    foldername: complete path of folder
    ranges: ranges = (row_range,col_range,aspect_ratio_range) each given in form [min,max].
            If ranges == None, plots complete histograms for row sizes, col sizes, and aspect ratios.                
    num_bins: num_bins = (num_row_bins, num_col_bins, num_aspect_ratio_bins) specifies number 
              of bins for histograms. If a single integer is given for num_bins, it is used for all 3. 
    """
    foldername = correct_foldername(foldername)
    if ranges is None: ranges = (None,None,None)
    if type(num_bins) == int: num_bins = (num_bins,num_bins,num_bins) 
    
    filenames = os.listdir(foldername)
    rowsizes, colsizes, aspect_ratios = [],[],[]
    
    for i in PBar(range(len(filenames))):
        img = plt.imread(foldername+filenames[i])
        rowsizes.append(img.shape[0])
        colsizes.append(img.shape[1])
        aspect_ratios.append(img.shape[1]/img.shape[0])
        
    plt.figure(figsize=(14,10))
    
    plt.subplot(2,2,1)
    plt.hist(rowsizes,range=ranges[0],bins=num_bins[0])
    plt.title('row sizes')
    
    plt.subplot(2,2,2)
    plt.hist(colsizes,range=ranges[1],bins=num_bins[1]) 
    plt.title('col sizes')
    
    plt.subplot(2,2,3)
    plt.hist(aspect_ratios,range=ranges[2],bins=num_bins[2]) 
    plt.title('aspect ratios')
    
# (1.2) Bounding Box Utilities
def hw_to_mm(box): 
    """ Function to convert a height-width bounding box to min-max bounding box. """
    return np.array([ box[0], box[1], box[2]+box[0]-1, box[3]+box[1]-1 ])

def mm_to_hw(box): 
    """ Function to convert a min-max bounding box to height-width bounding box."""
    return np.array([ box[0], box[1], box[2]-box[0]+1, box[3]-box[1]+1 ])

def convert_bbox_list(bbox_list):
    
    """ Function to convert a bbox list from standard form to dataloader form. 
    
        Input: Standard bbox list in form [(b_1,c_1),...,(b_n,c_n)]
        Output: Returns boxes,cats
                boxes = np.array([b_1,...,b_n])
                cats = np.array([c_1,...,c_n)
    """  
    boxes = np.array([box for box,cat in bbox_list])
    cats = np.array([cat for box,cat in bbox_list])     
    return boxes, cats

def rev_bbox_list(bbox_list):
    
    """ Function to convert a bbox list from extended dataloader form to standard form.
    
        Input: [torch.FloatTensor([b_1,...,b_N]),torch.LongTensor([c_1,...,c_N])].
        Output: [(b_1,c_1),...,(b_n,c_n)].        
        
        Here, for n<i<=N, c_i = -1 and b_i = [-1,-1,-1,-1].
        These 'null boxes' are used as padding for the dataloader, 
        so that all images in a batch have same number of bounding boxes.
    """
    
    N = len(bbox_list[0])
    bbox_list = [ARR(bbox_list[0]),ARR(bbox_list[1])]
    bbox_list = [(bbox_list[0][i],bbox_list[1][i]) for i in range(N)]
    for i in range(N):
        if bbox_list[i][1] == -1:
            bbox_list = bbox_list[:i]
            break
            
    return bbox_list    
        
def jaccard(Boxes1,Boxes2):
    
    """ Compute the jaccard index = Area(intersection)/Area(union) of every pair of 
        boxes (b1,b2) with b1 in Boxes1, b2 in Boxes2. Return as n x m torch Tensor, 
        where n = len(Boxes1), m=len(Boxes2).
        
        Arguments:
        Boxes1 = Tensor of size n by 4, Boxes2 = Tensor of size m by 4.
        Each row of Boxes1,Boxes2 is a bounding box in min-max form [xmin,ymin,xmax,ymax]. """
    
    if (len(Boxes1) == 0) or (len(Boxes2)==0): return TEN([])
    Boxes1,Boxes2 = Boxes1.float(),Boxes2.float()
    
    areas1 = (Boxes1[:,2] - Boxes1[:,0])*(Boxes1[:,3] - Boxes1[:,1]) # area of each box in Boxes1 
    areas2 = (Boxes2[:,2] - Boxes2[:,0])*(Boxes2[:,3] - Boxes2[:,1]) # area of each box in Boxes2
    
    B1,B2 = Boxes1.unsqueeze(1), Boxes2.unsqueeze(0)
    inter_w = (torch.min(B1[:,:,2], B2[:,:,2]) - torch.max(B1[:,:,0], B2[:,:,0])).clamp(min=0)
    inter_h = (torch.min(B1[:,:,3], B2[:,:,3]) - torch.max(B1[:,:,1], B2[:,:,1])).clamp(min=0)
    areas_inter = inter_w * inter_h # area of intersections of each pair of boxes (n x m)       
    
    areas_union = areas1.unsqueeze(dim=1) + areas2.unsqueeze(dim=0) - areas_inter # area of unions (n x m)
    return areas_inter/areas_union                

def get_AspectRatioScale(img,min_side,max_side):
    """Compute aspect_ratio and scaling factor for an image.
       For use with ImageDataObj.from_json_bbox() constructor. """ 
        
    rows, cols, channels = img.shape
    aspect_ratio = cols/rows
    smallest_side = min(rows, cols)
    largest_side = max(rows, cols)
    scale = min_side / smallest_side
    if largest_side * scale > max_side:
        scale = max_side / largest_side 
    return aspect_ratio, scale  


# SECTION (2) - IMAGE DISPLAY

def draw_outline(obj, lw):
    "draws a black outline around a matplotlib object"
    obj.set_path_effects([patheffects.Stroke(linewidth=lw, foreground='black'), 
                          patheffects.Normal()])

def get_colors(N):
    """ Function to generate a list of N colors for plotting display."""
    color_norm  = matplotlib.colors.Normalize(vmin=0, vmax=N-1)
    color_map = matplotlib.cm.ScalarMappable(norm=color_norm, cmap='Set3').to_rgba
    color_list = [color_map(float(x)) for x in range(N)]
    return color_list
    
def show_image(img,label=None,bboxes=None,preds=None,categories=None,figsize=(8,8),ax=None):   
    """ A function to display an image. Optionally also can display a class label,
        predicted probabilities of classes, or bounding boxes with their labels. 
    
    Arguments:
    img: An image object (numpy array)
    label: If given is an integer category label, or list of labels. 
    bboxes: If given is a list of form [(b_1,c_1),...,(b_n,c_n)] where:
            each c_j is an integer category label
            each b_j is a bounding box given in form np.array([xmin,ymin,xmax,ymax])
    preds: If given:
           * For single label classification, <preds> is the predicted probabilities 
             of class labels as np.array, e.g. np.array([0.1,0.3,0.6]). Here class 0 has prob 0.1, 
             class 1 has prob 0.3, class 2 has prob 0.6.
           * For bbox object detection, <preds> is an array of length N, where N = number of predicted
             bboxes. preds[i] = confidence score that predicted bbox b_i is of predicted category c_i.
           NOTE: For multi-label classification, always leave as default preds = None.            
    categories: If given is a dictionary of form {0:'cat',1:'tree',2:'boat'} mapping integer 
                category labels to category names.
    figsize: The size of the figure to display, if ax is None.
    ax: The matplotlib axes object to plot figure to.
    """
    
    # the image
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # IF label given AND no preds given: set title to image label
    if (label is not None) and (preds is None):
        if type(label) == int: title = categories[label]
        elif type(label) == list: title = ' '.join([categories[l] for l in label])  
        ax.set_title(title)
       
    # ELIF label and preds both given: set title to image label with predicted probs
    elif (label is not None) and (preds is not None):
        predicted_label_idx = np.argmax(preds)
        prob_true, prob_pred = str(preds[label])[:5], str(preds[predicted_label_idx])[:5]
        true_label, pred_label = categories[label], categories[predicted_label_idx]
        title = 'label: ' + true_label + ' (p=' + prob_true + ') \n'\
                'pred: '+ pred_label + ' (p=' + prob_pred + ')'
        ax.set_title(title)
    
    # ELIf bboxes given: display bounding boxes with their labels superimposed on image
    elif (bboxes is not None): 
        colors = get_colors(len(bboxes))
        for i,(box,cat) in enumerate(bboxes):
            box = mm_to_hw(box)
            xmin,ymin,width,height = box[0],box[1],box[2],box[3]
        
            # draw bbox
            patch = ax.add_patch(patches.Rectangle([xmin,ymin], width, height, 
                                 fill=False, edgecolor=colors[i], lw=2))
        
            # draw label
            fontsize = max(10,min(1.5*figsize[0],16))
            text_str = categories[cat] + ' '
            if preds is not None: text_str += str(preds[i])[:4]    
            text = ax.text(xmin, ymin, text_str, verticalalignment='top', 
                           color=colors[i], fontsize= fontsize, weight='bold')
        
            # draw outlines for box and label
            draw_outline(patch, 4)
            draw_outline(text, 2)  
            
def ShowImages(images,categories=None,num_cols=3,figsize=(16,8)): 
    
    """Function to display a collection of images from a list. Optionally can also display
       class labels, predicted probabilities of classes, or bounding boxes with labels. 
       
       NOTE: Bounding boxes may be the ground truth bounding boxes for train 
       or validation data, or may be the predicted bounding boxes from model. 
    
    Arguments:
    images: A list whose elements are dictionaries, each with the same keys. 
             These keys must incude 'img', may also include 'label','preds','bboxes'. 
             All keys given in same format as for function <show_image>. 
             If one or more of keys 'label','preds','bboxes' is specified,
             then <categories> must also be specified.
    categories: A dictionary of form {0:'cat',1:'tree',2:'boat'}.
    num_cols: Number of images to display in each column.
    figsize: The size of the figure to display.
    """
    
    num_images = len(images)
    num_rows = int(np.ceil(num_images/num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i,ax in enumerate(axes.flat):
        if i == len(images): break
        img = images[i]['img']
        label = images[i]['label'] if 'label' in images[i] else None
        preds = images[i]['preds'] if 'preds' in images[i] else None
        bboxes = images[i]['bboxes'] if 'bboxes' in images[i] else None
        fig_size = (figsize[0]/num_cols,figsize[1]/num_rows)
        show_image(img,label,bboxes,preds,categories,figsize=fig_size,ax=ax)
    plt.tight_layout()
            
def ShowImages_from_folder(foldername,random=True,num_images=6,num_cols=3,figsize=(16,8)):

    """ Function to display a set of images from a given folder. 
        <foldername> is complete path of the folder."""
    
    foldername = correct_foldername(foldername)
    image_names = os.listdir(foldername)
    if random == True: idxs = np.random.choice(len(image_names),num_images,replace=False)
    else: idxs = np.arange(num_images)
    images = [{'img':plt.imread(foldername+image_names[idxs[i]])} for i in range(num_images)]
    ShowImages(images,num_cols=num_cols,figsize=figsize)
    

# SECTION (3) - IMAGE TRANSFORMS

class Transform(object):
    
    """ General class for image transforms, for use with single_label or multi_label
        classification problems. Performs some subsequence of following sequence of 
        transforms on an image in order (depending on parameters):
    
    1. Pad Border
    2. Crop 
    3. Resize 
    4. Random Rotate-Zoom 
    5. Random LR-Flip or Random Dihedral
    6. Random Lighting Adjustment (Balance + Contrast)
    7. Add Random Noise
    8. Normalization
    
    NOTE 1: If Padding is used Cropping and Resizing should not be used. 
    NOTE 2: Cropping and Resizing done before later transforms, because later transforms
            are significantly faster on smaller images. Random Lighting and Normalization 
            will take a MUCH LONGER time if done on larger images. 
    
    Arguments for Initialization:
    tfm_type: 'Basic','SideOn', or 'TopDown'.
               Basic: No LR-Flip or Dihedral 
               SideOn: Random LR-Flip
               TopDown: Random Dihedral 
    crop_type: 'center', 'random', crop_point, or None, where crop_point is a float in range [0,1].
               Let H = img height, W = img width
               Ex) If crop_point = 0.25 and H > W:
                   crops img to a square with same width and y_min = (H-W)*0.25, y_max = y_min + W. 
               Ex) If crop_point = 0.7 and W > H:
                   crops img to a square with same height and x_min = (W-H)*0.7, x_max = x_min + H. 
    pad: int, num pixels of padding added to all sides of image (or None)
    sz: int, all images are resized to (height,width) = (sz,sz) (or None)
    max_deg: maximum degree for random rotation (or None)
    max_zoom: max zoom factor for random zooming (or None)
    bal_range: range to randomly choose bal parameter for lighting adjustment (or None)
    cont_range: range to randomly choose cont parameter for lighting adjustment (or None)
    max_noise: maximum magnitude for random noise term added to img (or None) 
    stats: stats for normalizing image (or None)
    """
    
    def __init__(self, tfm_type, crop_type, pad=None, sz=224, max_deg=10, max_zoom=1.05,
                 bal_range=[-0.05,0.05], cont_range=[0.95,1.05], max_noise=None, stats=imagenet_stats):
        
        if type(sz) == int: sz = (sz,sz)
        self.tfm_type, self.crop_type = tfm_type, crop_type
        self.pad, self.sz, self.max_deg, self.max_zoom = pad, sz, max_deg, max_zoom
        self.bal_range, self.cont_range = bal_range, cont_range 
        self.max_noise, self.stats = max_noise, stats
        
    def __call__(self,img):
        
        # STEP 1: Generate tfm values
        flip = np.random.randint(0,2)
        rot = np.random.randint(0,4)
        if self.max_deg: deg = np.random.uniform(-self.max_deg,self.max_deg)
        if self.max_zoom: zoom = np.random.uniform(1,self.max_zoom) 
        if self.bal_range: bal = np.random.uniform(self.bal_range[0],self.bal_range[1])
        if self.cont_range: cont = np.random.uniform(self.cont_range[0],self.cont_range[1])  
        if self.max_noise: 
            noise = np.random.uniform(-self.max_noise,self.max_noise,(self.sz[0],self.sz[1],3))
            noise = cv2.GaussianBlur(noise,(11,11),0).astype(np.float32)
        
        # STEP 2: Transforms of img 
        rows,cols = img.shape[0],img.shape[1]
        
        # pad border
        if self.pad: img = cv2.copyMakeBorder(img, self.pad, self.pad, self.pad, self.pad, borderType= cv2.BORDER_REFLECT)
        
        # cropping
        L = min(rows,cols)
        if self.crop_type is None: 
            pass
        elif rows > L: 
            if self.crop_type == 'center': r = (rows - L)//2
            elif self.crop_type == 'random': r = np.random.randint(0,rows-L+1)
            elif type(self.crop_type) == float: r = int((rows - L)*self.crop_type)
            img = img[r:r+L,:]           
        elif cols > L: 
            if self.crop_type == 'center': c = (cols - L)//2
            elif self.crop_type == 'random': c = np.random.randint(0,cols-L+1)
            elif type(self.crop_type) == float: c = int((cols - L)*self.crop_type)
            img = img[:,c:c+L]
        
        # resizing
        if self.sz: img = cv2.resize(img, (self.sz[1],self.sz[0]), interpolation=cv2.INTER_LINEAR)
        
        # rotate-zoom
        if self.max_deg:
            M = cv2.getRotationMatrix2D((self.sz[1]//2,self.sz[0]//2),deg,zoom)
            img = cv2.warpAffine(img,M,(self.sz[1],self.sz[0]),borderMode=cv2.BORDER_REFLECT)
        
        # lr-flip or dihedral
        if self.tfm_type in ['SideOn','TopDown'] and flip == 1: img = np.fliplr(img)
        if self.tfm_type == 'TopDown': img = np.rot90(img, rot)
            
        # brightness and contrast
        if self.bal_range:
            mu = np.mean(img,axis=(0,1))
            img = np.clip((img-mu)*cont + bal + mu,0.0,1.0)
            
        # random noise
        if self.max_noise: img = np.clip(img + noise,0.0,1.0)
            
        # normalization
        if self.stats: img = (img - self.stats[0].astype(np.float32))/self.stats[1].astype(np.float32)            
            
        # return output 
        return img       
        
def get_transforms(tfm_type,sz=224,stats=imagenet_stats):  
    
    """Returns pair of transforms [tfm_eval, tfm_aug] for single_label/multi_label image 
       classification problems. tfm_aug is used in training, tfm_eval is used in val and test. 
       Arguments are same as for the class <Transform> """
  
    tfm_eval = Transform('Basic','center',None,sz,None,None,None,None,stats=stats)
    tfm_aug = Transform(tfm_type,'random',None,sz,stats=stats)
    return [tfm_eval, tfm_aug]

class TransformBBox(object):
    """ Class to compute image transforms for bounding box object detection problems. 
    
        Images are scaled horizontally and vertically by the same factor 'scale', 
        but are not cropped or resized to make square. Thus, all objects within 
        the images are preserved, and retain the same aspect ratio. 
        Bounding boxes are also transformed to match transforms of image. 
        
    Arguments for Initialization:
    tfm_type: 'Basic' or 'SideOn' (Basic=No LR-Flip, SideOn=Random LR-Flip).
    bal_range: range to randomly choose bal parameter for lighting adjustment (or None)
    cont_range: range to randomly choose cont parameter for lighting adjustment (or None)
    stats: stats for normalizing image
    jitter: jit_row and jit_col are 2 numbers chosen independently at random from the set {0,1,2,...,jitter}. 
            Images are padded on left and top by jit_col and jit_row pixels of zeros (respectively).
    scale_range: rand_scale is a number chosen uniformly in the range scale_range. 
                 Images are scaled horizontally and vertically by a factor of scale*rand_scale, 
                 where 'scale' is the intrinsic image scale as computed by function get_AspectRatioScale().
    L: integer, should be at least the length of dataset for which given TransformBBox instance is the tfm.
    """
    
    def __init__(self, tfm_type, bal_range=[-0.05,0.05], cont_range=[0.95,1.05], 
                 stats=imagenet_stats, scale_range=[0.8,1.2], jitter=20, L=100000):
        
        self.tfm_type, self.stats, self.jitter, self.L = tfm_type, stats, jitter, L
        self.scale_range, self.bal_range, self.cont_range = scale_range, bal_range, cont_range      
        self.iter = None
    
    def get_values(self):
        self.row_jitter_values = np.random.randint(0,self.jitter+1,self.L)
        self.col_jitter_values = np.random.randint(0,self.jitter+1,self.L)
        self.flip_values = np.random.randint(0,2,self.L)
        self.scale_values = np.random.uniform(self.scale_range[0],self.scale_range[1],self.L)
        self.iter = self.__iter__()
        
    def __iter__(self):
        for row_jit, col_jit, flip, rand_scale in \
        zip(self.row_jitter_values, self.col_jitter_values, self.flip_values, self.scale_values):
            yield row_jit, col_jit, flip, rand_scale
        
    def __call__(self,img,target):
        
        # STEP 0: GET STORED VALUES of row_jit, col_jit, flip, scale, OR GENERATE RANDOM ONES       
        if self.iter: 
            row_jit, col_jit, flip, rand_scale = next(self.iter)
        else: 
            row_jit = np.random.randint(0,self.jitter+1)
            col_jit = np.random.randint(0,self.jitter+1)
            flip = np.random.randint(0,2)
            rand_scale = np.random.uniform(self.scale_range[0],self.scale_range[1])
        
        #STEP 1: TRANSFORM OF img
        
        # brightness and contrast
        if self.bal_range:
            bal = np.random.uniform(self.bal_range[0],self.bal_range[1])
            cont = np.random.uniform(self.cont_range[0],self.cont_range[1]) 
            mu = np.mean(img,axis=(0,1))
            img = np.clip((img-mu)*cont + bal + mu,0.0,1.0)        
        
        # normalization
        img = (img - self.stats[0].astype(np.float32))/self.stats[1].astype(np.float32)
        
        # random LR flip
        if self.tfm_type == 'SideOn' and flip == 1: 
            img = np.fliplr(img)
            
        # Note: Values of row_jit, col_jit, rand_scale are not applied on the img. 
        # Instead they are passed as output to AspectRatioCollater and applied there 
        # as batchwise opertations (using values of first image in batch), 
        # so as to avoid extra padding. 
        
        #STEP 2: TRANSFORM OF target (i.e. bboxes)  
        
        # Note: If for test data, where no bboxes are known, then target = 0.
        if target==0 or len(target)==0:
            bboxes, cats = np.array([]), np.array([])
        else:
            bboxes, cats = convert_bbox_list(target)
            if self.tfm_type == 'SideOn' and flip == 1:
                cols = img.shape[1]
                bboxes = np.array([ cols-bboxes[:,2], bboxes[:,1], cols-bboxes[:,0], bboxes[:,3] ]).T 
                
        # Return Output
        return [img,rand_scale,row_jit,col_jit,bboxes,cats]   

def get_transforms_bbox(tfm_type,jitter=20,scale_range=[0.8,1.2]):  
    
    """Returns pair of transforms [tfm_eval, tfm_aug] for bounding box object detection. 
       tfm_aug is used in training, tfm_eval is used in val and test. """
  
    tfm_eval = TransformBBox('Basic',None,None,jitter=0,scale_range=[1,1])
    tfm_aug = TransformBBox(tfm_type,jitter=jitter,scale_range=scale_range)
    return [tfm_eval, tfm_aug]
        
class TransformBBoxShowPreds(object):
    """ Class to compute image transforms specifically for ImageLearner.show_bbox_preds() 
        method in bbox object detection problems. """
        
    def __init__(self,stats=imagenet_stats):   
        self.stats = stats
      
    def __call__(self,img,scale):  
      
        # scaling
        rows, cols, channels = img.shape
        img = skimage.transform.resize(img,(int(rows*scale),int(cols*scale)),mode='reflect',anti_aliasing=False)
    
        # normalization
        img = (img - self.stats[0].astype(np.float32))/self.stats[1].astype(np.float32)
        
        # padding
        rows, cols, channels = img.shape
        pad_h = 32 - rows%32 if rows%32 > 0 else 0
        pad_w = 32 - cols%32 if cols%32 > 0 else 0
        new_img = np.zeros((rows + pad_h, cols + pad_w, channels)).astype(np.float32)
        new_img[:rows, :cols, :] = img.astype(np.float32) 
        
        return new_img
             
        
# SECTION (4) - DATASETS AND DATAOBJ

class ImageDataset(Dataset):
    """
    A class for an image dataset (either single-label classification, multi-label classification, 
    bounding box object detection, or image segmentation). Can be either train, val, or test dataset.
    
    Arguments for initialization:
    IMG_PATH: Folder where all images for the dataset are stored (possibly in sub-folders).
    images: A list of image dictionaries each with 2 keys: 'img' and 'target'
            * key 'img' gives the image filename within the folder IMG_PATH (e.g. 'img17.jpeg' or 'dogs/dog19.png')
            * key 'target' gives target output for input image with key 'img', form depends on <target_type>. 
              - If <target_type> is 'single_label', 'target' is an integer category label. 
              - If <target_type> is 'multi_label', 'target' is a 0-1 np.array specifying presence of categories in image. 
                Ex) If 'target' is [1,0,0,1,0] categories with integer labels 0 and 3 (out of {0,1,2,3,4}) are present. 
              - If <target_type> is 'bbox', 'target' is a standard bbox list of form [(b_1,c_1),...,(b_n,c_n)].
              - If <target_type> is 'segmentation', ... <<< SPECIFY LATER >>>
             NOTE: For a test dataset, if no targets are known, the key 'target' is given value 0 for all images.
    target_type: Either 'single_label','multi_label','bbox' or 'segmentation'.
    ds_type: Either 'train', 'val', or 'test'
    transform: A transform class to apply to the images.

    Attributes:  
    IMG_PATH, images, transform, target_type, ds_type: same as input.
    y: The target output for dataset, as a list [image['target'] for image in images]. 
    """
    
    def __init__(self,IMG_PATH,images,transform,target_type,ds_type): 
        
        self.IMG_PATH = IMG_PATH
        self.images = images
        self.transform = transform
        self.target_type = target_type
        self.ds_type = ds_type
        self.y = [images[i]['target'] for i in range(len(images))]
                                      
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        img = open_image(self.IMG_PATH + self.images[idx]['img'])
        
        if self.ds_type == 'test':
            target = 0
        elif self.target_type in ['single_label','multi_label','bbox']: 
            target = self.images[idx]['target']
        elif self.target_type == 'segmentation':
            pass
                
        if self.target_type in ['single_label','multi_label']: 
            img = self.transform(img)
            return img.transpose(2,0,1), target 
        elif self.target_type == 'bbox':
            scale = self.images[idx]['scale']
            img, rand_scale, row_jit, col_jit, bboxes, cats = self.transform(img,target)
            return img, scale, rand_scale, row_jit, col_jit, bboxes, cats
        elif self.target_type == 'segmentation':
            pass            

class AspectRatioSampler(Sampler):
    """
    Sampler which groups images in a dataset by their aspect ratios. Images are all ordered according 
    to their aspect ratios, and then this ordered set of images is divided into consecutive groups 
    of size=bs to use for minibatches. This sampler is used for bounding box datasets. 
    """

    def __init__(self, ds, bs):
        self.ds, self.bs = ds, bs
        self.groups = self.group_images()
        
    def group_images(self):
        
        # determine the order of the images
        aspect_ratios = [self.ds.images[i]['aspect_ratio'] for i in range(len(self.ds))]
        order = list(range(len(self.ds)))
        order.sort(key=lambda x: aspect_ratios[x])

        # divide into groups, one group = one batch
        L = len(self.ds)
        return [[order[x] for x in range(i, min(i+self.bs, L))] for i in range(0, L, self.bs)]        

    def __len__(self):
        return len(self.groups)
    
    def __iter__(self):
        np.random.shuffle(self.groups)
        for group in self.groups:
            yield group

def AspectRatioCollater(batch): 
    """
    Collate function to group image batches sampled according to AspectRatioSampler, 
    for use with bounding box datasets. (Also works for batches of single images for 
    bounding box datasets, as used for val).
    
    Images in each batch are padded so that both height and width are each 
    a multiple of 32 pixels, and all images in the batch are same size. 
    Original images are in the upper left corner, with any necessary padding 
    added to right and bottom of images. Also, bbox lists associated with 
    images are all padded to be of same length. 
    
    Input: <batch> is a list of elements, each element of form:
           (img, scale, rand_scale, row_jit, col_jit, bboxes, cats).
           
           For rand_scale, row_jit, and col_jit, the values for 1st 
           element in batch are applied to all imgs in batch.
    
    Output: imgs_padded, [bboxes_padded, cats_padded] where:
            
            * H & W = (max of img heights) & (max of img widths), 
              each padded to be a multiple of 32 pixels.
            * imgs_padded = Tensor of shape (bs x 3 x H x W)            
            * N = max number of bboxes for any img in batch.
            * bboxes_padded = Tensor of shape (bs x N x 4)
            * cats_padded = LongTensor of shape (bs x N)
    """
    
    # collect input
    bs = len(batch)
    imgs = [z[0] for z in batch]
    scales = [z[1] for z in batch]
    bboxes = [z[5] for z in batch]
    cats = [z[6] for z in batch]
    rand_scale = batch[0][2]
    row_jit = batch[0][3]
    col_jit = batch[0][4]
    
    # transform each img in batch, and its bboxes, by resizing and applying jitter
    transformed_imgs = []
    for i in range(bs):
        
        img, scale = imgs[i], scales[i]
        rows, cols, channels = img.shape
        img = cv2.resize(img,(int(cols*scale*rand_scale), int(rows*scale*rand_scale)))         
        if len(bboxes[i]) > 0:
            bboxes[i] = bboxes[i]*scale*rand_scale
        
        rows, cols, channels = img.shape
        rows2, cols2 = rows+row_jit, cols+col_jit   
        new_img = np.zeros((rows2, cols2, channels)).astype(np.float32)
        new_img[row_jit:, col_jit:, :] = img.astype(np.float32)
        if len(bboxes[i]) > 0:
            bboxes[i] = np.array([bboxes[i][:,0]+col_jit, bboxes[i][:,1]+row_jit, 
                                  bboxes[i][:,2]+col_jit, bboxes[i][:,3]+row_jit]).T
       
        new_img = new_img.transpose(2,0,1) #for pytorch Dataloader, need in (C,H,W) order.
        transformed_imgs.append(new_img)
        
    # pad all imgs in batch to same size, which must have height & width = multiple of 32 pixels
    max_height = int(max([img.shape[1] for img in transformed_imgs]))
    max_width = int(max([img.shape[2] for img in transformed_imgs])) 
    max_height, max_width = int(32*np.ceil(max_height/32)), int(32*np.ceil(max_width/32))
    imgs_padded = np.zeros((bs, 3, max_height, max_width)).astype(np.float32)
    for i in range(bs):
        img = transformed_imgs[i]
        imgs_padded[i, :, :img.shape[1], :img.shape[2]] = img
    
    # pad bboxes and cats to be of same length for each img in batch
    max_boxes = max([len(bboxes[i]) for i in range(bs)]) 
    if max_boxes > 0:
        bboxes_padded = np.ones((bs,max_boxes,4)).astype(np.float32) * (-1)
        cats_padded = np.ones((bs,max_boxes)).astype(np.int64) * (-1)
        for i,(b,c) in enumerate(zip(bboxes,cats)):
            if len(b)>0: 
                bboxes_padded[i, :len(b), :] = b
                cats_padded[i, :len(c)] = c
    else: 
        bboxes_padded = np.ones((bs,1,4)).astype(np.float32) * (-1)
        cats_padded = np.ones((bs,1)).astype(np.int64) * (-1)
    
    # return output    
    return TEN(imgs_padded,GPU=False), [TEN(bboxes_padded,GPU=False), TEN(cats_padded,GPU=False)]
    
class ImageDataObj(object):    
    
    """ Class for an image data object encompassing the datasets and 
    corresponding dataloaders for train, validation, and (optionally) test data, 
    along with a bit of extra information. Datasets can be for either 
    single-label classification, multi-label classification, bounding box 
    object detection, or image segmentation. 
    
    Arguments for initialization:
    PATH: Folder where all images are stored within labeled sub-folders (e.g. 'train' or 'val').
    target_type: Either 'single_label','multi_label','bbox', or 'segmentation'.
    categories: A dictionary of form {0:'dog',1:'tree',2:'book'} mapping integer 
                category labels to category names.     
    bs: the batch size to use for dataloaders
    transforms: transforms = [tfm_eval,tfm_aug]. tfm_aug used for training, tfm_eval used for val and test.
    train_images,val_images,test_images: Same format as variable <images> in initialization of ImageDataset. 
                                         test_images may be None, then no test dataset constructed.
    train_name,val_name,test_name: names of folders within PATH containing the train,val,test images.                               num_workers (optional): number of workers, i.e CPUs, to use for dataloaders.
                            NOTE: If target_type='bbox', num_workers is always set to 1, 
                                  regardless of given value. 
    
    Attributes:
    categories, target_type, bs: same as inputs
    train_ds, val_ds, test_ds: datasets for train, val, test
    train_dl, val_dl, test_dl: dataloaders for train, val, test
    sz: For target_type = single_label or multi_label, the size of transformed images 
        in datasets is (height,width) = (sz,sz). Otherwise sz is not defined. 
    """
    
    def __init__(self, PATH, target_type, categories, bs, transforms, train_images, val_images, 
                 test_images=None, train_name='train', val_name='val', test_name=None, num_workers=8):
        
        # basic attributes
        tfm_eval,tfm_aug = transforms[0],transforms[1]
        self.target_type, self.categories, self.bs = target_type, categories, bs
        if target_type in ['single_label','multi_label']: self.sz = tfm_eval.sz  
            
        # datasets
        PATH = correct_foldername(PATH)
        self.train_ds = ImageDataset(PATH+train_name+'/',train_images,tfm_aug,target_type,'train')
        self.val_ds = ImageDataset(PATH+val_name+'/',val_images,tfm_eval,target_type,'val')
        if test_name: self.test_ds = ImageDataset(PATH+test_name+'/',test_images,tfm_eval,target_type,'test')
        else: self.test_ds = None
        
        # dataloaders
        if target_type in ['single_label','multi_label']: 
            self.train_dl = DataLoader(self.train_ds,batch_size=bs,num_workers=num_workers,shuffle=True)
            self.val_dl = DataLoader(self.val_ds,batch_size=bs,num_workers=num_workers,shuffle=False)
            if test_name: self.test_dl=DataLoader(self.test_ds,batch_size=bs,num_workers=num_workers,shuffle=False)
            else: self.test_dl = None
    
        elif target_type == 'bbox': 
            sampler = AspectRatioSampler(self.train_ds, bs)
            self.train_dl = DataLoader(self.train_ds, collate_fn=AspectRatioCollater, batch_sampler=sampler, num_workers=1)
            self.val_dl = DataLoader(self.val_ds, batch_size=1, collate_fn=AspectRatioCollater, num_workers=1, shuffle=False)
            if test_name: 
                self.test_dl = DataLoader(self.test_ds,batch_size=1,collate_fn=AspectRatioCollater,num_workers=1,shuffle=False)
            else: self.test_dl = None
    
        elif target_type == 'segmentation':
            pass    
    
    @staticmethod
    def convert_labels_multi(dataframe,categories_rev):
        
        """ Function to convert a list of category labels in a pd.DataFrame to a presence/absence 
        list for those categories in an image. This function is used as part of the from_csv method 
        in case of multi-label classification.
            
        Arguments:
        dataframe: pd.DataFrame with second column entries, which are lists of categories, e.g. ['dog','cat','tree']. 
        categories_rev: dictionary mapping categories to integer labels, e.g. {'dog':0,'cat':1,'tree':2}.
        
        Output:
        Converts each second column entry of dataframe to a 0-1 numpy array specifying presence or absence of categories.
        Ex) If categories_rev is dictionary given above, then [1,0,1] means image contains categories
            'dog' and 'tree', but not category 'cat'. """
        
        num_cats = len(categories_rev)
        for i in range(len(dataframe)):
            n = len(dataframe.iloc[i][1])
            dataframe.iloc[i][1] = [categories_rev[dataframe.iloc[i][1][j]] for j in range(n)]
            presence_absence = np.zeros(num_cats)
            presence_absence[dataframe.iloc[i][1]] = 1 
            dataframe.iloc[i][1] = presence_absence.astype('float32')
                
    @classmethod
    def from_csv(cls,PATH,transforms,bs,train_csv='train.csv',val_csv=None,test_csv=None,train_name='train',
                 val_name=None,test_name=None,target_type='single_label',val_frac=0.2,skip_first=True,suffix=''):
        
        """Method to construct an ImageDataObj from csv files. 
           Works for target_type = single_label and multi_label. 
        
        Arguments
        PATH: The base path containing all folders of images.
        train_name,val_name,test_name: names of the folders located in PATH which contain the images.
        train_csv,val_csv,test_csv: names of csv files located in PATH which contain the image labels. 
        transforms: [tfm_eval,tfm_aug], tfm_aug used for training, tfm_eval used for val and test.
        bs: the batch size to use for the dataloaders.
        target_type: Type of targets for datasets, 'single_label' or 'multi_label' are only valid choices. 
        val_frac: fraction of training data chosen randomly to use for validation, assuming val_name = None. 
        skip_first: Use skip_first=True if cvs/s contain header row with column names, and skip_first=False if not.
        suffix: A file-type suffix such as '.jpeg' or '.png' which is part of each image
                filename, but not listed in the filenames in the csv files.
        
        NOTE 1: If <test_name> == None, does not contruct a test_ds. If <val_name> == None constructs the
                val_ds by randomly selecting a subset of training data (size of subset is val_frac).
                Also, even if test_name is provided no test_csv is necessary.
       
        NOTE 2: All csv files are assumed to be in the following form.
        
        SINGLE LABEL CSV FORMAT:
        img, category            <---- Header Row (may be missing)
        img1.jpeg, dog
        img2.jpeg, table
        img3.jpeg, tree
        ... etc.
        
        MULTI LABEL CSV FORMAT:
        img, categories             <---- Header Row (may be missing)
        img1.jpeg, dog cat
        img2.jpeg, table chair book
        img3.jpeg, tree
        ... etc.
     
        """ 
        
        # Construct pd.DataFrames for train data 
        PATH = correct_foldername(PATH)
        if skip_first == False: 
            TRAIN = pd.read_csv(PATH + train_csv, names = ['img_name','target'])
        elif skip_first == True: 
            TRAIN = pd.read_csv(PATH + train_csv, names = ['img_name','target'], skiprows=1)
        if target_type == 'multi_label': 
            TRAIN['target'] = TRAIN['target'].str.split()
            
        # Get the dictionaries <categories> and <categories_rev>.        
        if target_type == 'single_label':    
            category_names = TRAIN['target'].unique()
            category_names.sort()
        if target_type == 'multi_label':            
            category_names = set() 
            for i in range(len(TRAIN)):
                for cat in TRAIN.iloc[i][1]: category_names.add(cat)
            category_names = list(category_names)
            category_names.sort()
        categories = {i:category_names[i] for i in range(len(category_names))}
        categories_rev = {category_names[i]:i for i in range(len(category_names))}
        
        #Construct pd.DataFrames for val and test data 
        if val_csv and skip_first == False: 
            VAL = pd.read_csv(PATH + val_csv, names = ['img_name','target'])
            if target_type == 'multi_label': VAL['target'] = VAL['target'].str.split()
        elif val_csv and skip_first == True:
            VAL = pd.read_csv(PATH + val_csv, names = ['img_name','target'],skiprows=1)
            if target_type == 'multi_label': VAL['target'] = VAL['target'].str.split()
        else: 
            TRAIN,VAL = SplitTrainVal(TRAIN,val_frac=val_frac)
            TRAIN.index, VAL.index = range(len(TRAIN)),range(len(VAL))
            val_name = train_name
        
        if test_name and test_csv and (skip_first == False):
            TEST = pd.read_csv(PATH + test_csv, names = ['img_name','target'])
            if target_type == 'multi_label': TEST['target'] = TEST['target'].str.split()
        elif test_name and test_csv and (skip_first == True):
            TEST = pd.read_csv(PATH + test_csv, names = ['img_name','target'],skiprows=1)
            if target_type == 'multi_label': TEST['target'] = TEST['target'].str.split()
        elif test_name and not test_csv:
            TEST = pd.DataFrame({'img_name':os.listdir(PATH + test_name),'target':0})        
        else: TEST = None
        
        # Add suffix to file names in DataFrames
        TRAIN['img_name'] = TRAIN['img_name'] + suffix
        VAL['img_name'] = VAL['img_name'] + suffix
        if test_csv: TEST['img_name'] = TEST['img_name'] + suffix
                
        # If for single-label classification, relabel categories with integer labels.
        if target_type == 'single_label':               
            TRAIN['target'] = TRAIN['target'].astype('category')
            TRAIN['target'] = (TRAIN['target'].cat.rename_categories(categories_rev)).astype('int64')
            VAL['target'] = VAL['target'].astype('category')
            VAL['target'] = (VAL['target'].cat.rename_categories(categories_rev)).astype('int64')
            if test_csv:
                TEST['target'] = TEST['target'].astype('category')
                TEST['target'] = (TEST['target'].cat.rename_categories(categories_rev)).astype('int64')
         
        # If for multi-label classification, relabel categories with integer labels. 
        if target_type == 'multi_label':            
            ImageDataObj.convert_labels_multi(TRAIN,categories_rev)
            ImageDataObj.convert_labels_multi(VAL,categories_rev)
            if test_csv: ImageDataObj.convert_labels_multi(TEST,categories_rev)
            
        # Construct dictionaries for train_images, val_images, test_images from pd.DataFrames
        train_images = [{'img':TRAIN['img_name'][i],'target':TRAIN['target'][i]} for i in range(len(TRAIN))]
        val_images = [{'img':VAL['img_name'][i],'target':VAL['target'][i]} for i in range(len(VAL))]
        if TEST is not None: test_images = [{'img':TEST['img_name'][i],'target':TEST['target'][i]} for i in range(len(TEST))]
        else: test_images = None
        
        # Return Output
        return cls(PATH, target_type, categories, bs, transforms, train_images, 
                   val_images, test_images, train_name, val_name, test_name)
        
    @classmethod
    def from_folders(cls,PATH,transforms,bs,train_name='train', 
                     val_name=None,test_name=None,val_frac=0.2):
        
        """ Method to construct an ImageDataObj from labeled folders of images. 
            Works only for target_type = single_label.  
            
        Arguments
        PATH: The base path containing all folders of images.
        train_name: Name of a folder located in PATH which contains subfolders with names of each 
                    image category. Each subfolder contains trains images of that category. 
        val_name: Same as train_name, but for val data. If val_name == None, subset of training data
                  is chosen randomly for validation. 
        test_name: Name of a folder located in PATH containing test images (no labeled subfolders).
        transforms: [tfm_eval,tfm_aug], tfm_aug used for training, tfm_eval used for val and test.
        bs: the batch size to use for the dataloaders
        val_frac: fraction of training data chosen randomly to use for validation, if val_name == None. 
        """
        
        PATH = correct_foldername(PATH)
        category_names = os.listdir(PATH+train_name)
        category_names.sort()
        categories = {i:category_names[i] for i in range(len(category_names))}
        categories_rev = {category_names[i]:i for i in range(len(category_names))}
        
        train_images = []
        for cat in category_names:
            cat_images = os.listdir(PATH+train_name+'/'+cat)
            train_images += [{'img':cat+'/'+img,'target':categories_rev[cat]} for img in cat_images]
        
        if val_name:
            val_images = []
            for cat in category_names:
                cat_images = os.listdir(PATH+val_name+'/'+cat)
                val_images += [{'img':cat+'/'+img,'target':categories_rev[cat]} for img in cat_images]
        else: 
            val_name = train_name
            train_images,val_images = SplitTrainVal(train_images,val_frac=val_frac)
          
        if test_name: 
            test_images = [{'img':img,'target':0} for img in os.listdir(PATH+test_name)]
        else: test_images = None
                
        return cls(PATH, 'single_label', categories, bs, transforms, train_images, 
                   val_images, test_images, train_name, val_name, test_name)
          
    @classmethod    
    def from_json_bbox(cls,PATH,transforms,bs,train_json='train.json',val_json=None,test_json=None,
                       train_name='train',val_name=None,test_name=None,val_frac=0.2,suffix='',get_ARS=[608,1216]):
        
        """ Method to construct a bounding box ImageDataObj from labeled json annotation files. 
            Works for Pascal and COCO datasets (or any other dataset with same annotation format).  
            
        Arguments
        PATH: The base path containing all folders of images.
        train_name,val_name,test_name: names of the folders located in PATH which contain the images.
        train_json,val_json,test_json: names of json files located in PATH which contain the image labels. 
        transforms: [tfm_eval,tfm_aug], tfm_aug used for training, tfm_eval used for val and test.
        bs: the batch size to use for the train dataloader (bs=1 is used for val and test)
        val_frac: fraction of training data chosen randomly to use for validation, assuming val_name = None. 
        suffix: A file-type suffix such as '.jpeg' or '.png' which is part of each image
                filename, but not listed in the filenames in the json files.
        get_ARS: get_ARS = [min_side,max_side], parameters used for function get_AspectRatioScale() 
                 to compute base scaling factor 'scale' for each image.
        
        NOTE 1: If <test_name> == None, does not contruct a test_ds. If <val_name> == None constructs the
                val_ds by randomly selecting a subset of training data (size of subset is val_frac).
                Also, even if test_name is provided no test_json is necessary.
       
        NOTE 2: json files are assumed to be dictionaries containing following keys for train data, or val if given:
                <images>, <annotations>, <categories>. (Format used for standard Pascal and COCO datasets).
        
        <categories>: A list of dictionaries, one for each category (i.e. class) of objects in images.
                      Each entry for a single category is a dictionary containing the keys: 
                      <id> (e.g. 23) and <name> (e.g. 'cat'). 
                      
        <images>: A list of dictionaries, one for each image in dataset.
                  Each entry for a single image is a dictionary containg the keys:
                  <id> (e.g. 2374) and <file_name> (e.g. 'img_002374.jpg')
                  
        <annotations> A list of dictionaries, one for each ground truth bbox of an image in dataset.
                      Each entry for a single annotation (i.e. bbox) is a dictionary containing the keys:
                      <image_id>, e.g. 2374
                      <bbox>, e.g. [17.2,40,100.5,96] = [xmin,ymin,width,height]
                      <category_id>, e.g. 23
        """

        PATH = correct_foldername(PATH)
        
        # open json files
        with open(PATH + train_json) as json_file: 
            trn_json = json.load(json_file)
        
        if val_json:
            with open(PATH + val_json) as json_file: 
                val_json = json.load(json_file)
                
        if test_json:
            with open(PATH + test_json) as json_file: 
                test_json = json.load(json_file)
                
        # abstract categories and conversion dicts
        cats = trn_json['categories']
        categories = {i:cats[i]['name'] for i in range(len(cats))}
        cat2dscat = {i:cats[i]['id'] for i in range(len(cats))}
        dscat2cat = {v:k for k,v in cat2dscat.items()}
        
        # create train_images
        train_images = {} 
        print('getting train images')
        for image in PBar(trn_json['images']):
            ID, img_name = image['id'], image['file_name'] + suffix
            train_images[ID] = {'id':ID,'img':img_name,'target':[]}
            img = cv2.imread(PATH + train_name + '/' + img_name)
            aspect_ratio,scale = get_AspectRatioScale(img,get_ARS[0],get_ARS[1])
            train_images[ID]['aspect_ratio'] = aspect_ratio
            train_images[ID]['scale'] = scale
        for ann in trn_json['annotations']:
            if (('ignore' in ann) and (ann['ignore']==1)) or (('iscrowd' in ann) and (ann['iscrowd']==1)): 
                continue
            ID = ann['image_id']
            cat = dscat2cat[ann['category_id']]
            box = hw_to_mm(ann['bbox'])
            train_images[ID]['target'].append((box,cat))
        train_images = list(train_images.values())
        
        # create val_images
        if val_name:
            val_images = {} 
            print('getting val images')
            for image in PBar(val_json['images']):
                ID, img_name = image['id'], image['file_name'] + suffix
                val_images[ID] = {'id':ID,'img':img_name,'target':[]}
                img = cv2.imread(PATH + val_name + '/' + img_name)
                aspect_ratio,scale = get_AspectRatioScale(img,get_ARS[0],get_ARS[1])
                val_images[ID]['aspect_ratio'] = aspect_ratio
                val_images[ID]['scale'] = scale   
            for ann in val_json['annotations']:
                if (('ignore' in ann) and (ann['ignore']==1)) or (('iscrowd' in ann) and (ann['iscrowd']==1)): 
                    continue
                ID = ann['image_id']
                cat = dscat2cat[ann['category_id']]
                box = hw_to_mm(ann['bbox'])
                val_images[ID]['target'].append((box,cat))
            val_images = list(val_images.values())
        else: 
            train_images,val_images = SplitTrainVal(train_images,val_frac=val_frac)
            val_name = train_name
        
        #create test_images
        if test_name and test_json:
            test_images = {} 
            for image in test_json['images']:
                ID, img_name = image['id'], image['file_name'] + suffix
                test_images[ID] = {'id':ID,'img':img_name,'target':[]}                 
            for ann in test_json['annotations']:
                if (('ignore' in ann) and (ann['ignore']==1)) or (('iscrowd' in ann) and (ann['iscrowd']==1)): 
                    continue
                ID = ann['image_id']
                cat = dscat2cat[ann['category_id']]
                box = hw_to_mm(ann['bbox'])
                test_images[ID]['target'].append((box,cat))
            test_images = list(test_images.values())
        elif test_name:
            test_images = [{'img':fn,'target':0} for fn in os.listdir(PATH + test_name)] 
            del_idxs = []
            for i,x in enumerate(test_images):
                if x['img'][:2] == '._': del_idxs.append(i)
            test_images = list_del(test_images,del_idxs)
        else: test_images = None
        
        if test_name:
            print('getting test images')
            for i in PBar(range(len(test_images))):
                img = cv2.imread(PATH + test_name + '/' + test_images[i]['img'])
                aspect_ratio,scale = get_AspectRatioScale(img,get_ARS[0],get_ARS[1])
                test_images[i]['aspect_ratio'] = aspect_ratio
                test_images[i]['scale'] = scale          
        
        # return output
        data = cls(PATH, 'bbox', categories, bs, transforms, train_images, 
                   val_images, test_images, train_name, val_name, test_name)
        
        data.cat2dscat = cat2dscat
        return data                       


# SECTION (5) - ImageClassificationNet and ImageClassificationEnsembleNet
      
def default_cut(model): 
    """ Function to cut off the last few layers of a body arch coming from certain pretrained models 
        (resnet, resnext, senet, inceptionV4), such that the output of the cut model is the spatial "features" 
        of an image, inferred prior to avgpool/maxpool and classification. If model is not one of the above 
        types, then no cut is applied. This no-cut is compatible with inceptionresnetV2 and nasnetalarge,
        implementations in this library, which only output final features of model by default."""        
    if isinstance(model,models.ResNet):
        return nn.Sequential(*list(model.children())[:-2])
    elif isinstance(model,(vmods.ResNeXt101_32x4d, vmods.ResNeXt101_64x4d)):
        return model.features
    elif isinstance(model, vmods.SENet):
        return nn.Sequential(*list(model.children())[:5])
    elif isinstance(model,vmods.InceptionV4):
        return model.features
    else: return model
    
def default_split(precut_body,body):
    """ Function to split a body arch coming from certain pretrained models (resnet, resnext, senet, inceptionV4)
        into a set of 2 layer groups, about 1/2 way through. Any other model is not split. """        
        # NOTE: This splitting is useful for differential learning rate training with pretrained models.         
    if isinstance(precut_body,models.ResNet):
        m1 = nn.Sequential(*list(body.children())[:6])
        m2 = nn.Sequential(*list(body.children())[6:])
        return [m1,m2]
    elif isinstance(precut_body,(vmods.ResNeXt101_32x4d,vmods.ResNeXt101_64x4d)):
        m1 = nn.Sequential(*list(body.children())[:6])
        m2 = nn.Sequential(*list(body.children())[6:])
        return [m1,m2]
    elif isinstance(precut_body,vmods.SENet):
        m1 = nn.Sequential(*list(body.children())[:3])
        m2 = nn.Sequential(*list(body.children())[3:])
        return [m1,m2]
    elif isinstance(precut_body,vmods.InceptionV4):
        m1 = nn.Sequential(*list(body.children())[:11])
        m2 = nn.Sequential(*list(body.children())[11:])
        return [m1,m2]
    else: 
        return [body]

class ImageClassificationNet(nn.Module):
    
    """ Class for a general 2 part convolutional neural network for image classification
        (single or multi label), consisting of a pretrained "body" architecture followed 
        by a user specified "head".
                
        Arguments for Initialization:      
        
        data: A data object of class ImageDataObj, with target_type = 'single_label' or 'multi_label'.         
        
        arch: The body architecture, of class nn.Module or subclass thereof. 
              Can be user specified arch, or any of built in pytorch models (e.g. resnet34),
              or one of pretrained models from folder VisionModels. 
        
        cutpoint: Defines a point to cut the pretrained body arch at, such that the output 
                  of the cut model is the spatial "features" of an image, inferred prior to 
                  avgpool/maxpool and classification. This can be specified in 3 ways:                  
                  (1) If <cutpoint> == None, no cut is applied to the body arch. 
                      This assumes body arch outputs features directly, rather than logits. 
                  (2) If <cutpoint> == 'default', the function default_cut() is used.
                      Works with certain pretrained models, as in docstring of that function. 
                  (3) If <cupoint> == n, for an integer n, then cuts model at layer of index n.  
                      Here index of layer = index in list(body.children()).
                           
        head: The head architecture of class nn.Module. May be specified in 3 ways:
              (1) As an actual nn.Module to be used directly.  
              (2) As a list [layer_sizes,drops]. In this case:
                  head = nn.Sequential(AdaptiveConcatPool2d(),Flatten(),fully_connected) where,
                  fully_connected is a FullyConnectedNet object specified by layer_sizes and drops. 
              (3) As 'default' which is treated same as [layer_sizes=[512],drops=[0.25,0.25]].   
        
        splits: Determines where to split body arch into layer groups. Head is always 
                treated as a single layer group. May be specified in 4 ways:                 
                (1) If None: entire body is 1 layer group.
                (2) If 'default': Function default_split() is used. 
                (3) If list of indices: These are indices in list(body.children()) to split at.
                (4) If nn.ModuleList: These modules, in sequential order, should make up the body (after cutting).               
                                
        Attributes:
        body: The body architecture (after cutting).
        head: The head architecture. 
        layer_groups: List of layer groups [G1,...,GN], which partition the set of all layers 
                      in the model. Each layer group Gi is an nn.Module or nn.ModuleList.
                      Body and head always put in separate groups by default. 
                      Body can be split further as specified by user with <splits>. 
        param_groups: List of form [G1_1,...,G1_N,G2_1,...,G2_N] where:
                      G1_i consists of all non-batchnorm layers in layer group Gi.
                      G2_i consists of all batchnorm layers in layer group Gi. 
        """
        
    def __init__(self,data,arch,head='default',cutpoint='default',splits='default'):
        
        super().__init__()
        
        # construct the body   
        if cutpoint is None: 
            self.body = arch
        elif cutpoint == 'default': 
            self.body = default_cut(arch)
        elif type(cutpoint) == int: 
            self.body = nn.Sequential(*list(arch.children())[:cutpoint])
             
        # construct the head
        if isinstance(head,nn.Module): 
            self.head = head
        else:
            if type(head) == list: layer_sizes, drops = head[0], head[1]    
            elif head == 'default': layer_sizes, drops = [512], [0.25, 0.25]
            test_input = torch.zeros(1,3,data.sz[0],data.sz[1])
            nfeats = self.body(test_input).shape[1]  #num features in output of body
            ncats = len(data.categories)             #num categories for classification
            layer_sizes = [2*nfeats]+layer_sizes+[ncats]
            fully_connected = FullyConnectedNet(layer_sizes,drops)
            self.head = nn.Sequential(AdaptiveConcatPool2d(),Flatten(),fully_connected)
        
        # construct the layer_groups and param_groups
        if splits is None: 
            body_groups = [self.body]    
        elif type(splits) == str: 
            body_groups = default_split(arch,self.body)
        elif type(splits) == nn.ModuleList: 
            body_groups = [G for G in splits]   
        elif type(splits) == list:
            body_groups, layers = [], list(self.body.children())
            idxs = [0] + splits + [len(layers)]
            for i in range(len(idxs)-1):
                group = layers[idxs[i]:idxs[i+1]]
                body_groups.append(nn.Sequential(*group))      
        
        self.layer_groups = body_groups + [self.head]
        self.param_groups = separate_bn_layers(self.layer_groups)
        
    def forward(self,x_batch):
        return self.head(self.body(x_batch))

class ImageClassificationEnsembleNet(nn.Module):
    
    """Class for an ensemble model for image classification (single or multi label).
    
       Arguments for Initilization:
       
       models: A list of models, all trained on same image classification dataset. 
               Models do not have to have same architecture.           
       weights: A list of weights for averaging the outputs of models, should sum to 1. 
                If weights is None, all model weights are equal. 
       correction: Either 'single_label' or 'multi_label'. 
       
       Output:
       Returns a single model, which given an input x returns a weighted 
       average of the outputs of the individual models, after appropriate 
       correction (i.e. softmax correction for single_label, sigmoid for multi_label). 
       
       """
    
    def __init__(self,models,weights=None,correction='single_label'):
        
        super().__init__()        
        n = len(models)
        if weights: self.weights = weights
        else: self.weights = [1/n]*n
        self.correction = correction
        self.models = nn.ModuleList(models)
        self.layer_groups = models
        self.param_groups = separate_bn_layers(self.layer_groups)
        
    def forward(self,x):
        if self.correction == 'single_label':
            return sum(self.weights[i]*F.log_softmax(m(x),dim=1).exp() for i,m in enumerate(self.models))
        elif self.correction == 'multi_label':
            return sum(self.weights[i]*m(x).sigmoid() for i,m in enumerate(self.models))    
    

# SECTION (6) - ObjectDectionNet and OBJECT DETECTION LOSSES + METRICS

# See "Focal Loss for Dense Object Detection" https://arxiv.org/pdf/1708.02002.pdf 
# for model description and focal loss function used for classification of objects."""

# (6.1) ObjectDetectionNet
class ObjectDetectionNet(nn.Module):
    
    """ 
    This class uses a retinanet model with resnet50 backbone for bounding box object detection.
    
    Upon initialization, loads a model pretrained on COCO (33.4 MAP on [0.5,0.55,...,0.95] COCO thresholds).
    Retains pretrained resnet and feature_pyramid_network components, but uses randomly re-initialized 
    classifier and regressor components (with user specified number of classes, anchors, and prior). 
    Use for transfer learning to other bounding box datasets. 
        
    Arguments:
    num_classes = number of possible classes of objects to detect.
    ratios: aspect ratios for anchor boxes.
    scales: scaling factors for anchor boxes, relative to their base sizes.
    prior: prior probability that an anchor box contains an object of each class.
    feature_size: Before output layer, classifier and regressor components have 4 consecutive 
                  conv layers, each with num_channels_in/out = feature_size (and each followed by ReLU). 
    bn: If True, bn is used after ReLU for each of 4 conv layers in classifier and regressor components,
        and also before the first conv layer in classifier and regressor components. 
    drop: [drop0,drop1] or None. 
          drop0 = dropout prob before first conv layer in classifier and regressor components. 
          drop1 = dropout prob after ReLU for each of 4 conv layers in classifier and regressor components.
    """
    
    def __init__(self, num_classes, ratios=[0.5,1,2], scales=[2**0, 2**(1/3), 2**(2/3)], 
                 prior=0.01, feature_size=256, bn=False, drop=None):
        
        super().__init__()
        
        # Extract Pretrained Componets of Model: resnet backbone and fpn
        model = vmods.retinanet.retinanet()
        self.layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = model.layer1, model.layer2, model.layer3, model.layer4
        self.resnet = nn.ModuleList([self.layer0, self.layer1, self.layer2, self.layer3, self.layer4])
        self.fpn = model.fpn
                                    
        # Create New Randomly Initialized Classifier and Regressor Components
        num_features_in = 256
        num_anchors = len(ratios)*len(scales)
        self.classifier = vmods.retinanet.ClassificationModel(num_features_in, num_anchors, num_classes, feature_size, bn, drop)
        self.regressor = vmods.retinanet.RegressionModel(num_features_in, num_anchors, feature_size, bn, drop)
        self.head = nn.ModuleList([self.classifier,self.regressor])
        
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2/n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
                
        nn.init.constant_(self.classifier.output.weight, 0)
        nn.init.constant_(self.classifier.output.bias, -np.log((1.0-prior)/prior))
        nn.init.constant_(self.regressor.output.weight, 0)
        nn.init.constant_(self.regressor.output.bias, 0)
        
        # Create Layer Groups and Param Groups
        self.layer_groups = [self.resnet, self.fpn, self.head]
        self.param_groups = separate_bn_layers(self.layer_groups)
        
        # Define AnchorGenerator and BBoxPredictor
        self.AnchorGenerator = vmods.retinanet.AnchorGenerator(ratios,scales)
        self.BBoxPredictor = vmods.retinanet.BBoxPredictor()

    def forward(self,x):
        """ 
        Input: x is a batch of images (standard shape, bs x C x H x W)
        Ouput: A list [anchors,reg,clas]
        * anchors: (N x 4) Tensor of anchor boxes associated with images in img_batch.
        * reg: (bs x N x 4) Tensor of bbox regression activations for anchor boxes 
               of each img in img_batch. Default locations of anchor boxes are shifted to predicted 
               locations using these reg activations according to self.BBoxPredictor method. 
        * clas: (bs x N x num_classes) Tensor of classification activations for anchor boxes 
                of each img in img_batch, clas[i,j,k] = Prob(box j in img i is of class k). 
                      
        (Here N = number of anchor boxes for each img in img_batch, depends on img_batch dimensions.)
        """

        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])
        reg = torch.cat([self.regressor(feature) for feature in features], dim=1)
        clas = torch.cat([self.classifier(feature) for feature in features], dim=1)
        anchors = self.AnchorGenerator(x)
        
        return [anchors,reg,clas] 

# (6.2) Single-Shot-Detection Loss used for training + associated functions 
def match_anchors_objects(objects,anchors,pos_thresh=0.5,neg_thresh=0.4):
    """
    Function to match anchor boxes to the ground truth objects in a single image 
    which they are supposed to predict. Each anchor box is initially matched 
    to the object with which it has the highest jaccard overlap. But only matches
    with a high enough jaccard overlap are retained. 
    
    Arguments: (Let N = total number of anchor boxes for the image, m = number objects in image)
    anchors: (N x 4) Tensor, each row is an anchor box in min-max form [xmin,ymin,xmax,ymax]
    objects: (m x 4) Tensor, each row specifies bbox of an object in the image in min-max form. 
    pos_thresh: Minimum jaccard threshold for an anchor box to match with an object. 
    neg_thresh: Maximum jaccard threshold for an anchor box to be defined as NOT matched with an object. 
    NOTE: Must have pos_thresh >= neg_thresh. Any anchor box that has a maximum jaccard score
          in range [neg_thresh,pos_thresh] is considered "undetermined" and not used in computing loss. 
    
    Output: Returns pos_idxs, neg_idxs, matches
    Index objects as 0,1,...,m-1 and anchor boxes as 0,1,...,N-1.    
    * pos_idxs is a Tensor containing a list of idxs of anchor boxes that match with some object.
    * neg_idx is a Tensor containing a list of idxs of anchor boxes defined to have no match.
    * matches is a length N Tensor such that:
      1. If i in pos_idxs, matches[i] = idx of object to which anchor box i is matched.
      2. If i not in pos_idx, matches[i] = -1.
    """
    
    if len(objects) == 0:
        pos_idxs = torch.LongTensor([]).cuda()
        neg_idxs = torch.arange(len(anchors)).cuda()
        matches = -1*torch.ones(len(anchors)).long().cuda()   
    
    if len(objects) > 0: 
        jac = jaccard(objects,anchors)
        max_values, max_idxs = torch.max(jac,dim=0) 
        pos_idxs = (max_values > pos_thresh).nonzero().view(-1)
        neg_idxs = (max_values < neg_thresh).nonzero().view(-1)
        matches = (max_values > pos_thresh).long() * (max_idxs + 1)
        matches = matches - 1
    
    return pos_idxs, neg_idxs, matches 

def focal_loss_retina(pred,target,alpha=0.25,gamma=2.0):
    """
    Function to compute focal loss for a single image in bounding box object detection. 
    
    Arguments: Let n = (num pos_idxs) + (num neg_idxs) for image from function match_anchors_objects.
    pred: (n x num_classes) Tensor corresponding to predicted classes for anchors boxes in pos_idxs + neg_idxs:
           preds[i,j] = prob(box i predicts class j).
    target: (n x num_classes) Tensor corresponding to target classes for anchors boxes in pos_idxs + neg_idxs:
            each row is either all 0's, or a single 1 (for ground truth class) and all other entries 0.
    alpha, gamma: numerical parameters used in focal loss function (see code).
    """
    p = pred.clamp(1e-4, 1.0 - 1e-4) # For Numerical Stability
    pt = p*target + (1-p)*(1-target)
    wa = alpha*target + (1-alpha)*(1-target)
    w = wa*(1-pt).pow(gamma)
    losses = -w*(target*torch.log(p) + (1-target)*torch.log(1-p))
    num_positive_idxs = target.sum()
    return losses.sum()/num_positive_idxs.clamp(min=1)

def smoothL1_loss_retina(anchs,pred_shift,target):
    """
    Function to compute the smooth L1 loss for a single image in bounding box object detection. 
    
    Arguments: Let n = num pos_idxs for image from function match_anchors_objects.
    anchs: (n x 4) Tensor of pos_idx anchor boxes in min-max form [xmin,ymin,xmax,ymax].
    target: (n x 4) Tensor of ground truth bounding boxes matched with anchor boxes in anchs.
    pred_shift: (n x 4) Tensor of bbox regression activations used to shift anchor boxes in anchs 
                to predicted bbox locations (using method in vmods.retinanet.BBoxPredictor).
    """
    
    anch_w = anchs[:,2] - anchs[:,0]    
    anch_h = anchs[:,3] - anchs[:,1]
    anch_cx = anchs[:,0] + 0.5*anch_w
    anch_cy = anchs[:,1] + 0.5*anch_h
    
    target_w = target[:,2] - target[:,0]
    target_h = target[:,3] - target[:,1]
    target_cx = target[:,0] + 0.5*target_w
    target_cy = target[:,1] + 0.5*target_h
    
    target_w = target_w.clamp(min=1) #for numerical stability
    target_h = target_h.clamp(min=1) #for numerical stability
        
    dx = (target_cx - anch_cx)/anch_w  
    dy = (target_cy - anch_cy)/anch_h
    dw = torch.log(target_w/anch_w)
    dh = torch.log(target_h/anch_h)       
    
    true_shift = torch.stack((dx, dy, dw, dh)).t()                         
    true_shift = true_shift/TEN([[0.1, 0.1, 0.2, 0.2]])
    
    diff = torch.abs(true_shift - pred_shift)
    losses = 0.5*9*pow(diff,2) * (diff<1/9).float() + (diff - 0.5/9) * (diff>=1/9).float()
    return losses.mean()  

def ssd1(anchors,bboxes,cats,reg,clas,alpha=0.25,gamma=2.0):
    """ 
    Computes smoothL1_loss and focal_loss for a single image. 
    The ssd loss (single-shot-detection loss) is a weighted combination of these. 
       
    Arguments: (N = num anchor boxes for image, M = num ground truth objects in image.)               
    anchors: (N x 4) Tensor of anchor boxes in min-max form [xmin,ymin,xmax,ymax]
    bboxes: (M x 4) Tensor of ground truth bounding boxes of objects in image.
    cats: Length M Tensor of ground truth classes (categories) of each object in image. 
    reg: (N x 4) Tensor of bbox regression activations used to shift anchor boxes to predicted locations.  
    clas: (N x num_classes) Tensor of classification activations, clas[i][j] = Prob(box i is of class j).
    """
    
    N = len(anchors)
    num_classes = clas.shape[1]
    
    # compute index sets
    pos_idxs, neg_idxs, matches = match_anchors_objects(bboxes,anchors)
    well_defined_idxs = torch.cat([pos_idxs,neg_idxs])
    if len(pos_idxs) > 0:
        obj_idxs = matches[pos_idxs]
        cat_idxs = cats[obj_idxs]
    
    # compute cat_targ, cat_preds, box_targ, box_preds, anchs
    cat_targ = torch.zeros(N,num_classes).cuda()
    for i in range(len(pos_idxs)): cat_targ[pos_idxs[i],cat_idxs[i]] = 1
    cat_targ = cat_targ[well_defined_idxs]
    cat_preds = clas[well_defined_idxs]
    if len(pos_idxs) > 0:
        box_targ = bboxes[obj_idxs]
        box_preds = reg[pos_idxs]
        anchs = anchors[pos_idxs]     
    
    # compute losses
    clas_loss = focal_loss_retina(cat_preds,cat_targ,alpha,gamma)
    if len(pos_idxs) > 0: reg_loss = smoothL1_loss_retina(anchs,box_preds,box_targ)
    else: reg_loss = TEN(0.)
    return reg_loss, clas_loss

class SSD_loss(object):
    
    """Class for the SSD_Loss function, i.e. weighted combo of smoothL1_loss for 
       bbox regression and focal_loss for classification.
    
    Arguments for Initilization:
    beta: Weight for focal loss in range [0,1]. SSD_loss = (1-beta)*(reg_loss) + beta*clas_loss.
    alpha, gamma: parameters for focal loss.
    """
   
    def __init__(self, beta=0.5, alpha=0.25, gamma=2.0):
        self.beta, self.alpha, self.gamma = beta, alpha, gamma
        
    def __call__(self,activ,target):
        
        """ 
        Arguments: (Let M = max num objects of any image in batch.) 
        activ = [anchors, reg, clas], the output of ObjectDetectionNet
        target = [BBoxes, Cats] where:
                 * BBoxes is (bs x M x 4) Tensor, of ground truth bounding boxes in min-max form.
                   If an image i has m<M objects, then for all j>=m, each row BBoxes[i][j] = [-1,-1,-1,-1].
                 * Cats is (bs x M) Tensor of ground truth non-negative integer class labels. 
                   If an image i has m<M objects, then for all j>=m, Cats[i][j] = -1. 
        """
        
        BBoxes, Cats, bs = target[0], target[1], len(target[0])
        anchors, reg, clas = activ[0], activ[1], activ[2]
        reg_loss, clas_loss  = TEN(0.), TEN(0.)
                
        for i in range(bs):
            bboxes =  BBoxes[i][BBoxes[i]>=0].view(-1,4)
            cats = Cats[i][Cats[i]>=0]
            regloss,clasloss = ssd1(anchors,bboxes,cats,reg[i],clas[i],self.alpha,self.gamma)
            reg_loss += regloss
            clas_loss += clasloss
    
        self.reg_loss, self.clas_loss = reg_loss/bs, clas_loss/bs
        return (1-self.beta)*(reg_loss/bs) + self.beta*(clas_loss/bs)
       
class SSD_RegLoss(object):
    
    """Class to extract the SSD reg_Loss from the attribute of class SSD_loss. 
    Use as a metric in training to observe balance of SSD reg_loss and SSD clas_loss. 
    NOTE: Must have learner.loss_func = SSD_loss """
    def __init__(self,loss_func): 
        self.loss_func = loss_func
    def __call__(self,pred,target): 
        return self.loss_func.reg_loss

class SSD_ClasLoss(object):
    """Class to extract the SSD clas_Loss from the attribute of class SSD_loss. 
    Use as a metric in training to observe balance of SSD reg_loss and SSD clas_loss. 
    NOTE: Must have learner.loss_func = SSD_loss """
    def __init__(self,loss_func): 
        self.loss_func = loss_func
    def __call__(self,pred,target): 
        return self.loss_func.clas_loss    

# (6.3) Other Metrics   
class ComputeMaxOverlaps(object):
    """ Computes maximum jaccard overlap of each ground truth object in images with any anchor box.         
        Use as a metric when training to determine how well the anchor boxes are 'covering' the objects. 
        
        Output:
        Returns mean over images, of the mean over objects within an image, of maximum jaccard 
        overlap of that object with any anchor box. But it also stores all these maximum overlaps 
        in a list self.max_overlaps, if you want examine the distribution more fully. """ 
    
    def __init__(self):
        self.max_overlaps = []
        
    def __call__(self,activ,target):
        
        Objects, anchors, bs = target[0], activ[0], len(target[0])
        batch_max_overlaps = []
        
        for i in range(bs):
            objects = Objects[i][Objects[i]>=0].view(-1,4)
            if len(objects)==0: continue
            jac = jaccard(objects,anchors)
            max_overlaps = ARR(jac.max(dim=1)[0])
            self.max_overlaps += list(max_overlaps)
            batch_max_overlaps.append(max_overlaps.mean())
       
        if len(batch_max_overlaps)>0: 
            avg_max_overlap = np.array(batch_max_overlaps).mean()
        else: avg_max_overlap = 0.0
        return TEN(avg_max_overlap)
    
def mAP1(targs,preds,scores,thresh):
    """ Compute mean average precision of predictions for validation dataset,
        for a single jaccard threshold and object category.  
        
    Arguments: (Let N = num images in validation dataset, c = object category) 
    targs: Length N list, where ith entry is a list of ground truth bboxes for category c in image i.
    preds: Length N list, where ith entry is a list of predicted bboxes for category c in image i.
    scores: Length N list, where ith entry is a list of confidence scores for predicted bboxes 
            in image i being of category c. 
    thresh: jaccard threshold in range [0.5,1) for a predicted bounding box matching with a ground truth box. 
    (NOTE: All bboxes given in min-max form [xmin,ymin,xmax,ymax].)
    """
    
    # Construct the 0-1 list <IsCorrect> and the list <Scores>
    # ith entry of IsCorrect specifies whether ith predicted bbox of category c was a correct prediction
    # ith entry of Scores specifies confidence score for ith predicted bbox
    
    # NOTE: Can be at most 1 correct predicted bbox for a given ground truth bbox.
    #       We take predicted bbox that has greatest jaccard overlap with ground truth box. 
    
    N = len(targs)
    IsCorrect, Scores = [],[]
    
    for i in range(N):
        is_correct = [0]*len(preds[i])
        if (len(preds[i]) > 0) and (len(targs[i]) > 0):
            jac = jaccard(TEN(np.array(targs[i])),TEN(np.array(preds[i])))
            max_overlaps, max_idxs = jac.max(dim=1)
            for j,idx in enumerate(max_idxs):
                if max_overlaps[j] > thresh: 
                    is_correct[idx] = 1        
        IsCorrect += is_correct
        Scores += scores[i]
    
    # Sort both lists Scores and IsCorrect together according to the order of Scores
    SortedCombined = sorted(zip(Scores,IsCorrect),reverse=True)    
    IsCorrect = np.array([ic for s,ic in SortedCombined])
    
    # compute the raw recall and precision values using sorted version of IsCorrect
    L = len(IsCorrect)
    ntrue = sum(len(targs[i]) for i in range(N))
    running_total_true_pos = np.cumsum(IsCorrect)
    precision_vals = running_total_true_pos*np.array([1/n for n in range(1,L+1)])
    recall_vals = running_total_true_pos/ntrue
    
    # compute smoothed precision curve (with 1 precision value for each unique recall value)    
    precision_maxes = np.flip(np.maximum.accumulate(np.flip(precision_vals)))
    precision_smoothed = precision_maxes[IsCorrect.nonzero()[0]]     
    
    # return the integrated value of the smoothed precision curve
    # NOTE: By definition, precision is set to 0 for all recall values > max(recall_vals)
    return np.sum(precision_smoothed)/ntrue
            
def mAP(predictions,targets,categories,thresholds=COCO_thresholds):
    
    """ Compute mean average precision of predictions for validation dataset,
        averaged over object categories and a user specified range of jaccard thresholds. 
    
    Arguments: (Let N = num images in validation dataset)
    predictions:  A list of length N containing entries of form [pred_boxes,pred_classes,conf_scores].
                  Each entry corresponds to one validation image. 
                 * pred_boxes is list of predicted bboxes for the image in min-max form np.array([xmin,ymin,xmax,ymax]).
                 * pred_classes is corresponding list of predicted integer category labels.
                 * conf_scores is corresponding list of "confidence scores" that predicted box represents 
                   an item of predicted category. Each confidence score is a float in range [0,1].                   
    targets: For each validation image there is a list of labeled ground truth bounding boxes in standard
             form [(b1,c1),...,(bn,cn)]. targets is a list of such lists with length = N. 
    categories: dictionary of form {0:'cat',1:'tree',2:'boat'} mapping integer 
                category labels to category names.
    thresholds: list of jaccard thresholds in range [0.5,1) to use for determining if 
                predicted bbox matches ground truth bbox. 
    """ 
    
    N = len(predictions)
    C = len(categories)
    
    # seperate target bboxes, predicted bboxes, and confidence scores by category 
    targs = [[[] for i in range(N)] for j in range(C)]
    preds = [[[] for i in range(N)] for j in range(C)]
    scores = [[[] for i in range(N)] for j in range(C)]
    
    for i in range(N):        
        
        pred_boxes, pred_classes, conf_scores = predictions[i]
        for j in range(len(pred_boxes)):
            c = pred_classes[j]
            preds[c][i].append(pred_boxes[j])
            scores[c][i].append(conf_scores[j])            
        
        for b,c in targets[i]:
            targs[c][i].append(b)
            
    # compute map score for each (threshold, category) pair. 
    mAP_scores = np.zeros((len(thresholds),C))
    for c in range(C):
        for j,thresh in enumerate(thresholds):
            mAP_scores[j,c] = mAP1(targs[c],preds[c],scores[c],thresh)
            print('cat =',c,':',categories[c],' thresh =',thresh)
            print('cat-thresh mAP = ',mAP_scores[j,c])
            print('')
    
    print('Overall mAP = ', np.mean(mAP_scores))
    
    # return average of results
    return np.mean(mAP_scores)

    
# SECTION (7) - IMAGE LEARNER 

class ImageLearner(Learner):
    
    """Class for an Image Data Learner. Initialization and all attributes are same as 
       for parent class <Learner>. Also, all methods for class Learner are unmodified. 
       Only some extra methods added that are specific to image data."""
    
    def __init__(self, PATH, data, model, optimizer='default', loss_func='default', use_moving_avg=True):        
        super().__init__(PATH, data, model, optimizer, loss_func, use_moving_avg)
        
    def data_resize(self,sz,bs=None):
        
        """ For use with single_label/multi_label image classification data. 
            Modifies data.transforms to use the given value of <sz> for transforming images, 
            instead of previous value. Also, modifies the bs of dataloaders if <bs> is given. 
            This is useful because when image size is increased a batch may no longer fit if memory."""
        
        if type(sz) == int: sz = (sz,sz)
        self.data.sz = sz
        tfms = [self.data.train_ds.transform, self.data.val_ds.transform]
        if self.data.test_ds: tfms += [self.data.test_ds.transform]
        for tfm in tfms: tfm.sz = sz
        
        if bs:
            self.data.bs = bs
            nw = self.data.train_dl.num_workers
            self.data.train_dl = DataLoader(self.data.train_ds,batch_size=bs,num_workers=nw,shuffle=True)
            self.data.val_dl = DataLoader(self.data.val_ds,batch_size=bs,num_workers=nw,shuffle=False)
            if self.data.test_dl:
                self.data.test_dl=DataLoader(self.data.test_ds,batch_size=bs,num_workers=nw,shuffle=False)

    def switch_transform_stats(self,new_stats):
        
        """Modifies data transforms to use the given value of new_stats for transforming images
           instead of the previous value of stats. This is useful if you want to ensemble
           multiple models which use different transform stats, and want to check the ensembling 
           procedure on the same validation set."""
        
        tfms = [self.data.train_ds.transform, self.data.val_ds.transform]
        if self.data.test_ds: tfms += [self.data.test_ds.transform] 
        for tfm in tfms: tfm.stats = new_stats      
                   
    def confusion_matrix(self,pred_labels=None):
        
        """Function to plot the confusion matrix for validation data. 
           Works only with target_type = single_label. If <pred_labels> not given, 
           then self.predict('val') used to compute. """
        
        from sklearn.metrics import confusion_matrix
        true_labels = self.data.val_ds.y
        if pred_labels is None: pred_probs,pred_labels = self.predict('val')  
        cm = confusion_matrix(true_labels, pred_labels)
        classes = {self.data.categories[x]:x for x in self.data.categories}
        plot_confusion_matrix(cm, classes)
            
    def show_images(self,ds_type,classify_type=None,preds=None,
                    random=True,num_images=6,num_cols=3,figsize=(16,8)):
        
        """ 
        Function to show a collection of images from one of the datasets of a learner. 
        Images are not transformed in any way, but ground truth labels/bboxes are 
        shown for train and val datasets.
        
        Arguments:
        ds_type: 'train', 'val', or 'test'
        classify_type: 'correct','incorrect', or None. If None, shows any images. If 'correct' or 
                       'incorrect' shows only images which are correctly or incorrectly classified.
                       The 'correct' and 'incorrect' options work only for ds_type = 'val' and 
                       target_type = 'single_label'. 
        preds: preds = [pred_probs,pred_labels]. Use only for ds_type = 'val' and target_type = 'single_label'. 
                       If preds is None, self.predict('val') is used to compute predictions.
        random: If true selects images to display randomly, otherwise in order of the dataset. 
        num_images: number of images to display
        num_cols: number of columns of images
        figsize: the figure size
        """
        
        if ds_type in ['train','test'] and classify_type:
            raise ValueError("If ds_type is train or test, must have classify_type = None")
        if self.data.target_type != 'single_label' and classify_type:
            raise ValueError("If target_type is not 'single_label', must have classify_type = None")
        
        if ds_type == 'train': 
            images = self.data.train_ds.images
            IMG_PATH = self.data.train_ds.IMG_PATH
        elif ds_type == 'val': 
            images = self.data.val_ds.images
            IMG_PATH = self.data.val_ds.IMG_PATH
        elif ds_type == 'test': 
            images = self.data.test_ds.images
            IMG_PATH = self.data.test_ds.IMG_PATH
            
        true_labels = [image['target'] for image in images]
        idxs = list(range(len(images)))
        if ds_type == 'val' and classify_type:
            if preds: pred_probs,pred_labels = preds
            else: pred_probs,pred_labels = self.predict('val')
            correct_idxs,incorrect_idxs = [],[]
            for i in range(len(images)):
                if pred_labels[i] == true_labels[i]: correct_idxs.append(i)
                else: incorrect_idxs.append(i)
            if classify_type == 'correct':idxs = correct_idxs
            elif classify_type == 'incorrect':idxs = incorrect_idxs
        
        if random == True: 
            select_idxs = np.random.choice(idxs,num_images,replace = False)
        else: select_idxs = idxs[0:num_images]
        
        select_images = []
        for i in select_idxs:
            img = plt.imread(IMG_PATH + images[i]['img'])
            image = {'img':img}
            if self.data.target_type == 'single_label' and ds_type in ['train','val']:
                image['label'] = int(true_labels[i])
            elif self.data.target_type == 'multi_label' and ds_type in ['train','val']:
                image['label'] = list(np.where(true_labels[i] == 1)[0])
            elif self.data.target_type == 'bbox' and ds_type in ['train','val']:
                image['bboxes'] = true_labels[i]
            if self.data.target_type == 'single_label' and classify_type: 
                image['preds'] = pred_probs[i]   
            select_images.append(image)
            
        ShowImages(select_images,self.data.categories,num_cols,figsize)     
                        
    def show_bbox_preds(self, ds_type, predictions=None, thresh=0.05, max_overlap=0.5, rel_thresh=None, 
                        top_k=1000, max_boxes=20, dup=None, inc=None, random=True, num_images=6, num_cols=2):
        
        """Function to show predicted bboxes for a collection of images from 'val' or 'test' dataset.
           
        Arguments:
        ds_type: 'val' or 'test'
        num_images: number of images to display
        num_cols: number of columns of images
        random: If true selects images randomly, otherwise in order of the dataset.
        predictions: predictions for validation/test set in format of learner.predict() method, or None.
        other arguments: parameters for self.model.BBoxPredictor, to use if predictions is None.
        """
        
        transform = TransformBBoxShowPreds()
        
        if ds_type == 'val': 
            images = self.data.val_ds.images
            IMG_PATH = self.data.val_ds.IMG_PATH
        elif ds_type == 'test': 
            images = self.data.test_ds.images
            IMG_PATH = self.data.test_ds.IMG_PATH          
            
        # select idxs of images from dataset
        idxs = list(range(len(images)))
        if random == True: 
            select_idxs = np.random.choice(idxs,num_images,replace = False)
        else: select_idxs = idxs[0:num_images]
        
        # show select images
        self.model.eval()
        IMGS,BBOXES,SCORES = [],[],[]
        for i in PBar(select_idxs):
            img = plt.imread(IMG_PATH + images[i]['img']).astype(np.float32)/255            
            if predictions:
                predboxes, predclasses, confscores = predictions[i]
            else:  
                scale = images[i]['scale']
                img2 = transform(copy.deepcopy(img),scale)   
                img_batch = TEN(img2.transpose(2,0,1)).unsqueeze(0)
                anchors,reg,clas = self.model(img_batch)
                pred = self.model.BBoxPredictor(img_batch, reg, clas, anchors, thresh, max_overlap, 
                                                rel_thresh, top_k, max_boxes, dup, inc)
                predboxes, predclasses, confscores = pred[0][0], pred[1][0], pred[2][0]
                predboxes = list_mult(predboxes,1/scale) # correct predictions for scaling of images in training
            bboxes = [(b,c) for b,c in zip(predboxes,predclasses)]
            IMGS.append(img)
            BBOXES.append(bboxes)
            SCORES.append(confscores)
        
        num_rows = int(np.ceil(num_images/num_cols))
        figsize = (12*num_cols,8*num_rows)
        images = [{'img':IMGS[i],'bboxes':BBOXES[i],'preds':SCORES[i]} for i in range(num_images)]
        ShowImages(images,self.data.categories,num_cols,figsize)        
        
    def TTA(self,ds_type,beta=0.4):
        
        """
        Performs test time augmentation for either val or test dataset.
        Use only with target_type = 'single_label' or 'multi_label'.
           
        Predictions are made for 1 regular tfm_eval of each image, and also 4 random 
        tfm_aug kind of transforms. However, these 4 transforms have less random rotation 
        than standard tfm_aug and no zooming. Also, rather than being randomly cropped the 4 
        are cropped at evenly spaced points from left to right (if width > height) 
        or top to bottom (if height > width) in order to cover whole image. The 5 sets of 
        predictions are then combined using a weighted average. 
           
        Arguments:
        ds_type: Either 'val' or 'test'
        beta: weight beta in (0,1) is given to tfm_eval prediction and weight (1-beta)/4 is 
              given to all others when combining the predictions. 
           
        Output: 
        Returns pred_probs, pred_labels. Same form as output of combine_preds() function.
        """
        
        if ds_type == 'val': 
            IMG_PATH = self.data.val_ds.IMG_PATH
            images = self.data.val_ds.images
            tfm = self.data.train_ds.transform
        elif ds_type == 'test': 
            IMG_PATH = self.data.test_ds.IMG_PATH
            images = self.data.test_ds.images
            tfm = self.data.train_ds.transform
                    
        tfm_type, stats, sz = tfm.tfm_type, tfm.stats, tfm.sz
        num_workers = self.data.train_dl.num_workers
        bs = self.data.bs
        
        tfm0 = Transform('Basic','center',None,sz,None,None,None,None,stats=stats)
        tfm1 = Transform(tfm_type,0.0,None,sz,5,1.0,stats=stats)
        tfm2 = Transform(tfm_type,0.33,None,sz,5,1.0,stats=stats)
        tfm3 = Transform(tfm_type,0.67,None,sz,5,1.0,stats=stats)
        tfm4 = Transform(tfm_type,1.0,None,sz,5,1.0,stats=stats)
        tfms = [tfm0,tfm1,tfm2,tfm3,tfm4]
        
        preds_list = []
        for i in PBarTTA(range(5)):
            tfm = tfms[i]
            ds = ImageDataset(IMG_PATH,images,tfm,self.target_type,ds_type)
            dl = DataLoader(ds,batch_size=bs,num_workers=num_workers)
            preds = self.predict(dl)[0]
            preds_list.append(preds)
        
        weights = [beta, (1-beta)/4, (1-beta)/4, (1-beta)/4, (1-beta)/4]
        return combine_preds(preds_list,self.target_type,weights)
    
    def TTA_bbox(self, ds_type, transforms, thresh=0.05, max_overlap=0.5, 
                 rel_thresh=None, top_k=1000, max_boxes=20, dup=None, inc=None):
        
        """
        Performs test time augmentation for bounding box data learner, 
        for either val or test dataset.
           
        Predictions are made for 1 regular tfm_eval of each image, and also 4 random 
        tfm_aug transforms. These predictions are then concatenated and nms function
        is applied to the concatenated set of 5 predictions.
           
        Arguments:
        ds_type: Either 'val' or 'test'
        transforms: [tfm_eval,tfm_aug]
        other arguments: parameters for self.model.BBoxPredictor
           
        Output: 
        Same exact format as output of self.predict() method (with target_type = 'bbox').           
        """
        
        self.model.eval()
        tfm_eval, tfm_aug = transforms 
        
        if ds_type == 'val': 
            IMG_PATH = self.data.val_ds.IMG_PATH
            images = self.data.val_ds.images
        elif ds_type == 'test': 
            IMG_PATH = self.data.test_ds.IMG_PATH
            images = self.data.test_ds.images
        
        # Make Predictions (1 time with tfm_eval, 4 times with tfm_aug)
        PredBoxes, PredClasses, ConfScores = [],[],[]
        for i in PBarTTA(range(5)):
            if i==0: transform = copy.deepcopy(tfm_eval)
            else: transform = copy.deepcopy(tfm_aug)
            transform.get_values()
            ds = ImageDataset(IMG_PATH,images,transform,'bbox','test')
            dl = DataLoader(ds,batch_size=1,collate_fn=AspectRatioCollater,num_workers=1,shuffle=False)
            
            for j,(x_batch,y_batch) in enumerate(PBarPredict(dl)): 
                
                # make prediction on transformed image (batches x_batch have 1 image each)
                x_batch = to_cuda(x_batch)
                anchors, reg, clas = self.predict1minibatch(x_batch)
                pred = self.model.BBoxPredictor(x_batch, reg, clas, anchors, thresh, max_overlap, 
                                                rel_thresh, top_k, max_boxes, dup, inc)
                boxes, classes, scores = pred[0][0], pred[1][0], pred[2][0]
                
                # 'undo transform' to determine locations of predicted boxes relative to original image
                img = plt.imread(IMG_PATH + images[j]['img'])
                rows, cols, _ = img.shape
                scale = images[j]['scale']
                row_jit, col_jit = transform.row_jitter_values[j], transform.col_jitter_values[j]
                rand_scale, flip = transform.scale_values[j], transform.flip_values[j]
                
                if len(boxes) > 0:
                    boxes = np.array(boxes)
                    boxes = np.array([boxes[:,0]-col_jit, boxes[:,1]-row_jit, boxes[:,2]-col_jit, boxes[:,3]-row_jit]).T
                    boxes = (1/(rand_scale*scale))*boxes
                    if i>0 and flip==1:
                        boxes = np.array([ cols-boxes[:,2], boxes[:,1], cols-boxes[:,0], boxes[:,3] ]).T
                    boxes = list(boxes)
                
                # add predicted boxes,classes,scores to the lists
                PredBoxes.append(boxes)
                PredClasses.append(classes)
                ConfScores.append(scores)
            
        # Combine Predictions from these 5 passes through dataset
        AllPreds, L = [], len(ds)
        for l in range(L):
            boxes, classes, scores = PredBoxes[l], PredClasses[l], ConfScores[l]
            for j in [l+L, l+2*L, l+3*L, l+4*L]:
                boxes += PredBoxes[j]
                classes += PredClasses[j]
                scores += ConfScores[j]
            AllPreds.append([TEN(boxes),TEN(classes),TEN(scores)])
        
        #Apply nms function to remaining averaged predictions
        for l in range(L):
            pred_boxes, pred_classes, conf_scores = AllPreds[l]
            pred_boxes, pred_classes, conf_scores = \
            vmods.retinanet.nms(pred_boxes, pred_classes, conf_scores, max_overlap, rel_thresh, top_k, max_boxes, dup, inc)
            AllPreds[l] = [pred_boxes, pred_classes, conf_scores]
                                 
        return AllPreds       
           
    def compute_mAP(self, predictions=None, thresh=0.05, max_overlap=0.5, rel_thresh=None, top_k=1000, 
                    max_boxes=20, dup=None, inc=None, mAP_thresholds=COCO_thresholds):
        
        """Compute mAP for validations dataset. Use only with target_type = 'bbox'.
        
        Arguments:
        mAP_thresholds: list of overlap thresholds to use for computing mAP score. 
        predictions: predictions for validation set in format of learner.predict() method, or None.
        other arguments: parameters to make predictions with learner.predict() if predictions is None.  
        """
        
        categories = self.data.categories
        targets = self.data.val_ds.y 
        if predictions is None:
            predictions = self.predict('val', True, thresh, max_overlap, rel_thresh, top_k, max_boxes, dup, inc)        
        map_score = mAP(predictions,targets,categories,mAP_thresholds)
        del [predictions,targets]
        return map_score
    
    def coco_pascal_eval(self, val_json, predictions=None, thresh=0.05, max_overlap=0.5, rel_thresh=None, 
                         top_k=1000, max_boxes=20, dup=None, inc=None):
        
        """ Function to evaluate bbox predictions for COCO or Pascal dataset.
            In either case, uses COCO Eval Code from pycocotools. This works for 
            Pascal as well, because annotation files are in same format for both datasets. 
            
        Arguments:
        val_json: full path to val_json file
        predictions: predictions for validation set in format of learner.predict() method, or None.
        other arguments: parameters to make predictions with learner.predict() if predictions is None.
        """

        if predictions is None:
            predictions = self.predict('val', True, thresh, max_overlap, rel_thresh, top_k, max_boxes, dup, inc)
        
        preds, image_ids = [],[]
        for i in range(len(predictions)):
            ID = self.data.val_ds.images[i]['id']
            image_ids.append(ID)
            pred_boxes, pred_classes, conf_scores = predictions[i]
            for box,cat,score in zip(pred_boxes,pred_classes,conf_scores):
                score = float(score)
                cat = self.data.cat2dscat[cat]
                box = [float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])]     # [xmin,ymin,xmax,ymax] ->
                preds.append({'image_id':ID, 'category_id':cat, 'score':score, 'bbox': box}) # [xmin,ymin,w,h]
        
        json.dump(preds, open(self.PATH + 'preds.json','w'), indent=4)      
        coco_true = COCO(val_json)
        coco_pred = coco_true.loadRes(self.PATH + 'preds.json')
        
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate() 
        coco_eval.accumulate()
        coco_eval.summarize()
        