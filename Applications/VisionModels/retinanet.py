"""
I have modified this file from the retinanet implementation at 
https://github.com/yhenon/pytorch-retinanet. 
 
The BBoxPredictor section is substantially new. 
Other parts have only small modifications from original source.  
"""

from General.Core import TEN, ARR, list_del, joint_sort
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo 
import pathlib

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

#################### ResNet MODEL COMPONENTS #######################

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#################### RetinaNet ADDITIONAL MODEL COMPONENTS ####################

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        
        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):

        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]

class RegressionModel(nn.Module):
      
    """ 
    Let A = num_anchors for single grid cell, C = num_channels for input. 
    Input Shape: bs x C x H x W
    Output Shape: bs x (H*W*A) x 4
        
    Ordering for each element of bs (considered as an H*W*A x 4 Tensor) is as follows: 
    * H blocks each of size W*A concatenated vertically
    * Each block is W sub-blocks concatenated vertically 
    * Each sub-block has A rows of length 4 (for 4 coordinates of an anchor box)         
    """
    
    def __init__(self, num_features_in, num_anchors=9, feature_size=256, bn=False, drop=None):
        super(RegressionModel, self).__init__()
        self.drop0 = nn.Dropout(drop[0]) if drop else None
        self.drop = nn.Dropout(drop[1]) if drop else None
        self.bn0 = nn.BatchNorm2d(num_features_in, momentum=0.01) if bn else None
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(feature_size, momentum=0.01) if bn else None

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(feature_size, momentum=0.01) if bn else None

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(feature_size, momentum=0.01) if bn else None

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(feature_size, momentum=0.01) if bn else None

        self.output = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):
        
        if self.bn0: x = self.bn0(x)
        if self.drop0: x = self.drop0(x)

        out = self.conv1(x)
        out = self.act1(out)
        if self.bn1: out = self.bn1(out)
        if self.drop: out = self.drop(out)

        out = self.conv2(out)
        out = self.act2(out)
        if self.bn2: out = self.bn2(out)
        if self.drop: out = self.drop(out)

        out = self.conv3(out)
        out = self.act3(out)
        if self.bn3: out = self.bn3(out)
        if self.drop: out = self.drop(out)

        out = self.conv4(out)
        out = self.act4(out)
        if self.bn4: out = self.bn4(out)
        if self.drop: out = self.drop(out)

        out = self.output(out)

        # out is bs x C x H x W, with C = num_channels = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
      
        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    
    """ 
    Let A = num_anchors for single grid cell, let K = number of classes, let C = num_channels for input. 
    Input Shape: bs x C x H x W
    Output Shape: bs x (H*W*A) x K
        
    Ordering for each element of bs (considered as an H*W*A x 4 Tensor) is as follows: 
    * H blocks each of size W*A concatenated vertically
    * Each block is W sub-blocks concatenated vertically 
    * Each sub-block has A rows of length K (for K possible classes)         
    """
    
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, feature_size=256, bn=False, drop=None):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.drop0 = nn.Dropout(drop[0]) if drop else None
        self.drop = nn.Dropout(drop[1]) if drop else None
        self.bn0 = nn.BatchNorm2d(num_features_in, momentum=0.01) if bn else None
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(feature_size, momentum=0.01) if bn else None

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(feature_size, momentum=0.01) if bn else None

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(feature_size, momentum=0.01) if bn else None

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(feature_size, momentum=0.01) if bn else None

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        
        if self.bn0: x = self.bn0(x)
        if self.drop0: x = self.drop0(x)
            
        out = self.conv1(x)
        out = self.act1(out)
        if self.bn1: out = self.bn1(out)
        if self.drop: out = self.drop(out)

        out = self.conv2(out)
        out = self.act2(out)
        if self.bn2: out = self.bn2(out)
        if self.drop: out = self.drop(out)

        out = self.conv3(out)
        out = self.act3(out)
        if self.bn3: out = self.bn3(out)
        if self.drop: out = self.drop(out)

        out = self.conv4(out)
        out = self.act4(out)
        if self.bn4: out = self.bn4(out)
        if self.drop: out = self.drop(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is bs x C x H x W, with C = num_channels = n_classes * n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, height, width, channels = out1.shape

        out2 = out1.contiguous().view(batch_size, height, width, self.num_anchors, self.num_classes)
        
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes) 
    
##################### RetinaNet MODEL CLASS ########################### 
   
class RetinaNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(RetinaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2]-1].conv2.out_channels,
                         self.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels,
                         self.layer4[layers[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.regressionModel = RegressionModel(256)        
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        
        self.AnchorGenerator = AnchorGenerator()
        self.BBoxPredictor = BBoxPredictor()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2/n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        
        prior = 0.01 
        nn.init.constant_(self.classificationModel.output.weight, 0)
        nn.init.constant_(self.classificationModel.output.bias, -np.log((1.0-prior)/prior))
        nn.init.constant_(self.regressionModel.output.weight, 0)
        nn.init.constant_(self.regressionModel.output.bias, 0)     
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, img_batch):
        
        """ Ouput: A list [anchors,reg,clas]
        anchors: (N x 4) Tensor of anchor boxes associated with images in img_batch.
        reg: (bs x N x 4) Tensor of bbox regression activations for anchor boxes 
              of each img in img_batch. Default locations of anchor boxes are shifted to predicted 
              locations using these reg activations according to BBoxPredictor method. 
        clas: (bs x N x num_classes) Tensor of classification activations for anchor boxes of
               each img in img_batch. clas[i,j,k] = Prob(box j in img i is of class k). 
            
        (Here N = number of anchor boxes for each img in img_batch, depends on img_batch dimensions.)
        """
        
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])
        reg = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        clas = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        anchors = self.AnchorGenerator(img_batch)
        
        return [anchors,reg,clas]


##################### Methods to Call Specific Instances of 'RetinaNet' class ####################            
def retinanet18(num_classes, pretrained=False, **kwargs):
    """Constructs a RetinaNet model with resnet18 backbone. 
       Backbone pretrained on imagenet if pretrained==True."""
    model = RetinaNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model

def retinanet34(num_classes, pretrained=False, **kwargs):
    """Constructs a RetinaNet model with resnet34 backbone. 
       Backbone pretrained on imagenet if pretrained==True."""
    model = RetinaNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model

def retinanet50(num_classes, pretrained=False, **kwargs):
    """Constructs a RetinaNet model with resnet50 backbone. 
       Backbone pretrained on imagenet if pretrained==True."""
    model = RetinaNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model

def retinanet101(num_classes, pretrained=False, **kwargs):
    """Constructs a RetinaNet model with resnet101 backbone. 
       Backbone pretrained on imagenet if pretrained==True."""
    model = RetinaNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model

def retinanet152(num_classes, pretrained=False, **kwargs):
    """Constructs a RetinaNet model with resnet152 backbone. 
       Backbone pretrained on imagenet if pretrained==True."""
    model = RetinaNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model

def retinanet():
    "Constructs a RetinaNet model with a resnet50 backbone. Model is pretrained on COCO dataset."
    model_path = pathlib.Path('../Applications/VisionModels/RetinanetPretrainedCOCO.pt')
    model = RetinaNet(80, Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(torch.load(model_path))
    return model

################# AnchorGenerator Class and Associated Functions ####################

def get_anchor_set(ratios= [0.5, 1, 2], scales=[2**0, 2**(1/3), 2**(2/3)]):
    """Generate base set of anchors relative to the unit square centered at (0,0).
       Returns as (n x 4) np.array, where each row is a box in form [xmin,ymin,xmax,ymax]. """
    
    # Generate 1d vectors for heights and widths of anchors.
    # Area of each anchor box is scale (in scales), width/height is ratio (in ratios). 
    Scales = np.tile(scales,len(ratios))
    Ratios = np.repeat(ratios,len(scales))  
    Heights = Scales/np.sqrt(Ratios)
    Widths = Scales*np.sqrt(Ratios)

    #Return set of anchors in form (xmin, ymin, xmax, ymax) centered at (0,0).
    return np.array([-Widths/2, -Heights/2, Widths/2, Heights/2]).T
       
def get_anchor_shifts(shape, stride, anchors):
    """Shift a given collection of anchors centered on (0,0) to create larger collection 
       of anchors centered on each grid cell in a grid of 'shape' = (H,W) cells. 
       Each cell is of size ('stride' x 'stride') pixels."""
    
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).T

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shifted anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A,K = anchors.shape[0], shifts.shape[0]
    shifted_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    shifted_anchors = shifted_anchors.reshape((K * A, 4))

    return shifted_anchors
    
class AnchorGenerator(object):
    """ Class to generate a collection of bboxes for each input img_batch (depending on img_batch dimensions).
        This collection of boxes matches the reg and clas outputs of the RetinaNet model for the img_batch. """
    
    def __init__(self, ratios=[0.5, 1, 2], scales=[2**0, 2**(1/3), 2**(2/3)]):
        self.pyramid_levels = [3, 4, 5, 6, 7]
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        self.ratios = np.array(ratios)
        self.scales = np.array(scales)
        self.anchor_set = get_anchor_set(ratios,scales)
    
    def __call__(self,img_batch):
        
        img_shape = np.array(img_batch.shape[2:])
        grid_shapes = [(img_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        
        all_anchors = []
        for i,p in enumerate(self.pyramid_levels): 
            shifted_anchors = get_anchor_shifts(grid_shapes[i], self.strides[i], self.sizes[i]*self.anchor_set)
            all_anchors.append(shifted_anchors)
        
        return TEN(np.concatenate(all_anchors))

    
########## BBoxPredictor Class and Associated Functions ######################## 

def intersections(Boxes1,Boxes2): 
    
    """ Compute area of intersection of each pair of boxes (b1,b2) with b1 in Boxes1 and b2 in Boxes 2. 
        Boxes1 = (n x 4) np.array, Boxes2 = (m x 4) np.array, function returns (n x m) np.array. 
        All boxes given in min-max form [xmin,ymin,xmax,ymax]. """
       
    B1,B2 = np.expand_dims(Boxes1,axis=1),np.expand_dims(Boxes2,axis=0)
    inter_w = (np.minimum(B1[:,:,2], B2[:,:,2]) - np.maximum(B1[:,:,0], B2[:,:,0])).clip(0,None)
    inter_h = (np.minimum(B1[:,:,3], B2[:,:,3]) - np.maximum(B1[:,:,1], B2[:,:,1])).clip(0,None)
    return inter_w * inter_h

def jaccard(Boxes1,Boxes2):
    
    """ Compute jaccard index of each pair of boxes (b1,b2) with b1 in Boxes1 and b2 in Boxes 2. 
        Boxes1 = (n x 4) np.array, Boxes2 = (m x 4) np.array, function returns (n x m) np.array. 
        All boxes given in min-max form [xmin,ymin,xmax,ymax]. """
            
    areas1 = (Boxes1[:,2] - Boxes1[:,0])*(Boxes1[:,3] - Boxes1[:,1]) 
    areas2 = (Boxes2[:,2] - Boxes2[:,0])*(Boxes2[:,3] - Boxes2[:,1])
    areas_inter = intersections(Boxes1,Boxes2)
    areas_union = np.expand_dims(areas1,axis=1) + np.expand_dims(areas2,axis=0) - areas_inter
    return areas_inter/areas_union

def nms(pred_boxes, pred_classes, conf_scores, max_overlap=0.5, rel_thresh=None, 
        top_k=1000, max_boxes=20, dup=None, inc=None, print_it=False):
    
    """ Function to perform non-maximum supression on a collection of predicted bboxes 
        for single image. Also applies some other box pruning techniques, as described in arguments. 
        
    Arguments: (N = num predicted boxes given as input)  
    pred_boxes: Nx4 Tensor of predicted bboxes in min-max form [xmin,ymin,xmax,ymax]
    pred_classes: Length N 1d Tensor of predicted classes. 
    conf_scores: Length N 1d Tensor of confidence scores, that given predicted box is of predicted class.
    top_k: A maximum of top_k bboxes from the collection pred_boxes are used for non-max supression.  
           These bboxes are chosen as the ones with the highest confidence scores.
    max_overlap: 2 predicted bboxes of same class are considered the same in non-maximal-supression step 
                 if they have a jaccard score > max_overlap.
    rel_thresh: rel_thresh = [rel_thresh1, rel_thresh2] or None. 
                If rel_thresh is not None, then after non-max-supression following steps are applied in order:
                1. Remove any predicted box with conf_score < x*rel_thresh1. Here x = max conf_score of all predicted boxes.
                2. For each class c, remove any predicted box with (predicted class = c) and (conf_score < x_c*rel_thresh2).
                   Here x_c = max confidence score of all predicted boxes with predicted class = c.    
    inc: inc = [inc_thresh,inc_classes] or None. 
               If not None, then after relative threshold step, removes predicted boxes 
               contained in or containing single predicted box of same class, as in following example.                
               Example: Assume inc_thresh = 0.9 and class c is NOT in inc_classes. Further assume that:
                        (1) b_i,b_j are 2 boxes of predicted class c.
                        (2) b_j is 90% contained in b_i, and no other box b_k is 90% contained in b_i.
                        (3) Area(b_j) > 0.25*Area(b_i). 
                        Then if conf_score(b_i) < 0.75*conf_score(b_j) we remove b_i,
                        and if conf_score(b_j) < 0.75*conf_score(b_i) we remove b_j. 
    dup: dup = [dup_thresh,dup_pairs] or None.
               If not None, then after the inc step duplicate predictions for different classes in 'same'
               location are removed as follows. If predicted boxes b_i,b_j have predicted classes c_i,c_j ,
               and (c_i,c_j) in dup_pairs, and jaccard(b_i,b_j) > dup_thresh, then:
               If conf_score(b_i) < 0.75*conf_score(b_j) we remove b_i,
               If conf_score(b_j) < 0.75*conf_score(b_i) we remove b_j. 
    max_boxes: Maximum number of boxes to return as output. Remove extra boxes, if necessary, after dup step.
    print_it: If True, prints number of boxes remaining after each step (use for diagnostic purposes).
    
    Output: Returns pred_boxes, pred_classes, conf_scores.
    pred_boxes: list of bboxes in min-max form np.array([xmin,ymin,xmax,ymax]). 
    pred_classes: corresponding list of predicted classes.
    conf_scores: corresponding list of confidence scores (in range[0,1]) 
                 that a predicted box is of the predicted class. 
    
    Lists are sorted such that conf_scores is in descending order, and 
    pred_boxes and pred_classes are in corresponding order. 
    """
    
    if len(pred_boxes)==0: return [],[],[]
    
    #sort by conf_scores and keep top_k
    conf_scores, ordering = conf_scores.sort(descending=True)
    conf_scores = conf_scores[:top_k]
    pred_classes = pred_classes[ordering][:top_k]
    pred_boxes = pred_boxes[ordering][:top_k]
    
    if print_it == True:
        print('after top_k')
        print(len(pred_boxes),len(pred_classes),len(conf_scores))
   
    # convert to lists
    conf_scores = list(ARR(conf_scores.detach())) 
    pred_classes = list(ARR(pred_classes.detach()))
    pred_boxes = list(ARR(pred_boxes.detach()))
    
    # perform non-max supression
    conf_scores2, pred_classes2, pred_boxes2 = [],[],[]
    
    while len(conf_scores) > 0:
        jac = jaccard( np.array(pred_boxes)[:1], np.array(pred_boxes) )[0]
        big_overlaps = (jac > max_overlap).astype(int)
        matching_classes = (np.array(pred_classes)==pred_classes[0]).astype(int)
        del_idxs = list(np.where(big_overlaps*matching_classes == 1)[0])
        
        conf_scores2.append(conf_scores[0])
        pred_classes2.append(pred_classes[0])
        pred_boxes2.append(pred_boxes[0])
        
        conf_scores = list_del(conf_scores,del_idxs)
        pred_classes = list_del(pred_classes,del_idxs)
        pred_boxes = list_del(pred_boxes,del_idxs)
    
    conf_scores = conf_scores2
    pred_classes = pred_classes2
    pred_boxes = pred_boxes2
    
    if print_it == True:
        print('after non-max-supress')
        print(len(pred_boxes),len(pred_classes),len(conf_scores))    
    
    # relative threshold 
    if rel_thresh:         
        rel_thresh1, rel_thresh2 = rel_thresh
        
        # general relative thresholding
        for i in range(len(conf_scores)):
            if conf_scores[i] < rel_thresh1*conf_scores[0]:
                conf_scores = conf_scores[:i]
                pred_classes = pred_classes[:i]
                pred_boxes = pred_boxes[:i]
                break 
        
        # by class relative thresholding
        L, del_idxs = len(conf_scores), []
        for i in range(L-1):
            for j in range(i+1,L): 
                if (pred_classes[i] == pred_classes[j]) and (conf_scores[j] < rel_thresh2*conf_scores[i]):
                    del_idxs.append(j)
        
        if len(del_idxs) > 0:
            conf_scores = list_del(conf_scores,del_idxs)
            pred_classes = list_del(pred_classes,del_idxs)
            pred_boxes = list_del(pred_boxes,del_idxs)
    
    if print_it == True:
        print('after relative threshold')
        print(len(pred_boxes),len(pred_classes),len(conf_scores))
    
    # Filter single inclusions of same class      
    if inc:
        inc_thresh, inc_classes = inc 
        L = len(pred_classes)
        pc,pb = np.array(pred_classes), np.array(pred_boxes)
        EqualClasses = (np.expand_dims(pc,1) == np.expand_dims(pc,0)).astype(int)   
        IntersectAreas = intersections(pb,pb)
        BoxAreas = (pb[:,2] - pb[:,0])*(pb[:,3] - pb[:,1])
        Ratios = IntersectAreas/BoxAreas
        RatiosEq = Ratios*EqualClasses
        Ratios2 = np.expand_dims(BoxAreas,0)/np.expand_dims(BoxAreas,1)
        Inclusions = (RatiosEq > inc_thresh).astype(int) - np.identity(L,int)
        InclusionsBig = Inclusions * (Ratios2 > 0.25).astype(int)        
        single_inc_idxs = list((InclusionsBig.sum(axis=1) == 1).nonzero()[0])        
        
        del_idxs = []
        for i,idx in enumerate(single_inc_idxs):
            if int(pred_classes[idx]) in inc_classes: del_idxs.append(i)             
        single_inc_idxs = list_del(single_inc_idxs,del_idxs)                
        single_inc_idxs2 = [np.argmax(InclusionsBig[i]) for i in single_inc_idxs]
        single_inc_idxs = list(set(single_inc_idxs) - set(single_inc_idxs2))
        
        del_idxs = []
        for i in single_inc_idxs:
            j = np.argmax(InclusionsBig[i])
            if conf_scores[i] < 0.75*conf_scores[j]: del_idxs.append(i)
            elif conf_scores[j] < 0.75*conf_scores[i]: del_idxs.append(j)       
        
        if len(del_idxs) > 0:
            conf_scores = list_del(conf_scores,del_idxs)
            pred_classes = list_del(pred_classes,del_idxs)
            pred_boxes = list_del(pred_boxes,del_idxs) 
    
    if print_it == True:
        print('after filtering sinle inclusions')
        print(len(pred_boxes),len(pred_classes),len(conf_scores))    
    
    # Filter duplicate box predictions of different classes
    if dup: 
        dup_thresh, dup_pairs = dup
        stop = False
        while stop == False:
            stop = True
            jac = jaccard(np.array(pred_boxes),np.array(pred_boxes))
            L, del_idx = len(pred_boxes), -1
            for i in range(L-1):
                if del_idx >= 0: break   
                for j in range(i+1,L):
                    if (jac[i,j] > dup_thresh) and ((pred_classes[i],pred_classes[j]) in dup_pairs) \
                    and (conf_scores[j] < 0.75*conf_scores[i]):
                        del_idx = j
                        conf_scores = conf_scores[:del_idx] + conf_scores[del_idx+1:]
                        pred_classes = pred_classes[:del_idx] + pred_classes[del_idx+1:]
                        pred_boxes = pred_boxes[:del_idx] + pred_boxes[del_idx+1:]
                        stop = False
                        break    
    
    if print_it == True:
        print('after filtering duplicate predictions of different classes')
        print(len(pred_boxes),len(pred_classes),len(conf_scores))
    
    # restrict to at most max_boxes bounding boxes
    conf_scores = conf_scores[:max_boxes]
    pred_classes = pred_classes[:max_boxes]
    pred_boxes = pred_boxes[:max_boxes]  
    
    if print_it == True:
        print('after restrict to max_boxes')
        print(len(pred_boxes),len(pred_classes),len(conf_scores))
        print('')
    
    return pred_boxes, pred_classes, conf_scores
    
class BBoxPredictor(object):
    
    """ This class uses activations from a RetinaNet model to compute predicted locations 
        of bounding boxes. Thresholding and non-maximal supression are applied to prune
        the returned collection of predicted boxes. 
    
    Arguments for Initialization:
    mean, std: Let [reg0,reg1,reg2,reg3] be regression activations for a given anchor box. 
               Default anchor box locations [cent_x,cent_y,w,h] are shifted according to following rules:
               cent_x -> cent_x + w*dx,  where dx = reg0*std[0] + mean[0]
               cent_y -> cent_y + h*dy,  where dy = reg1*std[1] + mean[1]
               w -> w*exp(dw),           where dw = reg2*std[2] + mean[2]
               h -> h*exp(dh),           where dh = reg3*std[3] + mean[3]
    """

    def __init__(self, mean=[0., 0., 0., 0.], std=[0.1, 0.1, 0.2, 0.2]):
        super().__init__()
        self.mean, self.std = TEN(mean), TEN(std)

    def __call__(self, img_batch, reg, clas, anchors, thresh=0.05, max_overlap=0.5, 
                 rel_thresh=None, top_k=1000, max_boxes=20, dup=None, inc=None):
        
        """ 
        Arguments:
        img_batch: batch of images 
        anchors, reg, clas: output of RetinaNet model with input img_batch.
        thresh: predicted boxes with a confidence score < thresh are removed (prior to nms).
        max_overlap, rel_thresh, top_k, max_boxes, dup, inc: parameters for nms function.        
        
        Output: Returns PredBoxes, PredClasses, ConfScores (each is a list of length bs).    
        * PredBoxes[j] = [b_{j,1},...,b_{j,n_j}] where b_{j,i} is predicted location
          of ith box for image j in min-max form, b_{j,i} = np.array([xmin,ymin,xmax,ymax]). 
        * PredClasses[j] = [c_{j,1},...,c_{j,n_j}] is associated list of predicted classes.       
        * ConfScores[j] = [s_{j,1},....,s_{j,n_j}] is associated list of confidence scores. 
        """   
        
        bs, num_channels, height, width = img_batch.shape                
        W = anchors[:, 2] - anchors[:, 0]
        H = anchors[:, 3] - anchors[:, 1]
        Cx = anchors[:, 0] + 0.5*W
        Cy = anchors[:, 1] + 0.5*H 
        PredBoxes, PredClasses, ConfScores = [],[],[]

        for i in range(len(img_batch)):
            
            # threshold
            conf_scores, pred_classes = clas[i].max(dim=1)
            thresh_idxs = (conf_scores > thresh).nonzero().view(-1)
            if len(thresh_idxs) == 0: 
                PredBoxes.append([])
                PredClasses.append([])
                ConfScores.append([])
                continue
            conf_scores = conf_scores[thresh_idxs]
            pred_classes = pred_classes[thresh_idxs] 
            reg_activ = reg[i][thresh_idxs] 
            w, h, cx, cy = W[thresh_idxs], H[thresh_idxs], Cx[thresh_idxs], Cy[thresh_idxs]
            
            # compute predicted box locations (of above threshold boxes) using reg activations
            dx = reg_activ[:, 0] * self.std[0] + self.mean[0]
            dy = reg_activ[:, 1] * self.std[1] + self.mean[1]
            dw = reg_activ[:, 2] * self.std[2] + self.mean[2]
            dh = reg_activ[:, 3] * self.std[3] + self.mean[3]

            pred_cx = cx + w*dx
            pred_cy = cy + h*dy
            pred_w = w * torch.exp(dw)
            pred_h = h * torch.exp(dh)

            pred_xmin = pred_cx - 0.5*pred_w
            pred_ymin = pred_cy - 0.5*pred_h       
            pred_xmax = pred_cx + 0.5*pred_w
            pred_ymax = pred_cy + 0.5*pred_h
            
            pred_boxes = torch.stack([pred_xmin, pred_ymin, pred_xmax, pred_ymax], dim=1)
            
            # clip predicted box locations to img size 
            pred_boxes[:, 0] = torch.clamp(pred_boxes[:, 0], min=0)
            pred_boxes[:, 1] = torch.clamp(pred_boxes[:, 1], min=0)
            pred_boxes[:, 2] = torch.clamp(pred_boxes[:, 2], max=width)
            pred_boxes[:, 3] = torch.clamp(pred_boxes[:, 3], max=height)
            
            # then remove any boxes where clipped form has 0 area
            pos_w = (pred_boxes[:,2] - pred_boxes[:,0]) > 0
            pos_h = (pred_boxes[:,3] - pred_boxes[:,1]) > 0 
            good_idxs = (pos_w & pos_h).nonzero().view(-1)
            pred_boxes = pred_boxes[good_idxs]
            pred_classes = pred_classes[good_idxs]
            conf_scores = conf_scores[good_idxs]
            
            # apply non-maximal supression            
            pred_boxes, pred_classes, conf_scores = nms(pred_boxes, pred_classes, conf_scores, max_overlap, 
                                                        rel_thresh, top_k, max_boxes, dup, inc)
            
            # finally add pred_boxes, pred_classes, conf_scores for this image to the lists
            PredBoxes.append(pred_boxes)
            PredClasses.append(pred_classes)
            ConfScores.append(conf_scores)
        
        return PredBoxes, PredClasses, ConfScores
    