# NeuralNetworkLibrary

## 1. OVERVIEW
This repository contains a neural network library I have built in pytorch for structured data analysis, collaborative filtering, computer vision, and natural language processing. The library has been written primarily for my own understanding, but I have made an effort to fully comment it, so that it is readable for anyone who wishes to use or modify it.  

The library was inspired by the 2018 version of the fastai course (http://course18.fast.ai), and began as an effort to reconstruct the associated fastai library, in order to deepen my understanding of the course material and improve my programming skills. As such, the API and many of the variable and function names are deliberately quite similar to fastai. However, while a few simple functions have been copied directly, or with minor modification, from the fastai library or lecture notebooks, this library itself has been written primarily from the ground up, and significantly restructured -- using the fastai library only as a rough conceptual template. 

Moreover, the library has evolved somewhat beyond its original intent as I have been working on it, to include some additional content. Most notably, the section for bounding box object detection differs substantially from fastai, and also borrows from the RetinaNet implementation by Yann Henon (https://github.com/yhenon/pytorch-retinanet). Also, a large section on pre-processing and exploratory data analysis for structured data problems has been added, and some extra functionality for visualization and statistical analysis of image data has been incorporated as well.

## 2. HARDWARE REQUIREMENTS
Code is written for a machine containing a single Nvidia GPU. Code will not run on a machine without a GPU, 
and if a machine has multiple GPUs only 1 of them will be utilized (at least so far). 

I have run all my notebooks on a Paperspace P6000 machine which has the following specs: 
Quadro P6000 GPU, 8 CPUs, 30G Ram, and 24G GPU Ram. Batch sizes in notebooks may need 
to be reduced on a machine with significantly less memory. 

## 3. SOFTWARE REQUIREMENTS
1. Cuda Toolkit version 10.0 or 10.1
2. Anaconda python 3.7 with following packages installed: 
pytorch version 1.2 and torchvision version 0.4, numpy, pandas, matplotlib, seaborn, 
sklearn, skimage, ipython, psutil, gputil, spacy, OpenCV. 
 
NOTE: Probably any pytorch version 1.2+ should also work, but I did not test.

## 4. CONTENTS
There are 3 folders: General, Applications, and Examples. 

### General Folder: 
Contains library files which are of general purpose and used for various applications. These include:

1. Core.py - A collection of simple core functions and classes used in various other library files.

2. Layers.py - A small collection of pytorch layers (i.e. modules) used as building blocks in larger models.

3. LossesMetrics.py - A collection of loss functions and other metrics, which are not included as builtin pytorch loss functions. 

4. Optimizer.py - A wrapper around the pytorch optim.Optimizer class, with some extra functionality. 

5. Learner.py - Contains the 'Learner' class, which is the main class in the entire library. This class combines the 
                        pytorch model of a neural network together with the data, the optimizer, and the loss function. The 
                        class also has a large number of methods for training and evaluating models, saving and loading 
                        models, freezing and unfreezing layers, and plotting various quantities of interest. 

### Applications Folder: 
Contains library files used for specific applications. These Include:

1. CollabFiltering.py - For collaborative filtering problems.

2. StructuredData.py - For structured data (i.e. tabular data) problems.

3. Vision.py - For various problems in computer vision, including image classification, 
               multi-label image classification, and bounding box object detection. 

4. Text.py - For various problems in natural language processing, including language modeling 
             and text classification. 

In addition, the applications folder contains three subfolders:

#### Subfolder 1 - VisionModels

This folder contains a collection of image classification models (resnext, inceptionresnetV2, inceptionV4, senet, nasnet) 
which are pre-trained on ImageNet, but not contained in the builtin pytorch model zoo. These models are all taken directly 
from https://github.com/Cadene/pretrained-models.pytorch, in some instances with very minor 
modification to output features instead of logits.

In addition, the VisionModels folder also contains the files retinanet.py and RetinanetPretrainedCOCO.pt. 
The file retinanet.py is an implementation of the retinanet model for bounding box object detection, modified from 
https://github.com/yhenon/pytorch-retinanet. The file RetinanetPretrainedCOCO.pt contains the weights for 
such a retinanet model pre-trained on the COCO dataset (from the same source). 

#### Subfolder 2 - TextModels

This folder contains the weights for an AWD LSTM language model pre-trained on the wikitext103 corpus.
Original weights from the fastai implementation of the pre-trained model are available at "http://files.fast.ai/models/wt103/". Our implementation is significantly restructured code-wise 
from the fastai implementation, but is almost equivalent mathematically, and uses the same pre-trained 
weight matrices in initialization. 

#### Subfolder 3 - pycocotools

A slightly modified version of the python module pycocotools for evaluating bounding box datasets
with annotations in the same format as the COCO dataset (modified to properly deal with 'ignore' flags
from other datasets such as Pascal). 

### Examples Folder: 
Contains a collection of Jupyter Notebooks, each of which uses the library to analyze 1 particular dataset. 
For each notebook there is a folder with the same name, which contains any Kaggle submissions or other 
important files related to the dataset. The datasets themselves are not contained in this repository, 
for storage reasons, but each notebook provides a link to where the dataset can be downloaded. 

To view the notebooks, you should download or clone the repository and open them with the Jupyter Notebook Application. If you just click on the notebooks from the GitHub page, an error sometimes comes up loading them (and even if they load the display is a bit off in some places).

The datasets analyzed in the notebooks include:

1. Kaggle Dogs vs. Cats Dataset (Image Classification)
2. Kaggle Dog Breed Dataset (Image Classification)
3. Kaggle Planet Dataset (Multi-label Image Classification for Satellite Images)
4. Pascal Dataset (Bounding Box Object Detection)
5. MovieLens Dataset (Collaborative Filtering - Movie Ratings Prediction)
6. Kaggle Rossmann Dataset (Structured Data - Sales Prediction)
7. IMDB Large Movie Review Dataset (Language Modeling and Text Classification)

NOTE 1: Each of these datasets is used in the fastai 2018 course. My general approach in the Jupyter notebooks,
is often reasonably similar to the corresponding notebooks from the course. Although, there is often
a bit more exploratory data analysis at the beginning, as well as different models used for training in some cases,
and somewhat varying training strategies. In several cases, model ensembling has also been added at the 
end of the notebooks to improve the accuracy of predictions. 

NOTE 2:
The **DogsCats notebook should be looked at first**, before any of the others. It is thoroughly commented to explain 
the general procedure for training a model with this library, and the various specific steps that are taken along 
the way for the DogsCats example. Other notebooks assume a familiarity with the general procedure from 
the DogsCats notebook, and often comment only on the differences from this standard procedure that 
arise with the different examples. 

NOTE 3:
The output of many cells in the notebooks contains text like *"A Jupyter widget could not be displayed because the widget state could not be found …"* or *"HBox(children=(IntProgress(value=0, max=25000), HTML(value=’’)))"*. All of these outputs come from Jupyter widget progress bars used with training, evaluating, predicting, or other time consuming tasks. When the Jupyter notebook is closed they are not saved. But if you re-run the notebook, with widgets installed, progress bars will appear in these cells as they are running. 

### Internal Dependencies:

Stuff in General Folder:
1. Core.py: No internal dependencies (uses only external libraries such as numpy, matplotlib, … etc.)
2. Layers.py, LossesMetrics.py, Optimizer.py: Each import only Core.py.
3. Learner.py: Imports Core.py, Optimizer.py, LossesMetrics.py 

Stuff in Applications Folder:
1. CollabFiltering.py: Imports everything from General folder. 
2. StructuredData.py: Imports everything from General folder. 
3. Vision.py: Imports everything from General, VisionModels, and pycocotools folders.
4. Text.py: Imports everything from General folder (also uses saved weights in TextModels folder)

## 5. OTHER NOTES
All methods in this library are for supervised learning problems. That is, the training data consists of labeled pairs (x,y), where x is the given input datapoint (e.g. an image) and y is the ground truth output or label for that input (e.g. 'cat' or 'dog'). This naming scheme (x=input, y=output/label) is used in the variable names, docstrings, and comments in the various library files. Outputs predicted by a model are sometimes also denoted by y_hat or y_pred, to distinguish from ground truth values.

The following names in variables and docstrings are also standard in the library:

**bs**: batch size for mini-batches <br>
**ds**: dataset (e.g. train_ds, val_ds, test_ds) <br>
**dl**: data loader (e.g. train_dl, val_dl, test_dl) <br>
**idx**: index <br>
**lr**: learning rate <br>
**wd**: weight decay (specifically L2 weight decay coefficient) <br>
**bn**: batch norm <br>
**cat**: used to denote categorical variables <br>
**cont**: used to denote continuous or real-valued variables <br>
**bbox**: bounding box <br> 
**single_label**: refers to single label image classification, in which each image has 1 label (e.g., 'cat' or 'dog') <br>
**multi_label**: refers to multi label image classification, in which each image may have multiple labels <br>
                 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
                 (e.g. ['table', 'chair', 'book']  to label 1 image containing those 3 objects)
