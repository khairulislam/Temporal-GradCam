Table of contents
- [3D-GuidedGradCAM-for-Medical-Imaging](#3d-guidedgradcam-for-medical-imaging)
  - [Files:](#files)
  - [How to run](#how-to-run)
  - [deploy\_config.py-](#deploy_configpy-)
  - [Guided\_GradCAM\_3D\_config.py](#guided_gradcam_3d_configpy)
  - [Sample Guided-GardCAM](#sample-guided-gardcam)
  - [Datasets](#datasets)
    - [Caltech-UCSD Birds-200-2011 (CUB-200-2011)](#caltech-ucsd-birds-200-2011-cub-200-2011)
    - [FGVC-Aircraft](#fgvc-aircraft)
    - [Stanford Cars](#stanford-cars)

# [3D-GuidedGradCAM-for-Medical-Imaging](https://github.com/fitushar/3D-GuidedGradCAM-for-Medical-Imaging)
This project imports the the implemnetation of generating Guided-GradCAM for 3D medical Imaging using Nifti file in tensorflow 2.0 from the [original repo](https://github.com/fitushar/3D-GuidedGradCAM-for-Medical-Imaging). Different input files can be used in that case need to edit the input to the Guided-gradCAM model.

## Files:
```ruby  
   i) guided_Gradcam3D.py         |--> Generate Guided-GradCAM , input and output nifti data
  ii) Guided_GradCAM_3D_config.py |--> Configuration file for the Guided-GradCAM, Modify based on your need
 iii) Resnet_3D.py                |--> Network architecture
  iv) deploy_config.py               |--> Configuration file for the Network, Modify based on your need
   v) loss_funnction_And_matrics.py  |--> Loss functions for CNN
```     
## How to run

To run and generate Guided-GardCAM all is to need to configure the `Guided_GradCAM_3D_config.py` and `deploy_config.py`  based on your requiremnet.

## deploy_config.py-
CNN configuration Change based on your Network or complete replace by your CNN
```ruby

import tensorflow as tf
import math
from loss_funnction_And_matrics import*
##---Number-of-GPU
NUM_OF_GPU=1
DISTRIIBUTED_STRATEGY_GPUS=["gpu:0","gpu:1","gpu:2"]
#Network Configuration
NUMBER_OF_CLASSES=5
INPUT_PATCH_SIZE=(224,160,160, 1)
TRAIN_NUM_RES_UNIT=3
TRAIN_NUM_FILTERS=(16, 32, 64, 128)
TRAIN_STRIDES=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2))
TRAIN_CLASSIFY_ACTICATION=tf.nn.relu6
TRAIN_KERNAL_INITIALIZER=tf.keras.initializers.VarianceScaling(distribution='uniform')
TRAIN_CLASSIFY_LEARNING_RATE =1e-4
TRAIN_CLASSIFY_LOSS=Weighted_BCTL
OPTIMIZER=tf.keras.optimizers.Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE,epsilon=1e-5)
TRAIN_CLASSIFY_METRICS=tf.keras.metrics.AUC()
```

## Guided_GradCAM_3D_config.py
Input Configuration for the Guided-GradCAM
```ruby
MODEL_WEIGHT="Path/of/Model/Weight/XXX.h5"
CLASS_INDEX=1 # Index of the class for which you want to see the Guided-gradcam
INPUT_PATCH_SIZE_SLICE_NUMBER=64 # Input patch slice you want to feed at a time
LAYER_NAME='conv3d_18' # Name of the layer from where you want to get the Guided-GradCAM
NIFTI_PATH="imput/niftidata/path/XXX.nii.gz"
SAVE_PATH="/Output/niftydata/path/ML_Guided_GradCaN_XXXX.nii.gz"
```

## Sample Guided-GardCAM

![SAMPLE Guided-GradCAM1](/figures/example1.PNG)
![SAMPLE Guided-GradCAM2](/figures/example2.PNG)
![SAMPLE Guided-GradCAM3](/figures/example3.PNG)

## Datasets
### [Caltech-UCSD Birds-200-2011 (CUB-200-2011)](https://www.vision.caltech.edu/datasets/cub_200_2011/)

Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations.

Number of categories: 200
Number of images: 11,788
Annotations per image: 15 Part Locations, 312 Binary Attributes, 1 Bounding Box
For detailed information about the dataset, please see the technical report linked below. 

```
@techreport{WahCUB_200_2011,
	Title = ,
	Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
	Year = {2011}
	Institution = {California Institute of Technology},
	Number = {CNS-TR-2011-001}
}
```

Download
* [Images and annotations (1.1 GB)](https://data.caltech.edu/records/20098). We do not own the copyrights to these images. Their use is restricted to non-commercial research and educational purposes.
* [Segmentations (37 MB)](https://data.caltech.edu/records/20097). Provided by Ryan Farrell.

### [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

Fine-Grained Visual Classification of Aircraft (FGVC-Aircraft) is a benchmark dataset for the fine grained visual categorization of aircraft. The dataset contains 10,200 images of aircraft, with 100 images for each of 102 different aircraft model variants, most of which are airplanes. The (main) aircraft in each image is annotated with a tight bounding box and a hierarchical airplane model label.

1. [Data, annotations, and evaluation code](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz) [2.75 GB | MD5 Sum].
2. [Annotations and evaluation code only](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b-annotations.tar.gz) [375 KB | MD5 Sum].
3. Project [home page](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/).
4. This data was used as part of the fine-grained recognition challenge [FGComp 2013](https://sites.google.com/site/fgcomp2013/) which ran jointly with the ImageNet Challenge 2013 ([results](https://sites.google.com/site/fgcomp2013/results)). Please note that the evaluation code provided here may differ from the one used in the challenge.

Please use the following citation when referring to this dataset: Fine-Grained Visual Classification of Aircraft, S. Maji, J. Kannala, E. Rahtu, M. Blaschko, A. Vedaldi, [arXiv.org](http://arxiv.org/abs/1306.5151), 2013
```
@techreport{maji13fine-grained,
   title         = {Fine-Grained Visual Classification of Aircraft},
   author        = {S. Maji and J. Kannala and E. Rahtu
                    and M. Blaschko and A. Vedaldi},
   year          = {2013},
   archivePrefix = {arXiv},
   eprint        = {1306.5151},
   primaryClass  = "cs-cv",
}
```

### [Stanford Cars](https://paperswithcode.com/dataset/stanford-cars)

The Stanford Cars dataset consists of 196 classes of cars with a total of 16,185 images, taken from the rear. The data is divided into almost a 50-50 train/test split with 8,144 training images and 8,041 testing images. Categories are typically at the level of Make, Model, Year. The images are 360×240.

The [official repo](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) is dead, but copies can be found [here](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder) and also in [documentation from Tensorflow](https://www.tensorflow.org/datasets/catalog/cars196). 