# Data Augmentation with SynDataGeneration 

## Method

This code pasting and blending of objects images on road background images, to generate an augmented dataset for obstacle detection.
The dataset used for objects is the [RGBD_objects dataset](http://rgbd-dataset.cs.washington.edu/index.html). Only a single image of each object is used in our case (300 objects in total). These objects might be used several times with different backgrounds, to get the right number of augmented images if needed.
The dataset used for road background is [Cityscapes](https://www.cityscapes-dataset.com/).

This method allows for the compensation of imbalances in the dataset. Currently, it supports the compensation of object size imbalance in an object detection dataset.

This augmented dataset can then be used in experiments for the training of obstacle detectors, in addition or combination with other obstacle detection datasets such as Lost and Found.

## Implementation details

**Directory structure**

```
.  
├─ demo_data_dir/               demo dataset
├─ pb/                          poisson blending repository from https://github.com/yskmt/pb
├─ README.md                    this file
├─ aug_cityscapes.sh            augment the Cityscapes dataset
├─ aug_demo.sh                  augment the demo dataset
├─ cityscapes_preproc           selectively imports and modifies Cityscapes dataset for use
├─ convert_annotations.ipynb    converts xml annotations for use with our YOLOv3 package
├─ dataset_generator.py         script for generating augmented dataset
├─ defaults.py                  for configuring variables
├─ defaults_template.py         template for defaults
├─ delete_images.sh             delete generated images
└─ object_size.ipynb            visualises sizes of objects of non- and augmented datasets 
```

**Requirements**

```
OpenCV
PIL
PyBlur
```

## Usage

**Download the dataset**

Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset to your directory of choice.

Get images (.png) and annotations (.pbm) files from RGBD dataset (http://rgbd-dataset.cs.washington.edu/index.html) and move them to ./input_dir_cityscapes_rgbd/objects_dir/RGBD_objects.

The algorithm assumes that object masks for each image are present in the dataset.

**Set defaults**

Set up `defaults.py` to reflect the correct paths to various files and libraries on your system.

The other defaults refer to different image generating parameters that might be varied to produce scenes with different levels of clutter, occlusion, data augmentation etc.

In particular, select the pasting methods ('poisson','gaussian','motion').

**Initialisation**

`cd` to the repository root directory and run:
```
python init_packages.py -p Data_augmentation
```
This will install dependencies. It also saves a sample of Cityscapes into the `backgrounds` folder of the specified output directory.

**Generate an augmented dataset**

To augment the demo dataset with our default parameters, run
```
bash aug_demo.sh
```

To augment the Cityscapes dataset with our default parameters, run
```
bash aug_cityscapes.sh
```

In this script, the argument `--target_size_bins` allows to create a certain number of augmented images for each given size range. The notebook `object_size.ipynb` visualises the distribution of object sizes of the LaF dataset, and the generated augmented dataset.

Please see below for more details on the underlying Python script used for generation.
```
python dataset_generator.py [-h] [--selected] [--scale] [--rotation]
                            [--num NUM] [--dontocclude] [--add_distractors]
                            root exp --target_size_bins --id_start

Create dataset with different augmentations

positional arguments:
  root               The root directory which contains the images and
                     annotations.
  exp                The directory where images and annotation lists will be
                     created.

optional arguments:
  -h, --help         show this help message and exit
  --selected         Keep only selected instances in the test dataset. Default
                     is to keep all instances in the roo directory.
  --scale            Add scale augmentation.Default is to not add scale
                     augmentation.
  --rotation         Add rotation augmentation.Default is to not add rotation
                     augmentation.
  --num NUM          Number of times each image will be in dataset
  --dontocclude      Add objects without occlusion. Default is to produce
                     occlusions
  --add_distractors  Add distractors objects. Default is to not use
                     distractors
  --target_size_bins Size of bins and target nb of augmented image per bin :
                      (min_size,max_size,nb_bins), [nb of images per bin].
  
  --id_start         id of the first output_image.
  
  dataset_generator_cityscapes.sh is an example of how the arguments should look like
```

**Delete generated images and annotations**

Run
```
export DIR_OUTPUT=output_dir_path
```

Then run
```
./delete_images.sh
```

## Training an object detector

The code produces all the files required to train an object detector. The format was orginally designed for for Faster R-CNN but should be adapted for different object detectors. The different files produced are:

1. __labels.txt__ - Contains the labels of the objects being trained
1. __annotations/*.xml__ - Contains annotation files in XML format which contain bounding box annotations for various scenes
1. __images/*.jpg__ - Contain image files of the synthetic scenes in JPEG format 
1. __train.txt__ - Contains list of synthetic image files and corresponding annotation files
1. ./output_dir_cityscapes_rgbd/annotations_txt/annotations.txt : Contains each annotation for each image in a single file (format used in the Tensorflow implementation of YOLOv3 (https://github.com/wizyoung/YOLOv3_TensorFlow).

To include other classes (eg : car) in the the labels, use Add Cityscapes annotations.ipynb : by default, the resulting .txt annotation file will be generated in ./output_dir_cityscapes_rgbd/annotations_twoclasses_txt/annotations_twoclasses.txt
    
## Paper
This work is based on [Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection](https://arxiv.org/abs/1708.01642) and the author's repository : https://github.com/debidatta/syndata-generation.
