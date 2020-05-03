import argparse
import glob
import sys
import os
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
import cv2
import numpy as np
import random
from PIL import Image
import scipy
from multiprocessing import Pool
from functools import partial
import signal
import time

from defaults import *
sys.path.insert(0, POISSON_BLENDING_DIR)
from pb import *
import math
from pyblur import *
from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def randomAngle(kerneldim):
    """Returns a random angle used to produce motion blurring
    Args:
        kerneldim (int): size of the kernel used in motion blurring
    Returns:
        int: Random angle
    """ 
    kernelCenter = int(math.floor(kerneldim/2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])

def get_bins_from_arg(bins):
    #Takes in input a string (ex : '0,40000,2,250,250')
    #Returns target_bins and target nb of augmented images per bin (ex : [[0,20000],[20000,40000]],[250,250]).
    bins=bins.split(",")
    target_bins=[]
    nb_bins=int(bins[2])
    size_inf=int(bins[0])
    size_sup=int(bins[1])
    for i in range(nb_bins):
        target_bins.append((size_inf+i*int((size_sup-size_inf)/nb_bins),size_inf+(i+1)*int((size_sup-size_inf)/nb_bins)))
    target_nb_per_bin=[int(elt) for elt in bins[3:]]
    return(target_bins, target_nb_per_bin)

def LinearMotionBlur3C(img):
    """Performs motion blur on an image with 3 channels. Used to simulate 
       blurring caused due to motion of camera.
    Args:
        img(NumPy Array): Input image with 3 channels
    Returns:
        Image: Blurred image by applying a motion blur with random parameters
    """
    lineLengths = [3,5,7,9]
    lineTypes = ["right", "left", "full"]
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    lineTypeIdx = np.random.randint(0, len(lineTypes)) 
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)
    blurred_img = img
    for i in xrange(3):
        blurred_img[:,:,i] = PIL2array1C(LinearMotionBlur(img[:,:,i], lineLength, lineAngle, lineType))
    blurred_img = Image.fromarray(blurred_img, 'RGB')
    return blurred_img

def overlap(a, b):
    '''Find if two bounding boxes are overlapping or not. This is determined by maximum allowed 
       IOU between bounding boxes. If IOU is less than the max allowed IOU then bounding boxes 
       don't overlap
    Args:
        a(Rectangle): Bounding box 1
        b(Rectangle): Bounding box 2
    Returns:
        bool: True if boxes overlap else False
    '''
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    
    if (dx>=0) and (dy>=0) and float(dx*dy) > MAX_ALLOWED_IOU*(a.xmax-a.xmin)*(a.ymax-a.ymin):
        return True
    else:
        return False

def get_list_of_images(root_dir, glob_string, N=1):
    '''Gets the list of images of objects in the root directory. The expected format 
       is root_dir/<object>/<image>.jpg. Adds an image as many times you want it to 
       appear in dataset.
    Args:
        root_dir(string): Directory where images of objects are present
        N(int): Number of times an image would appear in dataset. Each image should have
                different data augmentation
        glob_string(string) : Suffix of the images (ex : "leftImg8bit.png")
    Returns:
        list: List of images(with paths) that will be put in the dataset
    '''
    #print(os.path.join(root_dir, "objects_dir",'*/*'+glob_string))
    img_list = glob.glob(os.path.join(root_dir,"objects_dir",'*/*'+glob_string)) #change this path
    img_list_f = []
    for i in xrange(N):
        img_list_f = img_list_f + random.sample(img_list, len(img_list))
    return img_list_f

def get_mask_file(img_file):
    '''Takes an image file name and returns the corresponding mask file. The mask represents
       pixels that belong to the object. Default implentation assumes mask file has same path 
       as image file with different extension only. Write custom code for getting mask file here
       if this is not the case.
    Args:
        img_file(string): Image name
    Returns:
        string: Corresponding mask file path
    '''
    mask_file = img_file.replace('.png','.pbm')
    return mask_file

def get_labels(imgs):
    '''Get list of labels/object names. Assumes the images in the root directory follow root_dir/<object>/<image>
       structure. Directory name would be object name.
    Args:
        imgs(list): List of images being used for synthesis 
    Returns:
        list: List of labels/object names corresponding to each image
    '''
    labels = []
    for img_file in imgs:
        label = img_file.split('/')[-2]
        labels.append(label)
    return labels

def get_annotation_from_mask_file(mask_file, scale=1.0):
    '''Given a mask file and scale, return the bounding box annotations
    Args:
        mask_file(string): Path of the mask file
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    if os.path.exists(mask_file):
        mask = cv2.imread(mask_file)
        if INVERTED_MASK:
            mask = 255 - mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if len(np.where(rows)[0]) > 0:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return int(scale*xmin), int(scale*xmax), int(scale*ymin), int(scale*ymax)
        else:
            return -1, -1, -1, -1
    else:
        print ("%s not found. Using empty mask instead."%mask_file)
        return -1, -1, -1, -1

def get_annotation_from_mask(mask):
    '''Given a mask, this returns the bounding box annotations
    Args:
        mask(NumPy Array): Array with the mask
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if len(np.where(rows)[0]) > 0:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, xmax, ymin, ymax
    else:
        return -1, -1, -1, -1

def write_imageset_file(exp_dir, img_files, anno_files):
    '''Writes the imageset file which has the generated images and corresponding annotation files
       for a given experiment
    Args:
        exp_dir(string): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        img_files(list): List of image files that were generated
        anno_files(list): List of annotation files corresponding to each image file
    '''
    with open(os.path.join(exp_dir,'train.txt'),'w') as f:
        for i in xrange(len(img_files)):
            f.write('%s %s\n'%(img_files[i], anno_files[i]))

def write_labels_file(exp_dir, labels):
    '''Writes the labels file which has the name of an object on each line
    Args:
        exp_dir(string): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        labels(list): List of labels. This will be useful while training an object detector
    '''
    unique_labels = ['__background__'] + sorted(set(labels))
    with open(os.path.join(exp_dir,'labels.txt'),'w') as f:
        for i, label in enumerate(unique_labels):
            f.write('%s %s\n'%(i, label))

def keep_selected_labels(img_files, labels):
    '''Filters image files and labels to only retain those that are selected. Useful when one doesn't 
       want all objects to be used for synthesis
    Args:
        img_files(list): List of images in the root directory
        labels(list): List of labels corresponding to each image
    Returns:
        new_image_files(list): Selected list of images
        new_labels(list): Selected list of labels corresponidng to each imahe in above list
    '''
    with open(SELECTED_LIST_FILE) as f:
        selected_labels = [x.strip() for x in f.readlines()]
    new_img_files = []
    new_labels = []
    for i in xrange(len(img_files)):
        if labels[i] in selected_labels:
            new_img_files.append(img_files[i])
            new_labels.append(labels[i])
    return new_img_files, new_labels

def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array
    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0])

def PIL2array3C(img):
    '''Converts a PIL image to NumPy Array
    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def create_image_anno_wrapper(args, w=WIDTH, h=HEIGHT, anno_file_txt=ANNO_FILE_TXT, scale_augment=False, rotation_augment=False, blending_list=['none'], dontocclude=False):
   ''' Wrapper used to pass params to workers
   '''
   return create_image_anno(*args, w=w, h=h, scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=blending_list, dontocclude=dontocclude)

def create_image_anno(objects, img_file, anno_file, bg_file, output_idx, target_size_bin=None, w=WIDTH, h=HEIGHT, anno_file_txt=ANNO_FILE_TXT, scale_augment=False, rotation_augment=False, blending_list=['none'], dontocclude=False):
    '''Add data augmentation, synthesizes images and generates annotations according to given parameters
    Args:
        objects(list): List of objects whose annotations are also important
        img_file(str): Image file name
        anno_file(str): Annotation file name
        bg_file(str): Background image path 
        target_size_bin(tuple) : interval of object size to reach
        w(int): Width of synthesized image
        h(int): Height of synthesized image
        scale_augment(bool): Add scale data augmentation
        rotation_augment(bool): Add rotation data augmentation
        blending_list(list): List of blending modes to synthesize for each image
        dontocclude(bool): Generate images with occlusion
    '''
    print("")
    print("objects : ",objects,len(objects))
    if 'none' not in img_file:
        return 
    print ("Working on %s" % img_file)
    if os.path.exists(anno_file):
        return anno_file
    
    #load background mask
    print("bg_file : ",bg_file)
    bg_mask_file=bg_file[:-15]+"gtFine_color.png"
    bg_mask=np.array(Image.open(bg_mask_file))
    road_colors=[[128,64,128, 255],[81,  0, 81,255]]
    
    print("Target size bin : ",target_size_bin)
        
    assert len(objects) > 0
    while True:
        #Try different pastings on the current background image. If the object is not on the road, try again.
        top = Element('annotation')
        background = Image.open(bg_file)
        bg_rescale=(background.size[0]/w,background.size[1]/h)  #bg_rescale = (1,1) if the out image has same dimensions as w
        print("bg_rescale : ",bg_rescale)
        background = background.resize((w, h), Image.ANTIALIAS)
        backgrounds = []
        for i in xrange(len(blending_list)):
            backgrounds.append(background.copy())
        if dontocclude:
            already_syn = []
            
        for idx, obj in enumerate(objects):
           attempt = 0
        
           foreground = Image.open(obj[0])
           xmin, xmax, ymin, ymax = get_annotation_from_mask_file(get_mask_file(obj[0]))
           if xmin == -1 or ymin == -1 or xmax-xmin < MIN_WIDTH or ymax-ymin < MIN_HEIGHT :
               continue
           foreground = foreground.crop((xmin, ymin, xmax, ymax))
           orig_w, orig_h = foreground.size
           mask_file =  get_mask_file(obj[0])
           mask = Image.open(mask_file)
           mask = mask.crop((xmin, ymin, xmax, ymax))
           if INVERTED_MASK:
               mask = Image.fromarray(255-PIL2array1C(mask))
           o_w, o_h = orig_w, orig_h
        
           #Rescale the object to the target size
           target_area = random.uniform(target_size_bin[0],target_size_bin[1])
           print("target_area : ",target_area)
           scale = math.sqrt(target_area/(orig_w*orig_h))
           o_w, o_h = int(scale*orig_w*bg_rescale[0]), int(scale*orig_h*bg_rescale[0])
           print("o_w,o_h : ",int(scale*orig_w), int(scale*orig_h), o_w, o_h)
            
           if  w-o_w < 0 or h-o_h < 0 or o_w < 0 or o_h < 0:
               break
           foreground = foreground.resize((o_w, o_h), Image.ANTIALIAS)
           mask = mask.resize((o_w, o_h), Image.ANTIALIAS)
           """
           if scale_augment:
                while True:
                    scale = random.uniform(MIN_SCALE, MAX_SCALE)
                    o_w, o_h = int(scale*orig_w), int(scale*orig_h)
                    if  w-o_w > 0 and h-o_h > 0 and o_w > 0 and o_h > 0:
                        break
                foreground = foreground.resize((o_w, o_h), Image.ANTIALIAS)
                mask = mask.resize((o_w, o_h), Image.ANTIALIAS)"""
           
           if rotation_augment:
               max_degrees = MAX_DEGREES  
               while True:
                   rot_degrees = random.randint(-max_degrees, max_degrees)
                   foreground_tmp = foreground.rotate(rot_degrees, expand=True)
                   mask_tmp = mask.rotate(rot_degrees, expand=True)
                   o_w, o_h = foreground_tmp.size
                   if  w-o_w > 0 and h-o_h > 0:
                        break
               mask = mask_tmp
               foreground = foreground_tmp
           xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)
           contained=False
            
           while True:
               print("attempt : ",attempt)
               attempt +=1
               #x = random.randint(int(-MAX_TRUNCATION_FRACTION*o_w), int(w-o_w+MAX_TRUNCATION_FRACTION*o_w))
               #y = random.randint(int(-MAX_TRUNCATION_FRACTION*o_h), int(h-o_h+MAX_TRUNCATION_FRACTION*o_h))
               x = random.randint(0,w-o_w-1)
               y = random.randint(0,h-o_h-1)
               print("x,y : ",x,y)
            
               #check if object is on the road
               for rc in road_colors :
                   print("bg_mask.shape : ",bg_mask.shape)
                   print(y+o_h,x+(o_w/2))
                   print("road colors : ",bg_mask[y+int(o_h/2)][x+o_w],rc)
                   if (bg_mask[y+int(o_h/2)][x+o_w]==rc).all():
                        contained=True
               #if (bg_mask[x+o_h][y+(o_w/2)]==road_colors[0]).all() or (bg_mask[x+o_h][y+(o_w/2)]==road_colors[1]).all():
                   #contained=True
               print("contained : ",contained)
               #prevent occlusion (ie : several pasted objects overlapping each other). 
               if dontocclude:
                   found = True
                   for prev in already_syn:
                       ra = Rectangle(prev[0], prev[2], prev[1], prev[3])
                       rb = Rectangle(x+xmin, y+ymin, x+xmax, y+ymax)
                       if overlap(ra, rb):
                             found = False
                             break
                   if found & contained:
                      break
               if contained:
                   break
               if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
                   break
           if dontocclude:
               already_syn.append([x+xmin, x+xmax, y+ymin, y+ymax])
           for i in xrange(len(blending_list)):
               if blending_list[i] == 'none' or blending_list[i] == 'motion':
                   backgrounds[i].paste(foreground, (x, y), mask)
               elif blending_list[i] == 'poisson':
                  offset = (y, x)
                  img_mask = PIL2array1C(mask)
                  img_src = PIL2array3C(foreground).astype(np.float64)
                  img_target = PIL2array3C(backgrounds[i])
                  img_mask, img_src, offset_adj \
                       = create_mask(img_mask.astype(np.float64),
                          img_target, img_src, offset=offset)
                  background_array = poisson_blend(img_mask, img_src, img_target,
                                    method='normal', offset_adj=offset_adj)
                  backgrounds[i] = Image.fromarray(background_array, 'RGB') 
               elif blending_list[i] == 'gaussian':
                  backgrounds[i].paste(foreground, (x, y), Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask),(5,5),2)))
               elif blending_list[i] == 'box':
                  backgrounds[i].paste(foreground, (x, y), Image.fromarray(cv2.blur(PIL2array1C(mask),(3,3))))
           if idx >= len(objects):
               continue 
           object_root = SubElement(top, 'object')
           object_type = obj[1]
           object_type_entry = SubElement(object_root, 'name')
           object_type_entry.text = str(object_type)
           object_bndbox_entry = SubElement(object_root, 'bndbox')
           x_min_entry = SubElement(object_bndbox_entry, 'xmin')
           x_min_entry.text = '%d'%(max(1,x+xmin))
           x_max_entry = SubElement(object_bndbox_entry, 'xmax')
           x_max_entry.text = '%d'%(min(w,x+xmax))
           y_min_entry = SubElement(object_bndbox_entry, 'ymin')
           y_min_entry.text = '%d'%(max(1,y+ymin))
           y_max_entry = SubElement(object_bndbox_entry, 'ymax')
           y_max_entry.text = '%d'%(min(h,y+ymax))
           difficult_entry = SubElement(object_root, 'difficult')
           difficult_entry.text = '0' # Add heuristic to estimate difficulty later on
           line_txt=str(output_idx)+" "+img_file+" "+str(w)+" "+str(h)+" 0 "+str(max(1,x+xmin))+" "+str(max(1,y+ymin))+" "+str(min(w,x+xmax))+" "+str(min(h,y+ymax))
        if attempt == MAX_ATTEMPTS_TO_SYNTHESIZE:
           continue
        else:
           break
    for i in xrange(len(blending_list)):
        if blending_list[i] == 'motion':
            backgrounds[i] = LinearMotionBlur3C(PIL2array3C(backgrounds[i]))
        backgrounds[i].save(img_file.replace('none', blending_list[i]))
    #Write .xml annotation
    xmlstr = xml.dom.minidom.parseString(tostring(top)).toprettyxml(indent="    ")
    print("xmlstr : ",xmlstr)
    with open(anno_file, "w") as f:
        f.write(xmlstr)
    #Write .txt annotation
    if "line_txt" in vars() or 'line_txt' in globals():
        print("line_txt : ",line_txt)
        with open(anno_file_txt, "a+") as f_txt:
            f_txt.write(line_txt+'\n')

def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)
 
def generate_synthetic_dataset(args):
    ''' Generate synthetic dataset according to given args : get objects, object labels, backgrounds, background labels.
    Get : img_files(list): List of object image files
        labels(list): List of labels for each object
        background_files(str) : list of background files
        img_dir(str): Directory where synthesized images will be stored
        anno_dir(str): Directory where corresponding annotations will be stored
        scale_augment(bool): Add scale data augmentation
        rotation_augment(bool): Add rotation data augmentation
        dontocclude(bool): Generate images with occlusion
        allow_duplicate_obj(bool) : Allow to use object images more than once if needed
    '''
    
    #Get object images, and their annotations
    img_files = get_list_of_images(args.root,OBJECT_GLOB_STRING,args.num) 
    
    labels = get_labels(img_files)
    print("img_files : ",len(img_files),img_files)
    print("labels : ",len(labels),labels)
    print("")
    
    #Filter selected images (option)
    if args.selected:
        img_files, labels = keep_selected_labels(img_files, labels)
        
    #Create output directories
    if not os.path.exists(args.exp):
        os.makedirs(args.exp)
    write_labels_file(args.exp, labels)
    anno_dir = os.path.join(args.exp, 'annotations')
    anno_dir_txt=os.path.join(args.exp,'annotations_txt')
    anno_twoclasses_txt=os.path.join(args.exp,'annotations_twoclasses_txt')
    img_dir = os.path.join(args.exp, 'images')
    if not os.path.exists(os.path.join(anno_dir)):
        os.makedirs(anno_dir)
    if not os.path.exists(os.path.join(img_dir)):
        os.makedirs(img_dir)
    if not os.path.exists(os.path.join(anno_dir_txt)):
        os.makedirs(anno_dir_txt)
    if not os.path.exists(os.path.join(anno_twoclasses_txt)):
        os.makedirs(anno_twoclasses_txt)
    #Pasting options
    scale_augment, rotation_augment, dontocclude=args.scale, args.rotation, args.dontocclude

    #Get size bins, and the number of augmented image per bin 
    target_bins,target_nb_per_bin=get_bins_from_arg(args.target_size_bins)
    print("bins : ", target_bins,target_nb_per_bin)
    
    #Get background images
    background_dir = BACKGROUND_DIR
    background_files = glob.glob(os.path.join(background_dir, BACKGROUND_GLOB_STRING))
    print ("Number of background images : %s"%len(background_files))
    

    img_labels = zip(img_files, labels)
    print("img_labels : ",len(img_labels))
    if args.allow_duplicate_obj:
        #ie : use each object more than once if needed
        n_dupl=int(sum(target_nb_per_bin)/len(img_labels))
        tmp=img_labels
        for i_dupl in range(n_dupl):
            img_labels=img_labels+tmp
    print("img_labels : ",len(img_labels))
    random.shuffle(img_labels)
    
    #Store parameters for the experiment (objects, img_file, anno_file, bg_file)
    output_idx = int(args.id_start)
    syn_img_files = []
    anno_files = []
    params_list = []
    
    for i_bin in range(len(target_bins)):
        #for each size bin, create the target number of augmented images
        for j_bin in range(target_nb_per_bin[i_bin]):
            # Get list of objects
            target_img_bin=target_bins[i_bin]
            print("j_bin : ",j_bin,i_bin)
            print("current bin : ",target_bins[i_bin], target_nb_per_bin[i_bin])
            objects = []
            n = min(random.randint(MIN_NO_OF_OBJECTS, MAX_NO_OF_OBJECTS), len(img_labels))
            for i in xrange(n):
                objects.append(img_labels.pop())
            output_idx += 1
            bg_file = random.choice(background_files)
            for blur in BLENDING_LIST:
                bg_name=bg_file.split("/")[-1].split(".")[0]
                img_file = os.path.join(img_dir, '%s_%i_%s.png'%(bg_name,output_idx,blur))
                anno_file = os.path.join(anno_dir, '%s_%i.xml'%(bg_name,output_idx))
                params = (objects, img_file, anno_file, bg_file,output_idx,target_img_bin)
                params_list.append(params)
                syn_img_files.append(img_file)
                anno_files.append(anno_file)
            if len(img_labels)==0:
                print("Not enough objects to create the desired number of synthetic images. All objects have been used once.")
                break
        if len(img_labels)==0:
            break

    #Width and height of output images
    w = WIDTH
    h = HEIGHT
    
    """
    #### DEBUGGING
    #Run the code without multiprocessing to troubleshoot
    for param in params_list:
        print("")
        print("param : ",param)
        (objects, img_file, anno_file, bg_file,output_idx,target_img_bin)=param
        print("objects : ",objects)
        print("img_file : ",img_file)
        print("anno_file : ",anno_file)
        print("bg_file : ",bg_file)
        print("target_img_bin : ",target_img_bin)
        
        create_image_anno(objects, img_file, anno_file, bg_file,output_idx,target_img_bin,w=WIDTH, h=HEIGHT,anno_file_txt=ANNO_FILE_TXT)
    ####
    """

    partial_func = partial(create_image_anno_wrapper, w=w, h=h, anno_file_txt=ANNO_FILE_TXT,scale_augment=scale_augment, rotation_augment=rotation_augment, blending_list=BLENDING_LIST, dontocclude=dontocclude) 
    p = Pool(NUMBER_OF_WORKERS, init_worker)
    try:
        p.map(partial_func, params_list)
    except KeyboardInterrupt:
        print ("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()
    
    #Write results
    write_imageset_file(args.exp, syn_img_files, anno_files)
    
def parse_args():
    '''Parse input arguments
    '''
    parser = argparse.ArgumentParser(description="Create dataset with different augmentations")
    parser.add_argument("root",
      help="The root directory which contains the images and annotations.")
    parser.add_argument("exp",
      help="The directory where images and annotation lists will be created.")
    parser.add_argument("--selected",
      help="Keep only selected instances in the test dataset. Default is to keep all instances in the root directory", action="store_true")
    parser.add_argument("--scale",
      help="Add scale augmentation.Default is to add scale augmentation.", action="store_false")
    parser.add_argument("--rotation",
      help="Add rotation augmentation.Default is to add rotation augmentation.", action="store_false")
    parser.add_argument("--num",
      help="Number of times each image will be in dataset", default=1, type=int)
    parser.add_argument("--dontocclude",
      help="Add objects without occlusion. Default is to produce occlusions", action="store_true")
    parser.add_argument("--id_start",
      help="id to start",default=0)
    parser.add_argument("--target_size_bins",
      help="Target bin sizes and number of images for each bin.",default=1)
    parser.add_argument("--allow_duplicate_obj",
      help="Allow to use object images more than once, in order to generate enough augmented images (required if we want to generate more than 300 images).",default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    generate_synthetic_dataset(args)