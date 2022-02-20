import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import copy
import math
from PIL import Image
import cv2
import albumentations as A  # our data augmentation library
import matplotlib.pyplot as plt
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm # progress bar
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
import shutil


# User parameters
SAVE_NAME = "./led-500.model"
USE_CHECKPOINT = True
IMAGE_SIZE = 800 # Row and column number 2180
DATASET_PATH = "./led_dies/"
TO_PREDICT_PATH = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH = "./Images/Prediction_Images/Predicted_Images/"
SAVE_FULL_IMAGES = False
SAVE_CROPPED_IMAGES = True
NUMBER_EPOCH = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 16
DIE_SPACING_SCALE = 0.99



def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))


def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
            A.Rotate(limit=[90,90], always_apply=True),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
            A.Rotate(limit=[90,90], always_apply=True),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform



# Starting stopwatch to see how long process takes
start_time = time.time()

# Deletes images already in "Predicted_Images" folder
deleteDirContents(PREDICTED_PATH)

dataset_path = DATASET_PATH



#load classes
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes = len(categories.keys())
categories

classes = [i[1]['name'] for i in categories.items()]
classes



# lets load the faster rcnn model
model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)


# TESTING TO LOAD MODEL
if os.path.isfile(SAVE_NAME):
    checkpoint = torch.load(SAVE_NAME)
if USE_CHECKPOINT and os.path.isfile(SAVE_NAME):
    model.load_state_dict(checkpoint)


device = torch.device("cuda") # use GPU to train
model = model.to(device)

model.eval()
torch.cuda.empty_cache()

transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
    A.Rotate(limit=[90,90], always_apply=True),
    ToTensorV2()
])


for image_name in os.listdir(TO_PREDICT_PATH):
    image_path = os.path.join(TO_PREDICT_PATH, image_name)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformed_image = transforms(image=image)
    transformed_image = transformed_image["image"]
    
    with torch.no_grad():
        prediction = model([(transformed_image/255).to(device)])
        pred = prediction[0]
    
    dieCoordinates = pred['boxes'][pred['scores'] > 0.8]
    # ALLdieCoordinates x y values are SWITCHED BUT IT WRKS
    # dieCoordinates[:, 0] = pred['boxes'][pred['scores'] > 0.8][:, 1]
    # dieCoordinates[:, 1] = pred['boxes'][pred['scores'] > 0.8][:, 0]
    # dieCoordinates[:, 2] = pred['boxes'][pred['scores'] > 0.8][:, 3]
    # dieCoordinates[:, 3] = pred['boxes'][pred['scores'] > 0.8][:, 2]
    
    box_width = int(dieCoordinates[0][2]-dieCoordinates[0][0]) 
    box_height = int(dieCoordinates[0][3]-dieCoordinates[0][1])
    line_width = round(box_width * 0.0222222222)
    
    if SAVE_FULL_IMAGES:
        test_image = draw_bounding_boxes(transformed_image,
            dieCoordinates,
            [classes[i] for i in pred['labels'][pred['scores'] > 0.8].tolist()], 
            width=line_width
            )
        
        # Saves full image with bounding boxes
        save_image((test_image/255), PREDICTED_PATH + image_name)
    
    if SAVE_CROPPED_IMAGES:
        # # Sets spacing between dies
        die_spacing_max = int(box_width * .1) # I guessed
        die_spacing = 1 + round( (die_spacing_max/box_width)*DIE_SPACING_SCALE, 3)
        
        # Grabbing max and min x and y coordinate values
        minX = int( torch.min(dieCoordinates[:, 0]) )
        minY = int( torch.min(dieCoordinates[:, 1]) )
        maxX = int( torch.max(dieCoordinates[:, 2]) )
        maxY = int( torch.max(dieCoordinates[:, 3]) )
        
        dieNames = []
        
        # Changes column names in dieNames
        for box_index in range(len(dieCoordinates)):
            
            x1 = int( dieCoordinates[box_index][0] )
            y1 = int( dieCoordinates[box_index][1] )
            x2 = int( dieCoordinates[box_index][2] )
            y2 = int( dieCoordinates[box_index][3] )
            
            midX = round((x1 + x2)/2)
            midY = round((y1 + y2)/2)
            
            # Creates dieNames list row and column number
            rowNumber = str(math.floor((y1-minY)/(box_width*die_spacing)+1) )
            colNumber = str(math.floor((x1-minX)/(box_height*die_spacing)+1) )
    
            # THIS PART IS FOR LED 160,000 WAFER!
            if int(colNumber)>200:
                colNumber = str( int(colNumber) )
            
            if int(colNumber) < 10:
                colNumber = "00" + colNumber
            elif int(colNumber) < 100:
                colNumber = "0" + colNumber
            
            if int(rowNumber)>200:
                rowNumber = str( int(rowNumber) )
            
            if int(rowNumber) < 10:
                rowNumber = "00" + rowNumber
            elif int(colNumber) < 100:
                rowNumber = "0" + rowNumber
            
            dieNames.append( "R_{}.C_{}".format(rowNumber, colNumber) )
            
            xmin = int(dieCoordinates[box_index][0])
            ymin = int(dieCoordinates[box_index][1])
            xmax = int(dieCoordinates[box_index][2])
            ymax = int(dieCoordinates[box_index][3])
            
            save_image(transformed_image[:, ymin:ymax, xmin:xmax]/255, 
                        PREDICTED_PATH + image_name[:-4] + "-R_{}.C_{}.jpg".format(rowNumber, colNumber) )




print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)