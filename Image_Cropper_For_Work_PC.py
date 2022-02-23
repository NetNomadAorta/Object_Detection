import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
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
import pathlib
from torch.autograd import Variable
from torchvision.transforms import transforms as T
import json


# User parameters
SAVE_NAME_OD = "./Models-OD/led-2180.model"
DATA_DIR = "./Images/Training_Images/"
USE_CHECKPOINT = True
IMAGE_SIZE = 2180 # Row and column number 2180
DATASET_PATH = "./led_dies/"
AOI_SHAREDRIVE_DIR = "//mcrtp-sftp-01/aoitool/"
TO_PREDICT_PATH = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH = "./Images/Prediction_Images/Predicted_Images/"
FILE_NAME_TO_CROP = "LED-TEST"
RENAME_TOGGLE = False
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
        full_path = os.path.join(dir, f)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)


# Deletes unnecessary string in file name
def replaceFileName(slot_path):
    for file_name in os.listdir(slot_path):
        # For loop with row number as "i" will take longer, so yes below seems
        #   redundant writing each number 1 by 1, but has to be done.
        file_path = os.path.join(slot_path, file_name)
        
        os.rename(file_path, 
                  file_path.replace("Stitcher-Snaps_for_8in_Wafer_Pave.", "")\
                          .replace("Die-1_Pave.", "")\
                          .replace("Die1_Pave.", "")\
                          .replace("Med_El-A_River_1_Pave.", "")\
                          .replace("new_RefDes_1_PaveP1.", "")\
                          .replace("new_RefDes_1_Pave.", "")\
                          .replace("Window_Die1_Pave.", "")\
                          .replace("Row_1.", "Row_01.")\
                          .replace("Col_1.", "Col_01.")\
                          .replace("Row_2.", "Row_02.")\
                          .replace("Col_2.", "Col_02.")\
                          .replace("Row_3.", "Row_03.")\
                          .replace("Col_3.", "Col_03.")\
                          .replace("Row_4.", "Row_04.")\
                          .replace("Col_4.", "Col_04.")\
                          .replace("Row_5.", "Row_05.")\
                          .replace("Col_5.", "Col_05.")\
                          .replace("Row_6.", "Row_06.")\
                          .replace("Col_6.", "Col_06.")\
                          .replace("Row_7.", "Row_07.")\
                          .replace("Col_7.", "Col_07.")\
                          .replace("Row_8.", "Row_08.")\
                          .replace("Col_8.", "Col_08.")\
                          .replace("Row_9.", "Row_09.")\
                          .replace("Col_9.", "Col_09.")\
                          .replace(".p0", "")\
                          .replace(".p1", "")\
                          .replace(".20",".P_")
                          )


# Creates class folder
def makeDir(dir, classes_2):
    for classIndex, className in enumerate(classes_2):
        os.makedirs(dir + className, exist_ok=True)



# Main():
# =============================================================================

# Starting stopwatch to see how long process takes
start_time = time.time()


dataset_path = DATASET_PATH



#load classes
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes_1 = len(categories.keys())
categories

classes_1 = [i[1]['name'] for i in categories.items()]
classes_1



# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes_1)


if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'


# TESTING TO LOAD MODEL
if os.path.isfile(SAVE_NAME_OD):
    checkpoint = torch.load(SAVE_NAME_OD, map_location = map_location)
if USE_CHECKPOINT and os.path.isfile(SAVE_NAME_OD):
    model_1.load_state_dict(checkpoint)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU to train

model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

transforms_1 = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
    A.Rotate(limit=[90,90], always_apply=True),
    ToTensorV2()
])


should_break = False
# This outer for loop runs through each program folder name
for sharedrive_file_name in os.listdir(AOI_SHAREDRIVE_DIR):
    if should_break:
        break
    sharedrive_file_path = os.path.join(AOI_SHAREDRIVE_DIR, sharedrive_file_name)
    
    if FILE_NAME_TO_CROP not in sharedrive_file_name:
        continue
    
    # Removes Thumbs.db if it is found to prevent it screwing up code ahead
    if os.path.isfile(sharedrive_file_path + "/Thumbs.db"):
        os.remove(sharedrive_file_path + "/Thumbs.db")

    # Runs through each slot file within the main file within stitched-image folder
    for slot_name in os.listdir(sharedrive_file_path):
        if should_break:
            break
        slot_path = os.path.join(sharedrive_file_path, slot_name)
        print("Starting", slot_path)
        
        cropped_path = os.path.join(AOI_SHAREDRIVE_DIR, 
                                    FILE_NAME_TO_CROP+"-CROPPED_IMAGES", 
                                    slot_name)
        # Checks to see if output sharedrive already has these files. If so, continue
        isDir = os.path.isdir( cropped_path )
        if isDir:
            print("  This slot already exist! Skipping to next slot!")
            should_break = True
            continue
        
        # Makes directory of cropped folder of what was just checked above
        os.makedirs(cropped_path, exist_ok=True)
        
        # Removes Thumbs.db if it is found to prevent it screwing up code ahead
        if os.path.isfile(slot_path + "/Thumbs.db"):
            os.remove(slot_path + "/Thumbs.db")
        
        # Removes unnecessary naming from original images
        if RENAME_TOGGLE == True:
            print("  Renaming started.")
            replaceFileName(slot_path)
            print("  Renaming completed.")
        
        
        
        pred_dict = {}
        for image_name in os.listdir(slot_path):
            image_path = os.path.join(slot_path, image_name)
            
            # Grabs row and column number from image name and corrects them
            path_row_number = int(image_name[4:6])
            path_col_number = int(image_name[11:13])
            if path_row_number % 2 == 1:
                path_row_number = (path_row_number + 1) // 2
            else:
                path_row_number = 20 + path_row_number // 2
            
            if path_col_number % 2 == 1:
                path_col_number = (path_col_number + 1) // 2
            else:
                path_col_number = 20 + path_col_number // 2  
            
            
            image_b4_color = cv2.imread(image_path)
            image_b4_color_and_rotated = cv2.rotate(image_b4_color, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)
            
            transformed_image = transforms_1(image=image)
            transformed_image = transformed_image["image"]
            
            with torch.no_grad():
                prediction_1 = model_1([(transformed_image/255).to(device)])
                pred_1 = prediction_1[0]
            
            dieCoordinates = pred_1['boxes'][pred_1['scores'] > 0.8]
            # ALLdieCoordinates x y values are SWITCHED BUT IT WRKS
            # dieCoordinates[:, 0] = pred_1['boxes'][pred_1['scores'] > 0.8][:, 1]
            # dieCoordinates[:, 1] = pred_1['boxes'][pred_1['scores'] > 0.8][:, 0]
            # dieCoordinates[:, 2] = pred_1['boxes'][pred_1['scores'] > 0.8][:, 3]
            # dieCoordinates[:, 3] = pred_1['boxes'][pred_1['scores'] > 0.8][:, 2]
            
            box_width = int(dieCoordinates[0][2]-dieCoordinates[0][0]) 
            box_height = int(dieCoordinates[0][3]-dieCoordinates[0][1])
            line_width = round(box_width * 0.0222222222)
            
            if SAVE_FULL_IMAGES:
                test_image = draw_bounding_boxes(transformed_image,
                    dieCoordinates,
                    [classes_1[i] for i in pred_1['labels'][pred_1['scores'] > 0.8].tolist()], 
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
                    rowNumber = math.floor((y1-minY)/(box_width*die_spacing)+1)
                    rowNumber = str(rowNumber)
                    colNumber = math.floor((x1-minX)/(box_height*die_spacing)+1)
                    colNumber = str(colNumber)
                    
                    real_rowNum = (path_row_number - 1)*10 + int(rowNumber)
                    real_colNum = (path_col_number - 1)*10 + int(colNumber)
                    
                    if int(real_colNum) < 10:
                        real_colNum = "00" + str(real_colNum)
                    elif int(real_colNum) < 100:
                        real_colNum = "0" + str(real_colNum)
                    
                    if int(real_rowNum) < 10:
                        real_rowNum = "00" + str(real_rowNum)
                    elif int(real_rowNum) < 100:
                        real_rowNum = "0" + str(real_rowNum)
                    
                    dieNames.append( "R_{}.C_{}".format(real_rowNum, real_colNum) )
                    
                    xmin = int(dieCoordinates[box_index][0])
                    ymin = int(dieCoordinates[box_index][1])
                    xmax = int(dieCoordinates[box_index][2])
                    ymax = int(dieCoordinates[box_index][3])
                    
                    real_image_name = "/R_{}.C_{}.jpg".format(real_rowNum, real_colNum)
                    save_image(transformed_image[:, ymin:ymax, xmin:xmax]/255, 
                                cropped_path + real_image_name)
                    


print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)