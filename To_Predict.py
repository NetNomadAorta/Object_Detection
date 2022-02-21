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


# User parameters
SAVE_NAME_OD = "./Models-OD/led-500.model"
SAVE_NAME_ML = "./Models-ML/LED-162.model"
DATA_DIR = "./Images/Training_Images/"
USE_CHECKPOINT = True
IMAGE_SIZE = 800 # Row and column number 2180
# RESIZE for ML part
ROW_RESIZE = 162 # Number of row pixels in each image to RESIZE
COL_RESIZE = 162 # number of column pixels in each image to RESIZE
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
        full_path = os.path.join(dir, f)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)


# Creates class folder
def makeDir(dir, classes_2):
    for classIndex, className in enumerate(classes_2):
        os.makedirs(dir + className, exist_ok=True)



# Starting stopwatch to see how long process takes
start_time = time.time()

# Deletes images already in "Predicted_Images" folder
deleteDirContents(PREDICTED_PATH)

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


# TESTING TO LOAD MODEL
if os.path.isfile(SAVE_NAME_OD):
    checkpoint = torch.load(SAVE_NAME_OD)
if USE_CHECKPOINT and os.path.isfile(SAVE_NAME_OD):
    model_1.load_state_dict(checkpoint)


device = torch.device("cuda") # use GPU to train
model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

transforms_1 = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
    A.Rotate(limit=[90,90], always_apply=True),
    ToTensorV2()
])



# FOR ML MODEL
# -----------------------------------------------------------------------------

# Checks to see what device used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

# Categories/Classes
root = pathlib.Path(DATA_DIR)
classes_2 = sorted([j.name.split('/')[-1] for j in root.iterdir()])

# CNN Network
class ConvNet(nn.Module):
    def __init__(self, num_classes_2 = len(classes_2)):
        super(ConvNet, self).__init__()
        
        # Output size after convolution filter
        # ((w-f+2P)/s)+1
        # ((ROW_RESIZE-kernel_size+2*padding)/stride)+1
        
        # Input shape = (BATCH_SIZE, 3, ROW_RESIZE, COL_RESIZE)
        
        # First layer
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 1, padding = 1)
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.bn1 = nn.BatchNorm2d(num_features = 8)
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.relu1 = nn.ReLU()
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        
        # Second layer
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.bn2 = nn.BatchNorm2d(num_features = 16) # MAYBE ADD THIS TO FIRST LAYER AND TAKE OFF WEIGHT DECAY LATER
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.relu2 = nn.ReLU()
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.pool2 = nn.MaxPool2d(kernel_size = 3)
        # Reduce the image size by factor of 3
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE/3, COL_RESIZE/3)
        self.dropout2 = nn.Dropout2d(p = 0.25)
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE/3, COL_RESIZE/3)
        
        # Third layer
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.bn3 = nn.BatchNorm2d(num_features = 16)
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.relu3 = nn.ReLU()
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.pool3 = nn.MaxPool2d(kernel_size = 3)
        # Reduce the image size by factor of 3
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE/9, COL_RESIZE/9)
        self.dropout3 = nn.Dropout2d(p = 0.25)
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE/9, COL_RESIZE/9)
        
        # 4th layer
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.bn4 = nn.BatchNorm2d(num_features = 16)
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.relu4 = nn.ReLU()
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.pool4 = nn.MaxPool2d(kernel_size = 3)
        # Reduce the image size by factor of 3
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE/27, COL_RESIZE/27)
        self.dropout4 = nn.Dropout2d(p = 0.25)
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE/27, COL_RESIZE/27)
        
        # 5th layer
        self.conv5 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.bn5 = nn.BatchNorm2d(num_features = 16)
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.relu5 = nn.ReLU()
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE, COL_RESIZE)
        self.pool5 = nn.MaxPool2d(kernel_size = 3)
        # Reduce the image size by factor of 3
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE/81, COL_RESIZE/81)
        self.dropout5 = nn.Dropout2d(p = 0.25)
        # Shape = (BATCH_SIZE, out_channels, ROW_RESIZE/81, COL_RESIZE/81)
        
        # Final layer
        self.flat_densefc = nn.Linear(in_features = int(16 * ROW_RESIZE/81 * COL_RESIZE/81), out_features = 32)
        self.relufc = nn.ReLU()
        self.dropoutfc = nn.Dropout2d(p = 0.5)
        self.densefc = nn.Linear(in_features = 32, out_features = len(classes_2))
        # self.softfc = nn.Softmax(dim = 1)
        
        
    # Feed forward function
    def forward(self, input):
        # First layer
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        # Second layer
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.pool2(output)
        output = self.dropout2(output)
        # Third layer
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = self.pool3(output)
        output = self.dropout3(output)
        # 4th layer
        output = self.conv4(output)
        output = self.bn4(output)
        output = self.relu4(output)
        output = self.pool4(output)
        output = self.dropout4(output)
        # 5th layer
        output = self.conv5(output)
        output = self.bn5(output)
        output = self.relu5(output)
        output = self.pool5(output)
        output = self.dropout5(output)
        # Above output will be in matrix form, with shape (BATCH_SIZE, out_channels, ROW_RESIZE/9, COL_RESIZE/9)
        # Final layer # I CHAAANGGEED ROW_RESIZE/9 * COL_RESIZE/9 -> ROW_RESIZE/3 * COL_RESIZE/3
        output = output.view(-1, int(16 * ROW_RESIZE/81 * COL_RESIZE/81))
        output = self.flat_densefc(output)
        output = self.relufc(output) 
        output = self.dropoutfc(output)
        output = self.densefc(output)
        # output = self.softfc(output)
        return output


checkpoint = torch.load(SAVE_NAME_ML, map_location = map_location)
model_2 = ConvNet( num_classes_2 = len(classes_2) )
model_2.load_state_dict(checkpoint)
model_2.eval()

transformer_2 = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize( (ROW_RESIZE, COL_RESIZE) ),
    T.ToTensor(), # Changes pixel range from 0-255 to 0-1 and from numpy to tensor
    T.Normalize([0.5], [0.5])
])
    

# Prediction_2 function
def prediction_2(window, real_image_name, transformer_2, 
               isEnd, hasDefect, typeDefect, hasClass):
    
    window2 = Image.fromarray(np.uint8(window) )
    
    image_tensor = transformer_2(window2).float()
    
    
    image_tensor = image_tensor.unsqueeze_(0)
    
    if torch.cuda.is_available():
        image_tensor.cuda()
    
    input = Variable(image_tensor)
    
    
    output = model_2(input)
    
    index = output.data.numpy().argmax()
    
    pred_2 = classes_2[index]
    
    if pred_2 != classes_2[1]:
        hasDefect = True
        typeDefect = pred_2
        for classIndex, className in enumerate(classes_2):
            if className in typeDefect:
                hasClass[classIndex] = True
    
    if hasDefect == True and isEnd == True:
        # Saves full image in appropriate class folder
        for listIndex, listName in enumerate(hasClass):
            if listName:
                print("  Defect image placed in", classes_2[listIndex] + "/" + real_image_name)
                save_image(transformed_image[:, ymin:ymax, xmin:xmax]/255, 
                            PREDICTED_PATH + classes_2[listIndex] + "/" + real_image_name)
                # cv2.imwrite(PREDICTED_PATH + classes_2[listIndex] + "/" + real_image_name, 
                #             image_b4_color_and_rotated[int(ymin*scale):int(ymax*scale), 
                #                                                  int(xmin*scale):int(xmax*scale)])
    # Saves no defect images if toggled on at top of this page
    elif hasDefect == False and isEnd == True:
        save_image(transformed_image[:, ymin:ymax, xmin:xmax]/255, 
                    PREDICTED_PATH + pred_2 + "/" + real_image_name)
        # cv2.imwrite(PREDICTED_PATH + pred_2 + "/" + real_image_name, 
        #             image_b4_color_and_rotated[int(ymin*scale):int(ymax*scale), 
        #                                                  int(xmin*scale):int(xmax*scale)])
    
    return pred_2, typeDefect, hasClass

# Create class folders in Predicted_Images folder
makeDir(PREDICTED_PATH, classes_2)
# -----------------------------------------------------------------------------


pred_dict = {}
for image_name in os.listdir(TO_PREDICT_PATH):
    image_path = os.path.join(TO_PREDICT_PATH, image_name)

    image_b4_color = cv2.imread(image_path)
    image_b4_color_and_rotated = cv2.rotate(image_b4_color, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)
    
    transformed_image = transforms_1(image=image)
    transformed_image = transformed_image["image"]
    
    with torch.no_grad():
        prediction_1 = model_1([(transformed_image/255).to(device)])
        pred_1 = prediction_1[0]
    
    # ML
    # -----------------------------------------------------------------------------
    hasDefect = False
    hasClass = []
    for classIndex in range(len(classes_2)):
        hasClass.append(False)
    typeDefect = 0
    # -----------------------------------------------------------------------------
    
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
            
            real_image_name = "-R_{}.C_{}.jpg".format(rowNumber, colNumber)
            # save_image(transformed_image[:, ymin:ymax, xmin:xmax]/255, 
            #             PREDICTED_PATH + image_name[:-4] + real_image_name)
            
            # Scans and categorizes window
            # ==================================================================================
            # ML Part
            # - - - - -
            scale = image_b4_color_and_rotated.shape[0]/IMAGE_SIZE
            pred_dict[image_path[image_path.rfind('/')+1:]] = prediction_2(image_b4_color_and_rotated[int(ymin*scale):int(ymax*scale), 
                                                             int(xmin*scale):int(xmax*scale)], 
                                                       real_image_name, 
                                                       transformer_2, 
                                                       True,
                                                       hasDefect,
                                                       typeDefect,
                                                       hasClass)
            pred_2, typeDefect, hasClass = pred_dict[image_path[image_path.rfind('/')+1:]]
            
            if pred_2 != classes_2[1]:
                hasDefect = True
            # ==================================================================================


print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)