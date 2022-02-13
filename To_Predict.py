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


# User parameters
SAVE_NAME = "./led.model"
USE_CHECKPOINT = True
IMAGE_SIZE = 2180 # Row and column number
DATASET_PATH = "./led_dies/"
NUMBER_EPOCH = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 16



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




image = cv2.imread("./Images/To_Predict/Row_01.Col_28.P_01.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
    A.Rotate(limit=[90,90], always_apply=True),
    ToTensorV2()
])

# transforms=get_transforms(False)
transformed_image = transforms(image=image)
transformed_image = transformed_image["image"]

with torch.no_grad():
    prediction = model([(transformed_image/255).to(device)])
    pred = prediction[0]

test_image = draw_bounding_boxes(transformed_image,
    pred['boxes'][pred['scores'] > 0.8],
    [classes[i] for i in pred['labels'][pred['scores'] > 0.8].tolist()], 
    width=4
    )
from torchvision.utils import save_image
save_image((test_image/255), "./Transformed_Images-Test.jpg")

xmin = int(pred['boxes'][pred['scores'] > 0.8][0][0])
ymin = int(pred['boxes'][pred['scores'] > 0.8][0][1])
xmax = int(pred['boxes'][pred['scores'] > 0.8][0][2])
ymax = int(pred['boxes'][pred['scores'] > 0.8][0][3])

save_image((transformed_image[:, ymin:ymax, xmin:xmax]/255), "./Transformed_Images-Cropped.jpg")




# img, _ = test_dataset[0]
# # img, _ = train_dataset[0]
# img_int = torch.tensor(img*255, dtype=torch.uint8)
# with torch.no_grad():
#     prediction = model([img.to(device)])
#     pred = prediction[0]

# from torchvision.utils import save_image
# save_image(img.float(), "./Transformed_Images.jpg")


# fig = plt.figure(figsize=(25, 25))
# plt.imshow(draw_bounding_boxes(img_int,
#     pred['boxes'][pred['scores'] > 0.8],
#     [classes[i] for i in pred['labels'][pred['scores'] > 0.8].tolist()], 
#     width=4
#     ).permute(1, 2, 0))


