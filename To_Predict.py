import os
import sys
import torch
from torchvision import models
import math
import re
import cv2
import albumentations as A  # our data augmentation library
import time
from torchvision.utils import draw_bounding_boxes
from albumentations.pytorch import ToTensorV2

from torchvision.utils import save_image
import shutil
import json


# User parameters
SAVE_NAME_OD = "./Models/Lord_of_Models-0.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/",1)[1].split(".model",1)[0] +"/"
MIN_IMAGE_SIZE          = 800 # Minimum size of image (ASPECT RATIO IS KEPT SO DONT WORRY). So for 1600x2400 -> 800x1200
TO_PREDICT_PATH         = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH          = "./Images/Prediction_Images/Predicted_Images/"
SAVE_ANNOTATED_IMAGES   = True
SAVE_ORIGINAL_IMAGE     = False
SAVE_CROPPED_IMAGES     = False
SAVE_LARGENED_CROPPED_IMAGES = False
BLACKENNED_NON_OBJ_IMG  = False
MIN_SCORE               = 0.6 # Default 0.6


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
def makeDir(dir, classes_1):
    for classIndex, className in enumerate(classes_1):
        os.makedirs(dir + className, exist_ok=True)



# Starting stopwatch to see how long process takes
start_time = time.time()

# Deletes images already in "Predicted_Images" folder
deleteDirContents(PREDICTED_PATH)

dataset_path = DATASET_PATH


f = open(os.path.join(dataset_path, "train", "_annotations.coco.json"))
data = json.load(f)
n_classes_1 = len(data['categories'])
classes_1 = [i['name'] for i in data["categories"]]


# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                   min_size=MIN_IMAGE_SIZE,
                                                   max_size=MIN_IMAGE_SIZE*3
                                                   )
in_features = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes_1)


# Loads last saved checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

if os.path.isfile(SAVE_NAME_OD):
    checkpoint = torch.load(SAVE_NAME_OD, map_location=map_location)
    model_1.load_state_dict(checkpoint)

model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

transforms_1 = A.Compose([ToTensorV2()])

# Start FPS timer
fps_start_time = time.time()

color_list =['green', 'red', 'blue', 'magenta', 'orange', 'cyan', 'lime', 'turquoise', 'yellow']
ii = 0
for image_name in os.listdir(TO_PREDICT_PATH):
    image_path = os.path.join(TO_PREDICT_PATH, image_name)
    
    image_b4_color = cv2.imread(image_path)
    orig_image = image_b4_color
    image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)
    
    transformed_image = transforms_1(image=image)
    transformed_image = transformed_image["image"]
    
    if ii == 0:
        line_width = max(round(transformed_image.shape[1] * 0.002), 1)
    
    with torch.no_grad():
        prediction_1 = model_1([(transformed_image/255).to(device)])
        pred_1 = prediction_1[0]
    
    coordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
    class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE]
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE]
            
    
    if SAVE_ANNOTATED_IMAGES:
        
        predicted_image = draw_bounding_boxes(transformed_image,
            boxes = coordinates,
            # labels = [classes_1[i] for i in class_indexes], 
            # labels = [str(round(i,2)) for i in scores], # SHOWS SCORE IN LABEL
            width = line_width,
            colors = [color_list[i] for i in class_indexes],
            font = "arial.ttf",
            font_size = 20
            )
        
        # Saves full image with bounding boxes
        if len(class_indexes) != 0:
            save_image((predicted_image/255), PREDICTED_PATH 
                       + image_name.replace(".jpg","") + "-Annot.jpg")
        
        # save_image((predicted_image/255), PREDICTED_PATH + image_name)
        
    if (SAVE_ORIGINAL_IMAGE and len(class_indexes) != 0):
        
        cv2.imwrite(PREDICTED_PATH + image_name, orig_image)
    
    # Saves image of cropped widened-boxed objects 
    #  - Uncomment and add interested only classes/labels
    if (SAVE_LARGENED_CROPPED_IMAGES and len(class_indexes) != 0):
        
        box_height_all = int(max(coordinates[:, 3])) - int(min(coordinates[:, 1]))
        box_width_all = int(max(coordinates[:, 2])) - int(min(coordinates[:, 0]))
        
        # Calculates what values to widen box to crop
        y_to_add = int( -101*(box_height_all/transformed_image.shape[1])+101 )
        x_to_add = int( -101*(box_width_all/transformed_image.shape[2])+101 )
        
        y_min = max(int(min(coordinates[:, 1]))-y_to_add, 
                    0
                    )
        y_max = min(int(max(coordinates[:, 3]))+y_to_add, 
                    transformed_image.shape[1]
                    )
        x_min = max(int(min(coordinates[:, 0]))-x_to_add, 
                    0
                    )
        x_max = min(int(max(coordinates[:, 2]))+x_to_add, 
                    transformed_image.shape[2]
                    )
        
        save_image(transformed_image[:, 
                                     y_min:y_max, 
                                     x_min:x_max
                                     ]/255, 
                    PREDICTED_PATH + image_name.replace(".jpg","") + "-Largen_Crop.jpg")
        
    
    # Saves image of Original image that is blackened where no object is at
    #  - Uncomment and add interested only classes/labels
    if (BLACKENNED_NON_OBJ_IMG and len(class_indexes) != 0):
        
        if len(coordinates) > 0:
        
            box_height_all = int(max(coordinates[:, 3])) - int(min(coordinates[:, 1]))
            box_width_all = int(max(coordinates[:, 2])) - int(min(coordinates[:, 0]))
            
            # Calculates what values to widen box to crop
            y_to_add = int( -101*(box_height_all/transformed_image.shape[1])+101 )
            x_to_add = int( -101*(box_width_all/transformed_image.shape[2])+101 )
            # y_to_add = 0
            # x_to_add = 0
            
            y_min = max(int(min(coordinates[:, 1]))-y_to_add, 
                        0
                        )
            y_max = min(int(max(coordinates[:, 3]))+y_to_add, 
                        transformed_image.shape[1]
                        )
            x_min = max(int(min(coordinates[:, 0]))-x_to_add, 
                        0
                        )
            x_max = min(int(max(coordinates[:, 2]))+x_to_add, 
                        transformed_image.shape[2]
                        )
            
            blackened_image = transformed_image.detach().clone()
            blackened_image[:, :y_min, :] = 0
            blackened_image[:, :, :x_min] = 0
            blackened_image[:, y_max:-1, :-1] = 0
            blackened_image[:, :-1, x_max:-1] = 0
            
            
            save_image(blackened_image/255, 
                        PREDICTED_PATH + image_name.replace(".jpg","") + "-Blackenned.jpg")
    
    
    if SAVE_CROPPED_IMAGES:
        
        for box_index in range(len(coordinates)):
            
            xmin = int(coordinates[box_index][0])
            ymin = int(coordinates[box_index][1])
            xmax = int(coordinates[box_index][2])
            ymax = int(coordinates[box_index][3])
            
            save_image(transformed_image[:, ymin:ymax, xmin:xmax]/255, 
                        PREDICTED_PATH + image_name.replace(".jpg","") + "-{}-Cropped.jpg".format(box_index))
    
    
    ten_scale = int(len(os.listdir(TO_PREDICT_PATH))*0.01)
    
    ii += 1
    if ii % ten_scale == 0:
        fps_end_time = time.time()
        fps_time_lapsed = fps_end_time - fps_start_time
        fps = round(ten_scale/fps_time_lapsed, 2)
        percent_progress = round(ii/len(os.listdir(TO_PREDICT_PATH))*100)
        images_left = len(os.listdir(TO_PREDICT_PATH)) - ii
        
        time_left = images_left/(fps) # in seconds
        mins = time_left // 60
        sec = time_left % 60
        
        sys.stdout.write('\033[2K\033[1G')
        print("  " + str(percent_progress) + "%",
              "-",  fps, "FPS -",
              "Time Left: {0}m:{1}s".format(int(mins), round(sec) ),
              end="\r"
              )
        fps_start_time = time.time()


print() # Since above print tries to write in last line used, this one clears it up
print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)