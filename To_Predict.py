import os
import sys
import torch
from torchvision import models
import math
import re
import cv2
import albumentations as A  # our data augmentation library
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2

from torchvision.utils import save_image
import shutil


# User parameters
SAVE_NAME_OD = "./Models/Lord_of_Models-0.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models/",1)[1].split(".model",1)[0] +"/"
IMAGE_SIZE              = 800
TO_PREDICT_PATH         = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH          = "./Images/Prediction_Images/Predicted_Images/"
SAVE_ANNOTATED_IMAGES   = False
SAVE_ORIGINAL_IMAGE     = False
SAVE_CROPPED_IMAGES     = False
SAVE_LARGENED_CROPPED_IMAGES = False
BLACKENNED_NON_OBJ_IMG  = True
SPACING_SCALE       = 0.99
MIN_SCORE               = 0.6 # Default 0.5


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
model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                   box_detections_per_img=500,
                                                   min_size=1650,
                                                   max_size=2500
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

transforms_1 = A.Compose([
    # A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
    # A.Rotate(limit=[90,90], always_apply=True),
    ToTensorV2()
])

# Start FPS timer
fps_start_time = time.time()

color_list =['green', 'red', 'blue', 'magenta', 'orange', 'cyan', 'lime', 'turquoise', 'yellow']
pred_dict = {}
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
    
    # # DELETES NOT WANTED LABELS
    # for index, class_index in enumerate(class_indexes):
    #     if len(class_indexes) > 0:
    #         coordinates = coordinates[class_indexes == 2]
    #         scores = scores[class_indexes == 2]
    #         class_indexes = class_indexes[class_indexes == 2]
            
    
    if SAVE_ANNOTATED_IMAGES:
        coordinates = coordinates[class_indexes != 1]
        class_indexes = class_indexes[class_indexes != 1]
        
        # Filters out sizes below certain size
        class_indexes = class_indexes[abs(coordinates[:,2]-coordinates[:,0])>35]
        coordinates = coordinates[abs(coordinates[:,2]-coordinates[:,0])>35]
        # Now height
        class_indexes = class_indexes[abs(coordinates[:,3]-coordinates[:,1])>35]
        coordinates = coordinates[abs(coordinates[:,3]-coordinates[:,1])>35]
        
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
        
    if (SAVE_ORIGINAL_IMAGE and len(class_indexes) != 0
        and (len(coordinates[class_indexes == 2]) != 0
              or len(coordinates[class_indexes == 3]) != 0
              )
        ):
        
        # Filters out sizes below certain size
        class_indexes = class_indexes[abs(coordinates[:,2]-coordinates[:,0])>35]
        coordinates = coordinates[abs(coordinates[:,2]-coordinates[:,0])>35]
        # Now height
        class_indexes = class_indexes[abs(coordinates[:,3]-coordinates[:,1])>35]
        coordinates = coordinates[abs(coordinates[:,3]-coordinates[:,1])>35]
        
        cv2.imwrite(PREDICTED_PATH + image_name, orig_image)
    
    # Saves image of cropped widened-boxed objects 
    #  - Uncomment and add interested only classes/labels
    if (SAVE_LARGENED_CROPPED_IMAGES 
        and (len(coordinates[class_indexes == 2]) != 0
              or len(coordinates[class_indexes == 3]) != 0
              )
        and len(class_indexes) != 0
        ):
        
        # Filters out sizes below certain size
        class_indexes = class_indexes[abs(coordinates[:,2]-coordinates[:,0])>35]
        coordinates = coordinates[abs(coordinates[:,2]-coordinates[:,0])>35]
        # Now height
        class_indexes = class_indexes[abs(coordinates[:,3]-coordinates[:,1])>35]
        coordinates = coordinates[abs(coordinates[:,3]-coordinates[:,1])>35]
        
        # Recreates coordinates with interested classes/labels
        cat_1 = coordinates[class_indexes == 2]
        cat_2 = coordinates[class_indexes == 3]
        coordinates = torch.cat((cat_1, cat_2), 0)
        
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
    if (BLACKENNED_NON_OBJ_IMG 
        and (len(coordinates[class_indexes == 2]) != 0
              or len(coordinates[class_indexes == 3]) != 0
              )
        and len(class_indexes) != 0
        ):
        
        # Filters out sizes below certain size
        class_indexes = class_indexes[abs(coordinates[:,2]-coordinates[:,0])>35]
        coordinates = coordinates[abs(coordinates[:,2]-coordinates[:,0])>35]
        # Now height
        class_indexes = class_indexes[abs(coordinates[:,3]-coordinates[:,1])>35]
        coordinates = coordinates[abs(coordinates[:,3]-coordinates[:,1])>35]
        
        # Recreates coordinates with interested classes/labels
        cat_1 = coordinates[class_indexes == 2]
        cat_2 = coordinates[class_indexes == 3]
        coordinates_cat = torch.cat((cat_1, cat_2), 0)
        
        if len(coordinates_cat) > 0:
        
            box_height_all = int(max(coordinates_cat[:, 3])) - int(min(coordinates_cat[:, 1]))
            box_width_all = int(max(coordinates_cat[:, 2])) - int(min(coordinates_cat[:, 0]))
            
            # Calculates what values to widen box to crop
            y_to_add = int( -101*(box_height_all/transformed_image.shape[1])+101 )
            x_to_add = int( -101*(box_width_all/transformed_image.shape[2])+101 )
            # y_to_add = 0
            # x_to_add = 0
            
            y_min = max(int(min(coordinates_cat[:, 1]))-y_to_add, 
                        0
                        )
            y_max = min(int(max(coordinates_cat[:, 3]))+y_to_add, 
                        transformed_image.shape[1]
                        )
            x_min = max(int(min(coordinates_cat[:, 0]))-x_to_add, 
                        0
                        )
            x_max = min(int(max(coordinates_cat[:, 2]))+x_to_add, 
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
        # Grabs row and column number from image name and corrects them
        path_row_number = int( re.findall(r'\d+', image_name)[0] )
        path_col_number = int( re.findall(r'\d+', image_name)[1] )
        
        if "SMiPE4" in SAVE_NAME_OD.split("./Models-OD/",1)[1].split("-",1)[0]:
            if path_row_number % 2 == 1:
                path_row_number = (path_row_number + 1) // 2
            else:
                path_row_number = 20 + path_row_number // 2
            
            if path_col_number % 2 == 1:
                path_col_number = (path_col_number + 1) // 2
            else:
                path_col_number = 20 + path_col_number // 2  
        elif "TPv2" in SAVE_NAME_OD.split("./Models-OD/",1)[1].split("-",1)[0]:
            path_part_number = int( re.findall(r'\d+', image_name)[2] )
            row_grouping = (10-path_part_number)%10
            col_grouping = (path_part_number-1)//10
        
        if len(coordinates) > 0:
            box_width = int(coordinates[0][2]-coordinates[0][0]) 
            box_height = int(coordinates[0][3]-coordinates[0][1])
        
        # # Sets spacing between objects
        if "SMiPE4" in SAVE_NAME_OD.split("./Models-OD/",1)[1].split("-",1)[0]:
            spacing_max = int(box_width * .1) # I guessed
            spacing = 1 + round( (spacing_max/box_width)*SPACING_SCALE, 3)
        elif "TPv2" in SAVE_NAME_OD.split("./Models-OD/",1)[1].split("-",1)[0]:
            spacing_max = int(box_width * 5) # I guessed
            spacing = 1 + round( (spacing_max/box_width)*SPACING_SCALE, 3)
        
        # Grabbing max and min x and y coordinate values
        if len(coordinates) > 0:
            minX = int( torch.min(coordinates[:, 0]) )
            minY = int( torch.min(coordinates[:, 1]) )
            maxX = int( torch.max(coordinates[:, 2]) )
            maxY = int( torch.max(coordinates[:, 3]) )
        
        names = []
        
        # Changes column names in names
        for box_index in range(len(coordinates)):
            
            x1 = int( coordinates[box_index][0] )
            y1 = int( coordinates[box_index][1] )
            x2 = int( coordinates[box_index][2] )
            y2 = int( coordinates[box_index][3] )
            
            midX = round((x1 + x2)/2)
            midY = round((y1 + y2)/2)
            
            # Creates names list row and column number
            rowNumber = math.floor((y1-minY)/(box_width*spacing)+1)
            rowNumber = str(rowNumber)
            colNumber = math.floor((x1-minX)/(box_height*spacing)+1)
            colNumber = str(colNumber)
            
            if "SMiPE4" in SAVE_NAME_OD.split("./Models-OD/",1)[1].split("-",1)[0]:
                real_rowNum = (path_row_number - 1)*10 + int(rowNumber)
                real_colNum = (path_col_number - 1)*10 + int(colNumber)
            elif "TPv2" in SAVE_NAME_OD.split("./Models-OD/",1)[1].split("-",1)[0]:
                real_rowNum = (path_row_number - 1)*40 + row_grouping*4 + int(rowNumber)
                real_colNum = (path_col_number - 1)*40 + col_grouping*4 + int(colNumber)
            
            # THIS PART IS FOR LED 160,000 WAFER!
            if int(real_colNum)>200:
                real_colNum = str( int(real_colNum) )
            
            if int(real_colNum) < 10:
                real_colNum = "00" + str(real_colNum)
            elif int(real_colNum) < 100:
                real_colNum = "0" + str(real_colNum)
            
            if int(real_rowNum)>200:
                real_rowNum = str( int(real_rowNum) )
            
            if int(real_rowNum) < 10:
                real_rowNum = "00" + str(real_rowNum)
            elif int(real_rowNum) < 100:
                real_rowNum = "0" + str(real_rowNum)
            
            names.append( "R_{}.C_{}".format(real_rowNum, real_colNum) )
            
            xmin = int(coordinates[box_index][0])
            ymin = int(coordinates[box_index][1])
            xmax = int(coordinates[box_index][2])
            ymax = int(coordinates[box_index][3])
            
            real_image_name = "R_{}.C_{}.jpg".format(real_rowNum, real_colNum)
            save_image(transformed_image[:, ymin:ymax, xmin:xmax]/255, 
                        PREDICTED_PATH + real_image_name)
    
    
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