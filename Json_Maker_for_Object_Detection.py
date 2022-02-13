# Import the necessary packages
import os
import glob
import imutils
import cv2
import time
# TESTING SVD FROM NUMPY
import numpy as np
import math

# User Parameters/Constants to Set
MATCH_CL = 0.60 # Minimum confidence level (CL) required to match golden-image to scanned image
# STICHED_IMAGES_DIRECTORY = "//mcrtp-sftp-01/aoitool/LED-Test/Slot_01/"
# GOLDEN_IMAGES_DIRECTORY = "C:/Users/ait.lab/.spyder-py3/Automated_AOI/Golden_Images/"
STICHED_IMAGES_DIRECTORY = "Images/Stitched_Images/"
GOLDEN_IMAGES_DIRECTORY = "Images/Golden_Images/"
SLEEP_TIME = 0.00 # Time to sleep in seconds between each window step
RENAME_TOGGLE = False
SHOW_WINDOW = False


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


# Deletes unnecessary string in file name
def replaceFileName(slotDir):
    for filename in glob.glob(slotDir + "/*"):
        # For loop with row number as "i" will take longer, so yes below seems
        #   redundant writing each number 1 by 1, but has to be done.
        os.rename(filename, 
                  filename.replace("Stitcher-Snaps_for_8in_Wafer_Pave.", "")\
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


def slidingWindow(fullImage, stepSizeX, stepSizeY, windowSize):
    # Slides a window across the stitched-image
    for y in range(0, fullImage.shape[0], stepSizeY):
        for x in range(0, fullImage.shape[1], stepSizeX):
            # Yield the current window
            yield (x, y, fullImage[y:y + windowSize[1], x:x + windowSize[0]])


# Comparison scan window-image to golden-image
def getMatch(window, goldenImage, x, y):
    h1, w1, c1 = window.shape
    h2, w2, c2 = goldenImage.shape
    
    if c1 == c2 and h2 <= h1 and w2 <= w1:
        method = eval('cv2.TM_CCOEFF_NORMED')
        res = cv2.matchTemplate(window, goldenImage, method)   
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > MATCH_CL: 
            print("\nFOUND MATCH")
            print("max_val = ", max_val)
            print("Window Coordinates: x1:", x + max_loc[0], "y1:", y + max_loc[1], \
                  "x2:", x + max_loc[0] + w2, "y2:", y + max_loc[1] + h2)
            
            # Gets coordinates of cropped image
            return (max_loc[0], max_loc[1], max_loc[0] + w2, max_loc[1] + h2, max_val)
        
        else:
            return ("null", "null", "null", "null", "null")


# MAIN():
# =============================================================================
# Starting stopwatch to see how long process takes
start_time = time.time()

# Clears some of the screen for asthetics
print("\n\n\n\n\n\n\n\n\n\n\n\n\n")

# Replaces names
if RENAME_TOGGLE:
    replaceFileName(STICHED_IMAGES_DIRECTORY)

goldenImagePath = glob.glob(GOLDEN_IMAGES_DIRECTORY + "*")
goldenImage = cv2.imread(goldenImagePath[0])
goldenImage = cv2.rotate(goldenImage, cv2.ROTATE_90_COUNTERCLOCKWISE)



# Parameter set
winW = round(goldenImage.shape[1] * 1.5) # Scales window width with full image resolution
# BELOW DEFAULT IS 1.5 CHANGE BACK IF NEEDED
winH = round(goldenImage.shape[0] * 1.5) # Scales window height with full image resolution
windowSize = (winW, winH)
stepSizeX = round(winW / 2.95)
stepSizeY = round(winH / 2.95)

# Predefine next for loop's parameters 
prev_y1 = stepSizeY * 9 # Number that prevents y = 0 = prev_y1
prev_x1 = stepSizeX * 9
rowNum = 0
colNum = 0
prev_matchedCL = 0
image_names = []
image_ids = []
image_height = []
image_width = []
die_index = 0

for image_index, image_name in enumerate(os.listdir(STICHED_IMAGES_DIRECTORY)):
    
    # TESTING - Only completes up to index 4
    if image_index == 5:
        break
    
    # For Json file
    image_names.append(image_name)
    image_ids.append(image_index)
    
    fullImagePath = os.path.join(STICHED_IMAGES_DIRECTORY, image_name)
    
    fullImage = cv2.imread(fullImagePath)
    fullImage = cv2.rotate(fullImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # For Json file
    image_height.append(fullImage.shape[0])
    image_width.append(fullImage.shape[1])
    
    cv2.destroyAllWindows()

    # Get's path's actual column and row number
    path_row_number = int(fullImagePath[-18:-16])
    path_col_number = int(fullImagePath[-11:-9])
    if path_row_number % 2 == 1:
        path_row_number = (path_row_number + 1) // 2
    else:
        path_row_number = 20 + path_row_number // 2
    
    if path_col_number % 2 == 1:
        path_col_number = (path_col_number + 1) // 2
    else:
        path_col_number = 20 + path_col_number // 2    
    
    # Adding list and arrray entry
    dieNames = ["Row_#.Col_#"]
    die_ids = []
    die_image_ids = []
    category_id = []
    bboxes = np.zeros([1, 4], np.int32)
    bbox_areas = []
    die_segmentations = []
    die_iscrowd = []

    # loop over the sliding window
    for (x, y, window) in slidingWindow(fullImage, stepSizeX, stepSizeY, windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        
        # Draw rectangle over sliding window for debugging and easier visual
        displayImage = fullImage.copy()
        cv2.rectangle(displayImage, (x, y), (x + winW, y + winH), (255, 0, 180), 2)
        displayImageResize = cv2.resize(displayImage, (1000, round(fullImage.shape[0] / fullImage.shape[1] * 1000)))
        if SHOW_WINDOW:
            cv2.imshow(str(fullImagePath), displayImageResize) # TOGGLE TO SHOW OR NOT
        cv2.waitKey(1)
        time.sleep(SLEEP_TIME) # sleep time in ms after each window step
        
        # Scans window for matched image
        # ==================================================================================
        # Scans window and grabs cropped image coordinates relative to window
        # Uses each golden image in the file if multiple part types are present
        for goldenImagePath in glob.glob(GOLDEN_IMAGES_DIRECTORY + "*"):
            goldenImage = cv2.imread(goldenImagePath)
            goldenImage = cv2.rotate(goldenImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Gets coordinates relative to window of matched dies within a Stitched-Image
            win_x1, win_y1, win_x2, win_y2, matchedCL = getMatch(window, goldenImage, x, y)
            
            # Saves cropped image and names with coordinates
            if win_x1 != "null":
                # Turns cropped image coordinates relative to window to stitched-image coordinates
                x1      = x + win_x1
                y1      = y + win_y1
                x2      = x + win_x2
                y2      = y + win_y2
                bbox_width   = x2 - x1
                bbox_height  = y2 - y1
                bbox_area    = bbox_width * bbox_height
                
                # Makes sure same image does not get saved as different names
                if y1 >= (prev_y1 + round(goldenImage.shape[0] / 4)) or y1 <= (prev_y1 - round(goldenImage.shape[0] / 4)):
                    rowNum += 1
                    colNum = 1
                    sameCol = False
                else:
                    if x1 >= (prev_x1 + round(goldenImage.shape[1] / 4)) or x1 <= (prev_x1 - round(goldenImage.shape[1] / 4)):
                        colNum += 1
                        prev_matchedCL = 0
                        sameCol = False
                    else: 
                        sameCol = True
                
                if sameCol == False: 
                    die_index += 1
                    die_ids.append(die_index)
                    die_image_ids.append(image_ids[-1]) # image_id to place in annotations category
                    category_id.append(1)
                    bboxes = np.append(bboxes, [[x1, y1, bbox_width, bbox_height]], axis=0)
                    bbox_areas.append(bbox_area)
                    die_segmentations.append("")
                    die_iscrowd.append(0)
                    
                elif sameCol == True and matchedCL > prev_matchedCL:
                    bboxes[len(bboxes)-1] = np.array([x1, y1, 
                                                      bbox_width, bbox_height], 
                                                     ndmin=2)
                    bbox_areas[-1] = bbox_area
                
                prev_y1 = y1
                prev_x1 = x1
                if sameCol == True and matchedCL > prev_matchedCL:
                    prev_matchedCL = matchedCL
                
            # ==================================================================================
    rowNum = 0
    colNum = 0
    sameCol = False










print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)