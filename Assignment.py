# Run this code at the start to make sure you have Ultralytics installed. 
# from IPython import display
# display.clear_output()
# import ultralytics
# ultralytics.checks()
import os
from ultralytics import YOLO
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image
import torch

# Get the directory that we are working on, to save the results
home = os.getcwd()
home = 'D:\Data Science Journey\Computer Vision\Main File'
# Create the YOLOv8n model 
model = YOLO('yolov8n-seg.pt')

# Ensuring that the model is running on the cpu.
model = model.to(torch.device('cpu'))

# Get the indexes of the classes that we want to detect:- chair, couch, bed, dining table
class_indexes = [56, 57, 59, 60]

# The access to the image that we want to predict.
image_path = 'D:/Data Science Journey/Computer Vision/Main File/best-furniture-for-your-home-2022-section-1.jpg' # Put here the path of the image that you want to predict
# We make a prediction of a specific specified photo, with the arguments that we want to detect specific classes and confidence of 50%
results = model.predict(image_path, save = True, save_txt= True, save_crop = True,
                exist_ok = True, 
                classes = [56, 57, 59, 60], show = True,
                project = home) #The argument project allows you to save the results wherever you want it to be.

nr_objects_detected = results[0].masks.shape[0] # We obtained the number of detected objects
# Here is a for loop that will save the mask of each detected object.
for i in range(nr_objects_detected): # We loop over the number of detected images
    # We then get the mask of each image at a time
    img = to_pil_image(results[0].masks.cpu().data[i])
    # We then convert it to a numpy array so we can save it
    img = np.array(img)
    # We first perform binary masking
    binary_mask = (img > 0.5).astype(np.uint8) * 255
    # We then save the mask to local file. 
    # You can choose the location where you want to save the mask by replacing the home variable below with the path you want.
    cv2.imwrite(home + f'\mask{i}.png', binary_mask) 





