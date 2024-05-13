# Furniture Detection with YOLOv8

This project uses the YOLOv8 model to detect specific furniture items (chair, couch, bed, dining table) in an image, and generates black and white masks for each detected item. The masks are then saved as PNG files in a specified directory.

## Table of Contents

- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Usage Instructions](#usage-instructions)
- [Code Explanation](#code-explanation)
- [Evaluation Criteria](#evaluation-criteria)

## Requirements

- Python 3.x
- Ultralytics YOLOv8
- OpenCV
- PyTorch
- TorchVision
- NumPy

## Setup Instructions

1. **Clone the repository and navigate to the project directory or download the detect_furniture.py file:**

   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install the required Python packages:**

   ```sh
   pip install ultralytics opencv-python-headless torch torchvision numpy
   ```

3. **Run the initial setup to ensure Ultralytics is properly installed:**

   ```python
   from IPython import display
   display.clear_output()
   import ultralytics
   ultralytics.checks()
   ```

## Usage Instructions

1. **Ensure the working directory and image path are set correctly in the script:**

   ```python
   # Set the directory to save the results
   home = 'D:/Data Science Journey/Computer Vision/Main File'

   # Set the path to the image you want to predict
   image_path = 'D:/Data Science Journey/Computer Vision/Main File/best-furniture-for-your-home-2022-section-1.jpg'
   ```

2. **Run the script to perform the detection and save masks:**

   ```sh
   python detect_furniture.py
   ```

3. **Check the specified directory for the saved results and masks:**

   The results, including masks, will be saved in the directory specified by the `home` variable.

## Code Explanation

### Main Script: `detect_furniture.py`

   ```python
   import os
   from ultralytics import YOLO
   import numpy as np
   import cv2
   from torchvision.transforms.functional import to_pil_image
   import torch

   # Set the directory to save the results
   home = os.getcwd()
   home = 'D:/Data Science Journey/Computer Vision/Main File'

   # Create the YOLOv8n model 
   model = YOLO('yolov8n-seg.pt')

   # Ensure the model runs on CPU
   model = model.to(torch.device('cpu'))

   # Class indexes for the furniture items to detect
   class_indexes = [56, 57, 59, 60]

   # Path to the image to predict
   image_path = 'D:/Data Science Journey/Computer Vision/Main File/best-furniture-for-your-home-2022-section-1.jpg'

   # Perform prediction
   results = model.predict(image_path, save=True, save_txt=True, save_crop=True,
                           exist_ok=True, classes=class_indexes, show=True,
                           project=home)

   # Number of detected objects
   nr_objects_detected = results[0].masks.shape[0]

   # Save the masks for each detected object
   for i in range(nr_objects_detected):
       img = to_pil_image(results[0].masks.cpu().data[i])
       img = np.array(img)
       binary_mask = (img > 0.5).astype(np.uint8) * 255
       cv2.imwrite(home + f'/mask{i}.png', binary_mask)
   ```

### Explanation

1. **Environment Setup:**
   - Ensure Ultralytics is installed and properly set up.

2. **Directory and Model Initialization:**
   - Set the working directory for saving results.
   - Initialize the YOLOv8 model and ensure it runs on the CPU.

3. **Prediction:**
   - Specify the classes for detection (`chair`, `couch`, `bed`, `dining table`).
   - Perform prediction on the specified image, saving results and masks.

4. **Mask Generation and Saving:**
   - Loop through each detected object, generate a binary mask, and save it as a PNG file.

By following these instructions, you can set up the environment, run the detection script, and save the masks for the specified furniture items in the image.
