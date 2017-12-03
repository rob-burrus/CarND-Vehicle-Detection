# Vehicle Detection and Tracking

## Overview
Software pipeline using traditional CV techniques to detect and draw bounding-boxes around vehicles in a video from a front-facing camera on a car. First, train a linear Support Vector Machine classifier on extracted features to identify car vs not-car. Then, process each frame of video using a sliding window technique, the trained classifier, and vehicle detections from previous frames. 

Images taken from GTI vehicle image dataset, the KITTI vision benchmark suite, and the Udacity highway videos

## Train Classifier

Example training image 

![original](./output_images/original.png) 

### 1. Spatial Binning

![spatial binning](./output_images/spatial_binning.png)

### 2. Color histogram
Y channel from YCrCb color space

![color hist](./output_images/color_hist.png)

### 3. Histogram of Oriented Gradients (HOG)

![hog](./output_images/hog.png)

### 4. Train classifier
Train Linear SVM Classifier

## Process Video Frames

### 1. Region of interest
Because we only expect cars to appear on the road, the lower half of the image frame serves as the region of interest (ROI). Compute HOG features on the ROI. Computing the HOG features once saves computation in the window sliding step, because no HOG features need be computed for every window.

### 2. Sliding windows
Search the region of interest at 3 different window sizes - 32x32, 64,64, 96x96. For each window:
* Sub-sample the HOG array for the current window
* Perform spatial binning and compute color histogram features
* Stack spatial binned, color, and HOG features together and use the SVM classifier to make a prediction. If the prediction is positive, use the current window positoin and size to draw a rectangle on the original image, and add "heat" to the heatmap


Results of sliding window search. Many duplicates for each car creates more "heat" on the heatmap. 

![window1](./output_images/window1.png)  ![heat1](./output_images/heat1.png) 

### 3. Combine previous heatmaps and draw final bounding box
After the sliding window technique has been applied at all 3 window sizes, combine the heatmap for the current image with the heatmaps from the previous 8 frames. Then apply a heatmap threshold to combined heatmap, and draw bounding boxes around each vehicle 

Labels are applied to the heatmap and final bounding boxes are drawn

![label1](./output_images/label1.png)  ![final1](./output_images/final1.png) 











