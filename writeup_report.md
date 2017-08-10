Vehicle Detection Project
--------------------------------

# 1. Dataset Prepaeration

## 1.1 Project Dataset provided by Udacity
Here are links to the labeled data for vehicle and non-vehicle examples to train your classifier. 

- https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip
- https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles_smallset.zip
- https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
- https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip

These example images come from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, 
- http://www.gti.ssr.upm.es/data/Vehicle_database.html
- http://www.cvlibs.net/datasets/kitti/
and examples extracted from the project video itself. 

- Udacity labeled dataset
You are welcome and encouraged to take advantage of the recently released Udacity labeled dataset to augment your training data.
- https://github.com/udacity/self-driving-car/tree/master/annotations

## 1.2 [CrowdAI](http://crowdai.com/) Dataset
imagedata http://bit.ly/udacity-annoations-crowdai

annotation list
https://github.com/udacity/self-driving-car/blob/master/annotations/labels_crowdai.csv

## 1.3 [Autti](http://autti.co/) Dataset
http://bit.ly/udacity-annotations-autti


## 1.4 Selecting Labeled Dataset

TODO ここから書く

具体的な対策予定：
1) 他の特徴量を加えて認識率を向上させる
2) スキャン方法と範囲を工夫して、FPと演算量を削減する
　かつスケーラを追加する
3) 既検出の自動車をトラッキングしてFN/FPを削減する
4) 目安、10FPSまで高速化する


データセットの比較の筆画

数、ラベル、備考

まずは smallsetで実験


### 1.4.1 


# 2. HOG feature and Other Features to Detect Vehicles


- For each channel and with your HOG parameters above, the HOG feature vector is 1764 long
- The spatial binning feature vector is 32 x 32 x 3 = 3072 long -> 16 x 16 x 3 = 768
- The histogram feature vector is 32 x 3 = 96
- Therefore, using only 1 HOG channel gives you a 1764 + 3072 + 96 = 4932 vector.
  This is one of the dimensions quoted by the error message.
- But your find_cars() function uses all 3 HOG channels -- see these lines:

Therefore the resulting HOG feature vector is 1764 x 3 = 5292. 
Add to that the spatial and histogram features and you get 5292 + 3072 + 96 = 8460, 
which is the other number quoted by the error message.




## Option
- color transform 
- binned color features
- histograms of color

# 3. Training with Linear SVM classifier
# 4. Sliding-Window Technique and Vehicle Tracking
# 5. Rejecting Outliers and follow detected vehicles.
Heat-map Creation of recurring detections frame by frame 
rejecting outliers and follow detected vehicles.
Estimate a bounding box for vehicles detected.

# 6. Run pipeline on a video stream
start with the test_video.mp4 and later implement on full project_video.mp4

## Udacity provided Video

test_video.mp4

project_video.mp4

## Hakone Video

60MB 比較的車が多い
LegacyVideo_05_40_20170725_135613.mp4

205MB 後半に工事車両
LegacyVideo_05_40_20170725_140502

621MB 霧、ワイパー
LegacyVideo_05_40_20170725_144407




# 7. Conclusion and Discussion



  ** Don't forget to normalize your features and randomize **

# Sample Writeup

## test_images:
Some example images for testing your pipeline on single frames are located in the test_images folder. 

## ouput_images
To help the reviewer examine your work, 
please save examples of the output from each stage of your pipeline in the folder called ouput_images, 
and include them in your writeup for the project by describing what each image shows. 

## input videos
The video called project_video.mp4 is the video your pipeline should work well on.

As an optional challenge Once you have a working pipeline for vehicle detection, 
add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

If you're feeling ambitious (also totally optional though), 
don't stop there! We encourage you to go out and take video of your own, 
and show us how you would implement this project on a new video!


[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


# Histogram of Oriented Gradients (HOG)

## 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

## 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

## 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

# Sliding Window Search

## 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

## 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

# Video Implementation

## 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


## 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

# Here are six frames and their corresponding heatmaps:

![alt text][image5]

# Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

# Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]




# Discussion

## 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

