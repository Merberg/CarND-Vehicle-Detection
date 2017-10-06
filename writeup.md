## Vehicle Detection Project

---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Additionally apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/1_false_positives.png
[image2]: ./output_images/1_features.png
[image3]: ./output_images/2_window_frames.png
[image4]: ./output_images/2_windows_and_scale.png
[image5]: ./output_images/2_thresholded.png
[image6]: ./output_images/3_history.png
[video1]: ./output_images/project_results.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
---
### Files
- **Writeup** _(This file)_: Provides a summary and reasoning for the techniques used in the vehicle pipeline.
- **lane_finding.py**: Functions for marking the lanes (from previous project)
- **plot_images.py**: Functions for plotting and visualizing data.
- **vehicle_features.py**: Functions that perform the feature extractions on images.
-  **vehicle_classifier.ipynb**: Main project notebook that contains the vehicle detection pipeline.
-  **output_images folder/project_results.mp4**: marked-up video with the results

### Histogram of Oriented Gradients (HOG)

#### 1. Features Extraction

The following features are used, in the listed order, for training and prediction:
1. HOG
2. Color spatial binning/down sampling
3. Color channel histograms

The code that executes features extraction is housed in vehicle_features.py.  The images, training or video frames, are first converted to the YCrCb color space.  The first channel of this color space is the light intensity, while channels two and three separate the primary colors.  This division allows for sharper divisions of black-and-white while also providing processing on the chroma channels.  This color space plays into the accuracy of the the color channel histograms; the accuracy, determined via testing, was maximized using YCrCb with an acceptable duration.

###### Color Space Accuracy and Timing

| Color Space | 1 | 2 | 3 | All |Train Time All (sec)|
|-------------|---|---|---|-----|--------------------|
| **YCrCb** | 0.95 | 0.97 | 0.96 | **0.99** | **6.16**|
| HSV   | 0.96 | 0.90 | 0.95 | 0.98 | 6.45|
| LUV   | 0.95 | 0.96 | 0.95 | 0.98 | 6.79|
| HLS   | 0.96 | 0.95 | 0.92 | 0.96 | 6.31|
| YUV   | 0.96 | 0.97 | 0.95 | 0.98 | 6.22|

Additional tuning was performed for the binning and histogram parameters:

###### Spatial Binning and Histogram Parameter Accuracy

|| Spatial Bin Size | # Histogram Bins | Train Time | Accuracy|
|-|---------|----|------|-------|
|1| (32,32) | 32 | 2.09 | 0.9666|
|2| (32,32) | 16 | 2.51 | 0.9520|
|3|**(24,24)** | **40** | **0.84** | **0.9833**|
|4| (16,16) | 40 | 0.35 | 0.9854|
|5| (16,16) | 32 | 0.30 | 0.9896|
|6| (8,8)   | 40 | 0.10 | 0.9854|
|7| (8,8)   | 32 | 0.09 | 0.9875|
|8| (8,8)   | 16 | 0.12 | 0.9624|

Rows 6-8 highlight promising values, however in operation these features produced numerous false positives.

![False Positive][image1]

From the theory behind Histogram of Oriented Gradients, the orientation is effective between 9-12, the number of cells per block is kept at a minumum (usually 2), and the pixels per cell range between 4-8.  To minimize time and maximize accuracy, these parameters where selected to produce the displayed results:

```
HOG_ORIENT = 9
HOG_PIX_PER_CELL = 8
HOG_CELLS_PER_BLOCK = 2
```

![HOG Features][image2]

In vehicle_classifier.ipynb, the `Training the SVC` cell houses the code which yields the following:
```
Training classifier:
	* 7140 feature vectors
	* Duration 15.0747 seconds
	* Accuracy of 0.9842
```


### 2. Sliding Window Search

In the `Sliding Windows` cell, the `findCars` functions stays fairly true to the technique taught in the lesson with one exception; the window was extended to cover the entire range provided.  This helped pick up cars to the right of the ego vehicle once they entered the camera's field of view.
Three windows are applied to each image at different scales:
```
WINDOW_SCALES = [1.0, 1.5, 2.0]
Y_STARTS = [375, 375, 375]
Y_STOPS = [500, 550, 600]
```
The coverage of the frames is visualized below:

![Window Frames][image3]

The different scales enable the detector to pick up different features.  For example, look at the variation in the coverage of the white car in the images below.

![Window Scales][image4]

Because the multiple window scales add numerous false positives, a `HEAT_THRESHOLD = len(WINDOW_SCALES)+1` needs to be applied.  The bottom images highlight the reduction of false positives with the threshold application.

![Threshold][image5]

The blue boxes in the right images above are drawn in the `determineBoundingBoxes` function.  It makes use of the scipy label function to cluster together nonzero heatmap values.  Boxes must be wider than 40 pixels to be valid (false positives reduction).

---

### 3. Vehicle Detection Pipeline (Video)
_Note: Lane Detection from the previous project was incorporated on the pipeline to hightlight the path of travel._

#### Detection History
The TrackHeatmap class in the `Vehicle Detection History` cell performs exponential smoothing (EMA) on consecutive images, with a slight bias towards the historic heatmap. EMA is performed before the heatmap threshold is applied.  These images show this moving average scheme:

![EMA Images][image6]

The `addVehiclesHeat` method of TrackHeatmap is used to strengthen the presence of the bounded boxes drawn on the image for the next round of detection, which has a stabilizing effect on the boxes.

#### Pipeline
This sequence forms the final pipeline:

```
heatmapper = TrackHeatmap(HEAT_THRESHOLD)
leftLane = LaneLine(True)
rightLane = LaneLine(False)

def processImage(img, vis=False):
    heatmap = np.zeros_like(img[:,:,0])
    for i, s in enumerate(WINDOW_SCALES):
        heatmap += findCars(img, s, Y_STARTS[i], Y_STOPS[i])
    h_smoothed = heatmapper.update(heatmap, vis)
    labels = label(h_smoothed)
    i_bbox, bboxes = determineBoundingBoxes(np.copy(img), labels)
    i_everything = findLanes(i_bbox, leftLane, rightLane, cameraMtx, distCoeffs)
    heatmapper.addVehiclesHeat(bboxes)
    return i_everything
```

The pipeline in action: [Results Video](https://github.com/Merberg/CarND-Vehicle-Detection/blob/master/output_images/project_results.mp4)

![Results on project_video][video1]




---

### 4. Discussion

I wanted to learn and experiment with Support Vector Machines and traditional machine learning, thus the use of a linear SVM, however I think deep learning would be a better application.  If the target hardware includes a GPU, this helps solidfy this option.  The classification layer could be trained to identify more types of vehicles, even cyclists that behave differently than cars on roadways.
This pipeline is by no means realtime, a huge drawback.  Numerous improvements could be made to help with timing.  Depending on the application of the tracking algorithm and deadlines, skipping some frames could be an option.  Combining logic from lane detection and road placement could be used to narrow the region of interest to look for vehicles.

