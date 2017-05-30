# Project: Search and Sample Return


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # "Image References"
[image1a]: ./output/grid.png
[image1b]: ./output/warped_grid.png
[image2a]: ./output/ground_extract.png
[image2b]: ./output/obstacle_extract.png
[image3]: ./output/result.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Training/Calibration (Notebook Analysis)
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

##### Perspective Transform

This part is left unchanged.

The transform is done using the following code:

```python
dst_size = 5 
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
```

Below is an example of a camera image with grid added (to the left) and this same image with perspective transform applied to it (right).

![alt text][image1a]  ![alt text][image1b]

##### Color Thresholding

A new function is created called `color_threshold()`.  It takes lower and upper thresholds. The function requires that each pixel in the input image be above all three lower threshold values in RGB and below the three  upper threshold values in RGB.

Here is the code implementing the `color_threshold()` function:

```python
def color_threshold(img, lower_thresh, upper_thresh):
    color_select = np.zeros_like(img[:,:,0])
   
    r_channel = img[:,:,0]
    g_channel = img[:,:,1]
    b_channel = img[:,:,2]
    r_binary = np.zeros_like(img[:,:,0])
    g_binary = np.zeros_like(img[:,:,0])
    b_binary = np.zeros_like(img[:,:,0])
    r_binary[(r_channel >= lower_thresh[0]) & (r_channel <= upper_thresh[0])] = 1
    g_binary[(g_channel >= lower_thresh[1]) & (g_channel <= upper_thresh[1])] = 1
    b_binary[(b_channel >= lower_thresh[2]) & (b_channel <= upper_thresh[2])] = 1
    
    color_select[(r_binary == 1) & (g_binary == 1) & (b_binary == 1)] = 1
    
    return color_select
```

This function is called with different thresholds for each of the 3 different cases:
1. **Ground extraction** which is implemented in `ground_extraction()`. Lower RGB threshold (160,160,160) and upper RGB threshold (255,255,255)
2. **Obstacle extraction** which is implemented in `obstacles_extract()`. Lower RGB threshold (1,1,1) and upper RGB threshold (150,150,150)
3. **(Yellow) Rock extraction** which is implemented in `rocks_extract()`.  Lower RGB threshold (130,100,0) and upper RGB threshold (200,200,70)

An example of Ground extraction (left) and Obstacle extraction (right):

![alt text][image2a] ![alt text][image2b]

##### Coordinate Transformations

The functions `rotate_pix()` and `translate_pix()` are completed respectively as follows:

```python
# Convert yaw to radians
yaw_rad = yaw * np.pi / 180
# Apply a rotation
xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.sin(yaw_rad)
```

```python
# Apply a scaling and a translation
xpix_translated = np.int_(xpos + (xpix_rot / scale))
ypix_translated = np.int_(ypos + (ypix_rot / scale))
```

##### The Resulting Process

Below are 4 images showing the process. Upper left is a camera image. Upper right is a perspective transform. Bottom left is a combined thresholded image. Here the blue part is the navigable ground and the red part are the obstacles. Bottom right is a coordinate transform with the arrow indicating direction of travel of the robot.

![alt text][image3]


#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

##### 1. Define source and destination points for perspective transform

The same transform, as previously mentioned, is used .

##### 2. Apply perspective transform

##### 3. Apply color threshold to identify navigable terrain/obstacles/rock samples

The color thresholds are applied to find the Obstacles, (Yellow) Rocks and (Navigable) Ground. The results are coordinates corresponding to the 3 categories. The extraction functions mentioned above are used for this.

##### 4. Convert thresholded image pixel values to rover-centric coords

The coordinates found in **3.** are transformed to the rover frame of reference using `rover_coords()`.

##### 5. Convert rover-centric pixel values to world coords

The coordinates in the rover frame of reference are now converted to world frame of reference using `pix_to_world()`. This function makes use of the rotations implemented in `rotate_pix()` and translations implemented in `translate_pix()`, which have been mentioned already.

##### 6. Update worldmap (to be displayed on right side of screen)

If the robot's roll or pitch are below a certain value - meaning when the robot is on a level surface - then the obstacles, the yellow rocks and the navigable ground are added to the world map. In this case the value is set to `0.5`.

The navigable ground pixels have priority over the obstacle pixels. Hence, when certain pixels are found to be navigable (i.e. the blue channel is incremented) then their obstacle 'property' is removed (i.e. the red channel is set to `0`). 

##### 7. Make a mosaic image

Creating a composite image for the video.

##### The resulting video 

The video can be found on [YouTube](https://youtu.be/m1RWjIM2QJY). An mp4 file can also be found in the '/output' directory of this repo.

##### The Jupyter Notebook

The Jupyter notebook can be found [here](Rover_Project_Test_Notebook.html) or in the '/code' directory of this repo.


### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

The code of the function `decision_step()` in `decision.py` has been left as is, given that the robot drives around the environment in a satisfactory fashion.

The code of the function `perception_step()` in `perception.py` has been modified to do the exact same as the function `process_image()` which is described above. Certain variable names have been changed to fit this context.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

The simulator was run in autonomous mode at a frame rate between 29 and 35 fps and in a window with a resolution of 1280x768 at 'Fantastic' graphics quality. 

The robot is capable of mapping more than 40% of the terrain at more than 60% fidelity. Sometimes the fidelity starts out below 60% but then it quickly goes up above 60%.

A demo video can be found [here](https://youtu.be/JYbaht1_TJU).

In addition it happens that the robot rover gets stuck for a short while (a few seconds) when rotating away from certain obstacles. Despite this it manages to get loose and continue its exploration.

On very rare occasions the robot gets stuck in obstacles and is not able to continue its exploration path. In this case the addition of a 'reverse' driving functionality in the `decision_step()` function might be of use.

In order to increase the percentage of mapped terrain, some sort of weighting of the pixels in the world map could be used. The pixels which correspond to mapped terrain could acquire a higher weight than non-mapped ones. Hence the exploration could be guided towards the pixels on the world map with lower weight.