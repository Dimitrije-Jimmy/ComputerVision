# Object Detection and Tracking with Kalman Filter

This Python script performs object detection and tracking using OpenCV. It detects squares and circles in video frames and tracks them using a Kalman Filter. The tracking results are saved to a file for further analysis or evaluation.

## Features

- **Object Detection**: The script can detect squares and circles in video frames using contour detection and the Hough Circle Transform, respectively.
- **Object Tracking**: Detected objects are tracked across frames using a Kalman Filter, ensuring smooth tracking even when objects temporarily disappear or overlap.
- **File Logging**: All object coordinates and tracking information are logged to a text file, making it easier to review or analyze the tracking performance.
- **Real-Time Visualization**: The script displays the detection and tracking results in real-time using OpenCV, showing bounding boxes and object IDs on the video.

## Project Context
This project was developed as part of a job interview task, where I had to demonstrate my ability to apply computer vision techniques to real-world scenarios. The entire project, including learning the necessary concepts, libraries and implementation, was completed within a 5-hour timeframe. Despite being new to some of the tools and techniques used, I successfully built a functional object detection and tracking system within the given time limit.

## Prerequisites

Make sure you have the following Python packages installed:
- `opencv-python`
- `numpy`
- `scipy`

You can install these packages using pip:

```bash
pip install opencv-python numpy scipy
```

## Usage

### **1. Prepare Your Video File**
Ensure that the video file you want to process is named `luxonis_task_video.mp4` and is located in the same directory as the script.

### **2. Run the Script**
```bash
python main6.py
```

### **3. Script Overview**

#### **`KalmanFilter Class`**

**Purpose**: Implements a Kalman Filter for tracking objects.
**Parameters**:
- **`__init__(x, y, w, h, des)`**: Initializes the Kalman Filter with the object's position, size, and descriptor.
- **`predict()`**: Predicts the next position of the object.
- **`correct(x, y)`**: Corrects the filter based on the observed position.
**Usage**:
```python
kf = KalmanFilter(x=100, y=200, w=50, h=50, des=0)
predicted = kf.predict()
kf.correct(x=105, y=205)
```

#### **`ObjectTracker Class`**

**Purpose**: Manages multiple Kalman Filters and tracks objects across frames.
**Parameters**:
- **`__init__()`**: Initializes the tracker.
- **`update(detections)`**: Updates the tracker with new detections and returns the tracking results and coordinates.
**Usage**:
```python
tracker = ObjectTracker()
results, coords = tracker.update(detections)
```
This class manages multiple object trackers, assigns new detections to existing trackers or creates new ones, and returns the tracking results and coordinates.

#### **`main(input_file, output_file)`**

**Purpose**: The main function for processing video files, detecting objects, tracking them, and saving coordinates to a file.
**Parameters**:
- **`input_file`** (str): Path to the input video file.
- **`output_file`** (str): Path to the output file where coordinates will be saved.
**Usage**:
```python
main('luxonis_task_video.mp4', 'coords.txt')
```
This function initializes the coordinates file and the object tracker. It then reads frames from the video file, detects squares and circles, updates the tracker, and appends the coordinates to the output file. It also visualizes the tracked objects in the video frames.


## Notes

- The Kalman Filter is used to predict and correct the positions of tracked objects based on their previous states.
- The threshold values used for detection and tracking can be adjusted based on the specific requirements of your application.

## Future Updates

There are several potential improvements and extensions for this project:
- **Improved Occlusion Handling**: Enhance the Kalman Filter or integrate additional algorithms to better manage cases where objects overlap or are partially hidden.
- **Multi-Class Object Tracking**: Extend the tracking system to handle multiple object classes beyond squares and circles, with more sophisticated detection models.
- **Performance Optimization**: Optimize the code for real-time performance, particularly when dealing with high-resolution video or multiple objects.
- **Model Integration**: Integrate machine learning models for more robust object detection, which could improve accuracy in diverse and challenging environments.

This project served as a valuable learning experience and a solid foundation for more advanced computer vision work in the future.

## License
This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/license/mit) file for details.

## Acknowledgements
* Luxonis company for providing the opportunity and task
