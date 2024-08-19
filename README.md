# Object Detection and Tracking with Kalman Filter

This Python script performs object detection and tracking using OpenCV. It detects squares and circles in video frames and tracks them using a Kalman Filter. The tracking results are saved to a file.

### Features

- **Object Detection**: Detects squares and circles in video frames.
- **Object Tracking**: Tracks detected objects using a Kalman Filter.
- **File Logging**: Saves object coordinates and tracking information to a text file.
- **Visualization**: Displays detection and tracking results in real-time using OpenCV.

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
python main.py
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

