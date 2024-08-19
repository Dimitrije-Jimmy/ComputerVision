import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def initialize_file(filename):
    with open(filename, 'w') as file:
        file.write('')

def append_file(filename, coords):
    with open(filename, 'a') as file:
        coords_str = "{"
        coords_str += ", ".join(f"{key}: {value}" for key, value in coords.items())
        coords_str += "}"
        file.write(f"{coords_str}\n")


class KalmanFilter:
    def __init__(self, x, y, w, h, des):
        """
        Initializes the Kalman Filter with the object's position, size, and descriptor.

        Parameters:
        - x (int): The x-coordinate of the object's position.
        - y (int): The y-coordinate of the object's position.
        - w (int): The width of the object.
        - h (int): The height of the object.
        - des (int): Descriptor value (e.g., radius for circles, 0 for squares).
        """
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        self.kalman.statePre = np.array([x, y, 0, 0], np.float32)
        self.kalman.statePost = np.array([x, y, 0, 0], np.float32)

        self.predicted = (x, y)
        self.w = w
        self.h = h
        self.des = des
        self.missed_frames = 0

    def predict(self):
        """
        Predicts the next position of the object.

        Returns:
        - (tuple): The predicted (x, y) coordinates of the object.
        """
        pred = self.kalman.predict()
        self.predicted = (int(pred[0]), int(pred[1]))
        return self.predicted

    def correct(self, x, y):
        """
        Corrects the filter based on the observed position.

        Parameters:
        - x (int): The x-coordinate of the observed position.
        - y (int): The y-coordinate of the observed position.
        """
        self.kalman.correct(np.array([x, y], np.float32))


class ObjectTracker:
    def __init__(self):
        """
        Initializes the tracker with an empty dictionary of trackers and an ID counter.
        """
        self.trackers = {}
        self.id_count = 0

    def update(self, detections):
        """
        Updates the tracker with new detections.

        Parameters:
        - detections (list): A list of detections where each detection is represented as [x, y, w, h, des].

        Returns:
        - results (list): A list of tracked objects with their ID and coordinates.
        - coords (dict): A dictionary with object IDs as keys and their coordinates as values.
        """
        # Predict the next position for each tracker
        for obj_id, tracker in list(self.trackers.items()):
            tracker.predict()
            tracker.missed_frames += 1

            if tracker.missed_frames > 10:  # Remove trackers with too many missed frames
                del self.trackers[obj_id]

        if detections:
            object_ids = list(self.trackers.keys())
            predicted_positions = np.array([tracker.predicted for tracker in self.trackers.values()])
            detected_positions = np.array([(d[0] + d[2] // 2, d[1] + d[3] // 2) for d in detections])

            if predicted_positions.size:
                distance_matrix = np.linalg.norm(predicted_positions[:, None] - detected_positions[None, :], axis=2)
                row_ind, col_ind = linear_sum_assignment(distance_matrix)

                assigned_ids = set()
                for r, c in zip(row_ind, col_ind):
                    if distance_matrix[r, c] < 100:  # Threshold to consider a match
                        self.trackers[object_ids[r]].correct(*detected_positions[c])
                        self.trackers[object_ids[r]].missed_frames = 0
                        assigned_ids.add(c)

                # Add new trackers for unassigned detections
                for i, detection in enumerate(detections):
                    if i not in assigned_ids:
                        self.trackers[self.id_count] = KalmanFilter(*detection)
                        self.id_count += 1
            else:
                # If no objects were tracked, create new trackers for all detections
                for detection in detections:
                    self.trackers[self.id_count] = KalmanFilter(*detection)
                    self.id_count += 1

        # Collect tracking results
        results = []
        coords = {}
        for obj_id, tracker in self.trackers.items():
            x, y = tracker.predicted
            coords[obj_id] = [x, y]
            results.append([obj_id, x - tracker.w // 2, y - tracker.h // 2, tracker.w, tracker.h, tracker.des])

        print(coords)
        return results, coords


def detect_squares(gray):
    """
    Detects squares in a grayscale image.

    Parameters:
    - gray (numpy.ndarray): A grayscale image.

    Returns:
    - detections (list): A list of detections where each detection is represented as [x, y, w, h, 0].
    - gray (numpy.ndarray): The original grayscale image.
    """
    detections = []

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the approximated polygon has 4 sides
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)

            # Check if the aspect ratio is close to 1 (for squares)
            aspect_ratio = float(w) / h
            if 0.9 < aspect_ratio < 1.1:
                detections.append([x, y, w, h, 0])

    return detections, gray


def detect_circles(gray):
    """
    Detects circles in a grayscale image using the Hough Circle Transform.

    Parameters:
    - gray (numpy.ndarray): A grayscale image.

    Returns:
    - detections (list): A list of detections where each detection is represented as [x-r, y-r, 2*r, 2*r, r].
    - gray (numpy.ndarray): The original grayscale image.
    """
    detections = []

    # Apply Hough Circles detection
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            detections.append([x-r, y-r, 2*r, 2*r, r])

    return detections, gray


def main(input_file, output_file):
    """
    Processes video files, detects objects, tracks them, and saves coordinates to a file.

    Parameters:
    - input_file (str): Path to the input video file.
    - output_file (str): Path to the output file where coordinates will be saved.
    """

    initialize_file(output_file)
    tracker = ObjectTracker()
    cap = cv2.VideoCapture(input_file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections_squares, gray = detect_squares(gray)
        detections_circles, gray = detect_circles(gray)
        detections = detections_squares + detections_circles

        # Update tracker with detections
        boxes_ids, coords = tracker.update(detections)

        append_file(output_file, coords)

        # Draw results
        for box_id in boxes_ids:
            obj_id, x, y, w, h, r = box_id
            cv2.putText(frame, str(obj_id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            if r == 0:  # It's a square
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            elif r != 0:  # It's a circle
                cv2.circle(frame, (x + r, y + r), r, (0, 255, 0), 3)

            posx, posy = coords[obj_id]
            cv2.rectangle(frame, (posx, posy), (posx + 5, posy + 5), (0, 0, 255), 3)

        cv2.imshow("Gray", gray)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(10)
        if key == 27:  # ESC to break
            break

    cap.release()
    cv2.destroyAllWindows()


main("luxonis_task_video.mp4", 'coords.txt')