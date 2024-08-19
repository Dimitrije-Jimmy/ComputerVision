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
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)

        self.predicted = (x, y)
        self.w = w
        self.h = h
        self.des = des
        self.missed_frames = 0

    def predict(self):
        pred = self.kalman.predict()
        self.predicted = (int(pred[0].item()), int(pred[1].item()))
        return self.predicted

    def correct(self, x, y):
        self.kalman.correct(np.array([[x], [y]], np.float32))


class ObjectTracker:
    def __init__(self):
        self.trackers = {}
        self.id_count = 0

    def update(self, detections):
        # Predict the next position for each tracker
        for obj_id, tracker in list(self.trackers.items()):
            tracker.predict()
            tracker.missed_frames += 1

            # Remove trackers that have missed for too many frames
            if tracker.missed_frames > 10:  # Allow some frames for occlusion
                del self.trackers[obj_id]

        if len(detections) > 0:
            object_ids = list(self.trackers.keys())
            predicted_positions = np.array([self.trackers[obj_id].predicted for obj_id in object_ids])

            detected_positions = np.array([(d[0] + d[2] // 2, d[1] + d[3] // 2) for d in detections])
            if len(predicted_positions) > 0:
                distance_matrix = np.linalg.norm(predicted_positions[:, None] - detected_positions[None, :], axis=2)

                row_ind, col_ind = linear_sum_assignment(distance_matrix)

                assigned_ids = set()

                for r, c in zip(row_ind, col_ind):
                    if distance_matrix[r, c] < 100:  # Threshold to consider a match
                        self.trackers[object_ids[r]].correct(detected_positions[c][0], detected_positions[c][1])
                        self.trackers[object_ids[r]].missed_frames = 0
                        assigned_ids.add(c)

                # Add new trackers for unassigned detections
                for i, detection in enumerate(detections):
                    if i not in assigned_ids:
                        self.trackers[self.id_count] = KalmanFilter(detection[0], detection[1], detection[2], detection[3], detection[4])
                        self.id_count += 1
            else:
                # If no objects were tracked, create new trackers for all detections
                for detection in detections:
                    self.trackers[self.id_count] = KalmanFilter(detection[0], detection[1], detection[2], detection[3], detection[4])
                    self.id_count += 1

        # Collect tracking results
        results = []
        coords = {}
        for obj_id, tracker in self.trackers.items():
            x, y = tracker.predicted
            w = tracker.w
            h = tracker.h
            des = tracker.des
            coords[obj_id] = [x, y]
            results.append([obj_id, x - w // 2, y - h // 2, w, h, des])

        print(coords)
        return results, coords


def detect_squares(gray):
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
    detections = []

    # Apply Hough Circles detection
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            detections.append([x-r, y-r, 2*r, 2*r, r])

    return detections, gray


# Initialize the coordinates file
filename = 'coords.txt'
initialize_file(filename)

# Initialize the tracker
tracker = ObjectTracker()

# Video capture and processing
cap = cv2.VideoCapture("luxonis_task_video.mp4")

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

    append_file(filename, coords)

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