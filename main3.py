import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanFilter:
    def __init__(self, x, y, w, h):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)

        self.predicted = (x, y)
        self.w = w
        self.h = h
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
                        self.trackers[self.id_count] = KalmanFilter(detection[0], detection[1], detection[2], detection[3])
                        self.id_count += 1
            else:
                # If no objects were tracked, create new trackers for all detections
                for detection in detections:
                    self.trackers[self.id_count] = KalmanFilter(detection[0], detection[1], detection[2], detection[3])
                    self.id_count += 1

        # Collect tracking results
        results = []
        for obj_id, tracker in self.trackers.items():
            x, y = tracker.predicted
            w = tracker.w
            h = tracker.h
            results.append([x - w // 2, y - h // 2, w, h, obj_id])

        return results


def detect_squares(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Filter out small areas, adjust as needed
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])
    return detections

def detect_circles(frame):
    detections = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            detections.append([x-r, y-r, 2*r, 2*r])
    return detections, gray


# Initialize the tracker
tracker = ObjectTracker()

# Video capture and processing
cap = cv2.VideoCapture("luxonis_task_video.mp4")
kernel = np.ones((5, 5), np.uint8)
object_detector = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=100)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mask = object_detector.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    detections_squares = detect_squares(mask)
    detections_circles, gray = detect_circles(frame)
    detections = detections_squares + detections_circles

    # Update tracker with detections
    boxes_ids = tracker.update(detections)

    # Draw results
    for box_id in boxes_ids:
        x, y, w, h, obj_id = box_id
        cv2.putText(frame, str(obj_id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Gray", gray)
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(10)
    if key == 27:  # ESC to break
        break

cap.release()
cv2.destroyAllWindows()