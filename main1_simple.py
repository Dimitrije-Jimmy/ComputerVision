import cv2      # OpenCV for detection of objects

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                #dist = math.hypot(cx - pt[0], cy - pt[1])
                distsq = (cx - pt[0])**2 + (cy - pt[1])**2

                #if dist < 25:
                if distsq < 625:
                #if distsq < 10300:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
    
# Create tracker object
tracker = EuclideanDistTracker()

#importing the video, has to be in same cell
cap = cv2.VideoCapture("luxonis_task_video.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=100)

# going through the frames and extracting it
while True:
    ret, frame = cap.read()

    mask = object_detector.apply(frame)

    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    
    for cnt in contours:
        #cv2.drawContours(frame, [cnt], -1, (255, 0, 255), 2)

        x, y, w, h = cv2.boundingRect(cnt)
        detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)

    # we extract each object in this frame from the boxes_ids
    for box_id in boxes_ids:
        x, y, w, h, id = box_id

        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        # Plots the rectangle of each bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    

    # This shows each frame
    #cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # key 27 is ESC, this will break the loop
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()