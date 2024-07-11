from ultralytics import YOLO
import cv2
import math
import cvzone
from SORT import *

cap = cv2.VideoCapture("cars.mp4")

model = YOLO('yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("car_image_new.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]

totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    
    detections = np.empty((0,5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass in ["car", "truck", "motorbike"] and conf > 0.3:
                # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1), max(35,y1)), scale=0.8, thickness=1)
                # cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 3)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack([detections, currentArray])
          
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cvzone.putTextRect(img, f'{int(Id)}', (max(0,x1), max(35,y1)), scale=2, thickness=3, offset=10)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)
        width, height = x2-x1, y2-y1
        cx, cy = x1+width//2, y1+height//2
        
        if limits[0]<cx<limits[2] and limits[1]-15<cy<limits[1]+15:
            if totalCount.count(Id) == 0:
                totalCount .append(Id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
                
    cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50))
                      
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)

