# Import required libraries
from ultralytics import YOLO  # YOLO object detection model
import cv2  # OpenCV for camera and image processing
import math  # Math functions for calculations

# Start webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Load pre-trained YOLO model (downloads automatically first time)
model = YOLO("yolo11n.pt")  # "n" = nano (fastest, smallest model)

# Object class names that YOLO can detect (80 objects from COCO dataset)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Main detection loop
while True:
    # Read frame from webcam
    success, img = cap.read()
    
    # Run YOLO detection on the frame
    results = model(img, stream=True)
    
    # Process each detection result
    for r in results:
        boxes = r.boxes  # Get all detected bounding boxes
        
        # Loop through each detected object
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]  # Top-left and bottom-right corners
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
            
            # Draw rectangle around detected object
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # Get confidence score (how sure the model is)
            confidence = math.ceil((box.conf[0]*100))/100  # Round to 2 decimals
            
            # Get class name of detected object
            cls = int(box.cls[0])
            object_name = classNames[cls]
            
            # Create label text with object name and confidence
            label = f'{object_name} {confidence}'
            
            # Calculate text size for background
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            
            # Calculate position for text background
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            
            # Draw filled rectangle as text background
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            
            # Put text label on the image
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
    # Display the frame with detections
    cv2.imshow('Object Detection', img)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

