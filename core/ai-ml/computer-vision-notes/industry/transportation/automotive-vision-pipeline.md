# **Computer Vision Pipeline for Automotive** 

## Use case: ADAS Systems

**Problem**: Real-time lane detection for autonomous driving.


## **Pipeline Overview**

```mermaid
graph LR;
    A[Video Acquisition] --> B[Preprocessing];
    B --> C[Region of Interest];
    C --> D[Lane Detection];
    D --> E[Lane Prediction];
    E --> F[Decision Making];
```

### Description

1. **Video Acquisition**: Camera captures continuous frames from a vehicle.  
2. **Preprocessing**: Frame filtering, edge detection (Canny, Sobel).  
3. **Region of Interest (ROI)**: Extracting relevant road portions.  
4. **Lane Detection**: Using Hough Transform to detect lane lines.  
5. **Lane Prediction**: Predict the lane position for safe driving decisions.  
6. **Decision Making**: Send steering commands to adjust vehicle position.

## **Implementation (Python): Real-time Lane Detection**

This code identifies lane lines in real-time from a vehicle's camera.

```python
import cv2
import numpy as np

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width, height), (width//2, height//2)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

def detect_lanes(frame):
    edges = preprocess_frame(frame)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=150)
    return lines

def draw_lines(frame, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 10)

def lane_detection(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lines = detect_lanes(frame)
        draw_lines(frame, lines)
        cv2.imshow('Lane Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

lane_detection()
```

**Output**
- TBD


## References
- TBD


