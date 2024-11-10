# OpenCV - Notes

## Table of Contents (ToC)

  - [Introduction](#introduction)
    - [What's OpenCV?](#whats-opencv)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [OpenCV Architecture Pipeline](#opencv-architecture-pipeline)
    - [How OpenCV Works?](#how-opencv-works)
    - [OpenCV Techniques](#opencv-techniques)
    - [Some Hands-on Examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)

## Introduction
OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library.

### What's OpenCV?
- A library of programming functions for real-time computer vision.
- Supports a wide range of applications in image and video processing.
- Provides tools for both academic and commercial use in computer vision.

### Key Concepts and Terminology
- **Image Processing**: Techniques to enhance or extract information from images.
- **Computer Vision**: Field of study focused on enabling machines to interpret and understand visual information.
- **Contours**: Curves joining all continuous points along a boundary with the same color or intensity.
- **Feature Detection**: Identifying important visual elements within an image.

### Applications
- Object and face detection.
- Image segmentation and classification.
- Video analysis, including motion tracking and object recognition.
- Augmented reality and computer graphics.

## Fundamentals

### OpenCV Architecture Pipeline
- Loading and reading image or video data.
- Preprocessing with techniques like resizing, filtering, and color conversion.
- Applying algorithms for detection, recognition, and analysis.
- Displaying results and performing further actions based on analysis.

### How OpenCV Works?
- Utilizing functions from the `cv2` module to handle image and video data.
- Applying various image processing techniques such as blurring, thresholding, and edge detection.
- Using feature detection methods like SIFT, SURF, and ORB.
- Implementing machine learning models for tasks such as face and object detection.

### OpenCV Techniques
- **Image Filtering**: Smoothing, sharpening, and edge detection using filters.
- **Object Detection**: Techniques like Haar cascades and deep learning-based methods.
- **Feature Matching**: Matching keypoints between images using descriptors.
- **Geometric Transformations**: Operations like rotation, translation, and scaling.

### Some Hands-on Examples
- **Edge Detection**: Using Canny edge detector.
- **Face Detection**: Implementing Haar cascades for face recognition.
- **Object Tracking**: Using meanshift or camshift algorithms.
- **Image Transformation**: Applying perspective transformation on images.

## Tools & Frameworks
- OpenCV
- NumPy
- OpenCV's Contrib modules for extra functionality
- Integration with other libraries like TensorFlow and PyTorch

## Hello World!
```python
import cv2

# Load an image from file
image = cv2.imread('path_to_image.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray_image, 100, 200)

# Display the result
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Lab: Zero to Hero Projects
- **Project 1**: Building a real-time face detection application.
- **Project 2**: Creating a motion detection and tracking system.
- **Project 3**: Developing an image stitching tool for panorama creation.
- **Project 4**: Implementing augmented reality with marker detection.

## References
- OpenCV documentation: https://docs.opencv.org/
- OpenCV Python tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- OpenCV GitHub repository: https://github.com/opencv/opencv
- Additional OpenCV resources: https://opencv.org/extras/

- FreeCodeCamp : OpenCV Course - Full Tutorial with Python
https://www.youtube.com/watch?v=oXlwWbU8l2o

- LEARN OPENCV C++ in 4 HOURS | Including 3x Projects | Computer Vision
https://www.youtube.com/watch?v=2FYm3GOonhk

- FreeCodeCamp : OpenCV Python Course - Learn Computer Vision and AI
https://www.youtube.com/watch?v=P4Z8_qe2Cu0

- Google : How Computer Vision Works
https://www.youtube.com/watch?v=OcycT1Jwsns

- Crash Course - Computer Vision: Crash Course Computer Science #35
https://www.youtube.com/watch?v=-4E2-0sxVUM



