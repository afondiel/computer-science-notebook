# 3D Computer Vision - Notes

## Table of Contents (ToC)
- [Introduction](#introduction)
  - [What's 3D Computer Vision?](#whats-3d-computer-vision)
  - [Key Concepts and Terminology](#key-concepts-and-terminology)
  - [Applications](#applications)
- [Fundamentals](#fundamentals)
  - [3D Computer Vision Architecture Pipeline](#3d-computer-vision-architecture-pipeline)
  - [How 3D Computer Vision works?](#how-3d-computer-vision-works)
  - [Types of 3D Computer Vision](#types-of-3d-computer-vision)
  - [Some hands-on examples](#some-hands-on-examples)
- [Tools & Frameworks](#tools--frameworks)
- [Hello World!](#hello-world)
- [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
- [References](#references)

## Introduction
3D computer vision involves extracting, analyzing, and understanding real-world 3D information from images, videos, or sensors to perceive the depth and structure of objects.

### What's 3D Computer Vision?
- A field focused on capturing and processing 3D geometry from 2D images and depth data.
- Enables machines to perceive spatial depth and understand object structures.
- Utilizes stereoscopic imaging, depth sensors, and computer algorithms.

### Key Concepts and Terminology
- **Depth Estimation**: Infers the distance of objects from the camera.
- **Point Cloud**: A set of data points representing the 3D shape of objects.
- **Stereo Vision**: Technique to obtain 3D depth information from two cameras.
- **SLAM (Simultaneous Localization and Mapping)**: Building a map while keeping track of the camera’s location.
- **3D Reconstruction**: Rebuilding 3D models of objects or environments from 2D images.

### Applications
- Autonomous vehicles for obstacle detection and environment mapping.
- Augmented and virtual reality to create interactive 3D experiences.
- Robotics for navigation and object manipulation in 3D space.
- Medical imaging for creating 3D models of organs and tissues.
- 3D scanning for industrial design, prototyping, and quality inspection.

## Fundamentals

### 3D Computer Vision Architecture Pipeline
- **Image Acquisition**: Collecting 2D images or depth data using sensors (e.g., stereo cameras, LIDAR).
- **Feature Extraction**: Identifying key features such as edges or textures.
- **Depth Estimation**: Calculating depth information from stereo vision or depth sensors.
- **3D Reconstruction**: Creating 3D models from extracted data.
- **Rendering and Visualization**: Displaying the 3D data or models for analysis or interaction.

### How 3D Computer Vision works?
- Uses 2D images combined with depth data to infer 3D information.
- Techniques include stereo vision, structure from motion (SfM), and depth-sensing technologies like LIDAR.
- Machine learning algorithms are used to predict depth and reconstruct 3D scenes.

### Types of 3D Computer Vision
- **Stereo Vision**: Extracts depth from two camera views (binocular disparity).
- **Depth Sensors**: LIDAR, Kinect, and time-of-flight cameras measure distances directly.
- **Multiview Stereo**: Uses multiple camera views to reconstruct dense 3D point clouds.
- **Photogrammetry**: Uses image sequences to create 3D models and maps.
- **Volumetric Reconstruction**: Reconstructs 3D volumes using techniques like voxel grids or mesh generation.

### Some hands-on examples
- Depth map generation using stereo images.
- 3D reconstruction using structure from motion (SfM) techniques.
- Creating a point cloud from LIDAR data with Python.
- Simulating 3D object detection using TensorFlow and Open3D.

## Tools & Frameworks
- **OpenCV**: Provides stereo vision and depth estimation functionalities.
- **Open3D**: For processing and visualizing 3D data, including point clouds and meshes.
- **PCL (Point Cloud Library)**: Specialized for processing 3D point cloud data.
- **Blender**: For 3D modeling, rendering, and simulation.
- **ROS**: For 3D perception in robotics, including SLAM and navigation.
- **PyTorch3D**: A library for 3D deep learning and mesh processing.

## Hello World!
```python
import open3d as o3d

# Simple point cloud visualization with Open3D
def visualize_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])

# Visualize a point cloud file
visualize_point_cloud("sample_point_cloud.ply")
```

## Lab: Zero to Hero Projects
- Build a 3D reconstruction pipeline using stereo vision with OpenCV.
- Perform 3D object detection and segmentation using Open3D and PyTorch3D.
- Implement SLAM with ROS for autonomous navigation in a 3D environment.
- Create 3D models from images using photogrammetry techniques in Python.
- Visualize large point clouds and perform mesh generation using PCL.

## References
- Szeliski, R. (2010). *Computer Vision: Algorithms and Applications*. Springer.
- Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision*. Cambridge University Press.
- Open3D Documentation: http://www.open3d.org/
- PCL Documentation: https://pointclouds.org/
- OpenCV Documentation: https://docs.opencv.org/
