# SOTA Vision Foundation Models Leaderboard (08/Sep/2024)


**Table of Contents**
- [**Overview**](#overview)
- [**Image Classification Leaderboard**](#image-classification-leaderboard)
- [**Object Detection Leaderboard**](#object-detection-leaderboard)
- [**Instance Segmentation Leaderboard**](#instance-segmentation-leaderboard)


### Overview

Ranking of Vision Foundation Models by Task. 

- Last update: (26/Sep/2024)

### **Image Classification Leaderboard**

| Rank | Model Name | Architecture | Pretraining Data | Accuracy | Parameters | Datasets | Release Date |
|--|--|--|--|--|--|--|--|
| 1 | ViT-G/14 | Vision Transformer (ViT) | JFT-3B | 90.3% | 6.5B | ImageNet-1k, ImageNet-21k | 2022 |
| 2 | Swin Transformer | Hierarchical Transformer | ImageNet-22k | 88.4% | 197M | ImageNet-1k, COCO | 2021 |
| 3 | ConvNeXt | Convolution-based Model | ImageNet-22k | 87.8% | 350M | ImageNet-1k | 2022 |
| 4 | DeiT-III | Data-efficient ViT | ImageNet-1k | 87.1% | 87M | ImageNet-1k | 2021 |
| 5 | EfficientNetV2 | EfficientNet Architecture | ImageNet-21k | 85.7% | 480M | ImageNet-1k | 2021 |
| 6 | ResNet-RS | Residual Networks | ImageNet-1k | 85.2% | 100M | ImageNet-1k | 2021 |
| 7 | CoAtNet | Convolution-Attention Hybrid | ImageNet-22k | 84.9% | 366M | ImageNet-1k | 2021 |
| 8 | NFNet-F6 | Normalizer-Free Network | ImageNet-1k | 86.5% | 438M | ImageNet-1k | 2021 |
| 9 | MLP-Mixer | MLP-based Vision Model | ImageNet-1k | 85.3% | 59M | ImageNet-1k | 2021 |
| 10 | RegNetY | Regularization Networks | ImageNet-1k | 85.4% | 145M | ImageNet-1k | 2021 |

---

### **Object Detection Leaderboard**

| Rank | Model Name | Architecture | Backbone | mAP (COCO) | Params | GFLOPs | Datasets | Release Date |
|--|--|--|--|--|--|--|--|--|
| 1 | Swin-L | Swin Transformer | Swin-L | 58.7 | 284M | 1382 | COCO | 2021 |
| 2 | DETR | End-to-End Detection Transformer | ResNet-50 | 48.1 | 41M | 86 | COCO | 2020 |
| 3 | YOLOv7 | You Only Look Once | CSP-Darknet53 | 56.8 | 67M | 18 | COCO | 2022 |
| 4 | EfficientDet-D7 | EfficientNet Backbone | EfficientNet | 52.2 | 51M | 325 | COCO | 2020 |
| 5 | Cascade Mask R-CNN | CNN + Cascade | ResNet-101 | 53.3 | 101M | 390 | COCO | 2019 |
| 6 | Faster R-CNN | CNN + Region Proposal | ResNet-50 | 42.1 | 41M | 184 | COCO | 2017 |
| 7 | CenterNet | Center-based Detection | Hourglass-104 | 47.0 | 202M | 630 | COCO | 2020 |
| 8 | FCOS | Fully Convolutional Detection | ResNeXt-101 | 45.2 | 64M | 128 | COCO | 2019 |
| 9 | YOLOv4 | You Only Look Once | CSP-Darknet53 | 51.5 | 65M | 117 | COCO | 2020 |
| 10 | RetinaNet | Focal Loss-based Detection | ResNet-101 | 40.1 | 56M | 96 | COCO | 2018 |

---

### **Instance Segmentation Leaderboard**

| Rank | Model Name | Architecture | Backbone | AP (COCO) | Params | GFLOPs | Datasets | Release Date |
|--|--|--|--|--|--|--|--|--|
| 1 | Swin-L Mask R-CNN | Swin Transformer | Swin-L | 53.8 | 284M | 1382 | COCO | 2021 |
| 2 | Cascade Mask R-CNN | Cascade CNN | ResNet-101 | 50.2 | 101M | 390 | COCO | 2019 |
| 3 | SOLOv2 | Segment Objects by Locations | ResNet-50 | 42.1 | 34M | 113 | COCO | 2020 |
| 4 | Mask R-CNN | Region Proposal + CNN | ResNet-101 | 39.8 | 101M | 296 | COCO | 2017 |
| 5 | PointRend | Point-based Segmentation | ResNet-50 | 41.2 | 46M | 96 | COCO | 2020 |
| 6 | HTC | Hybrid Task Cascade | ResNet-101 | 43.6 | 128M | 400 | COCO | 2019 |
| 7 | YOLACT++ | Real-time Instance Segmentation | ResNet-101 | 31.2 | 74M | 100 | COCO | 2019 |
| 8 | CondInst | Conditional Instance Segmentation | ResNet-50 | 39.1 | 31M | 73 | COCO | 2020 |
| 9 | TensorMask | Mask Head CNN | ResNet-101 | 37.1 | 130M | 430 | COCO | 2019 |
| 10 | DeepLabV3+ | Atrous CNN for Segmentation | Xception-71 | 35.4 | 43M | 100 | COCO | 2018 |


