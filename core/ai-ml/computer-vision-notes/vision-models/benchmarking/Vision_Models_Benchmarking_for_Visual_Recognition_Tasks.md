# Vision Models Benchmarking for Visual Recognition Tasks

## Table of Contents
  - [Overview](#overview)
  - [Image Classification](#image-classification)
  - [Object Detection](#object-detection)
  - [Semantic Segmentation](#semantic-segmentation)
  - [Instance Segmentation](#instance-segmentation)
  - [Image Captioning](#image-captioning)

## Overview

This is a compact template to benckmarking Vision Models based on visual recognition tasks capability and efficiency.

- See the upcoming results on the [leaderboard](./Vision-Foundation-Models-Leaderboard.md) (live soon on [HF space)](https://huggingface.co/spaces)).

## Image Classification
| SOTA Vision Models              | Description                                                    | Capability                     | Performance | Metrics           | Datasets                  |
|----------------------------------|----------------------------------------------------------------|---------------------------------|-------------|-------------------|---------------------------|
| Vision Transformer (ViT)         | Transformer-based model for image classification               | High performance on image classification tasks | Varies by model   | Accuracy, Precision, Recall | ImageNet, CIFAR-10, CIFAR-100 |
| ResNet                           | Convolutional neural network for image recognition              | Good for feature extraction    | High         | Accuracy          | ImageNet, MS COCO          |
| EfficientNet                     | Scalable neural network for image classification and efficiency | Optimized for speed and accuracy | Very High    | Top-1 Accuracy    | ImageNet, OpenImages       |
| ConvNeXt                         | Modernized architecture with convolutions                      | High accuracy and speed        | Very High    | Accuracy, F1 Score | ImageNet, COCO             |
| DenseNet                         | CNN with dense connections to enhance feature propagation       | High feature reuse             | High         | Accuracy, Precision | ImageNet, CIFAR-10         |
| Swin Transformer                 | Transformer-based hierarchical vision transformer               | Strong performance on various vision tasks | Very High    | Top-1 Accuracy    | ImageNet, ADE20K           |
| RegNet                           | Regularized network designed for efficiency                     | Efficient and scalable         | High         | Accuracy          | ImageNet                   |
| InceptionV3                      | Deep CNN with efficient architecture                            | High classification performance | High         | Accuracy, F1 Score | ImageNet                   |
| MnasNet                          | Mobile neural architecture search network for classification    | Designed for mobile performance | Medium       | Top-1 Accuracy    | ImageNet                   |
| MobileNetV3                      | Lightweight CNN for mobile and edge devices                     | Optimized for mobile efficiency | High         | Accuracy          | ImageNet                   |

## Object Detection
| SOTA Vision Models              | Description                                                     | Capability                     | Performance | Metrics           | Datasets                  |
|----------------------------------|-----------------------------------------------------------------|---------------------------------|-------------|-------------------|---------------------------|
| DETR                             | End-to-end object detection model                               | Performs object detection directly | High         | mAP (mean Average Precision) | COCO, Pascal VOC          |
| YOLOv5                           | Real-time object detection                                      | Extremely fast and accurate detection | Very High    | Precision, Recall  | COCO, VOC                 |
| Faster R-CNN                     | Region-based convolutional neural network for object detection  | Highly accurate but slower      | High         | mAP               | COCO, Pascal VOC           |
| RetinaNet                        | Focal loss-based detector to handle class imbalance             | Accurate object detection       | High         | mAP               | COCO, OpenImages           |
| CenterNet                        | Object detection by predicting object center points             | Efficient and fast              | High         | mAP               | COCO, Pascal VOC           |
| YOLOv7                           | Fast object detection with improved performance                 | Real-time detection             | Very High    | mAP, Precision     | COCO, OpenImages           |
| SSD                              | Single shot multibox detector for real-time detection           | Very fast object detection      | Medium       | mAP               | COCO, Pascal VOC           |
| Cascade R-CNN                    | Cascade-based object detection model                           | High accuracy with refinement   | High         | mAP               | COCO, VOC                  |
| R-FCN                            | Region-based fully convolutional networks for object detection  | Fast and accurate               | Medium       | mAP               | COCO, VOC                  |
| EfficientDet                     | Scalable and efficient object detector                         | Balanced between speed and accuracy | High         | mAP               | COCO, OpenImages           |

## Semantic Segmentation
| SOTA Vision Models              | Description                                                    | Capability                     | Performance | Metrics           | Datasets                  |
|----------------------------------|----------------------------------------------------------------|---------------------------------|-------------|-------------------|---------------------------|
| DeepLabV3                        | Semantic segmentation with deep learning                       | High segmentation accuracy      | High         | IoU (Intersection over Union) | COCO, Cityscapes          |
| U-Net                            | Convolutional network for biomedical image segmentation         | Best suited for medical imaging | High         | Dice Coefficient  | Medical imaging datasets   |
| SegFormer                        | Transformer-based model for segmentation                       | Efficient and fast segmentation | Very High    | IoU               | ADE20K, Cityscapes         |
| PSPNet                           | Pyramid scene parsing network for pixel-level segmentation      | High performance on large scenes | High         | mIoU              | Cityscapes, ADE20K         |
| FCN                              | Fully convolutional network for dense prediction                | Basic but effective for segmentation | Medium       | mIoU              | PASCAL VOC, COCO           |
| HRNet                            | High-resolution network for segmentation                       | High accuracy and resolution    | Very High    | IoU               | Cityscapes, ADE20K         |
| OCRNet                           | Object context reasoning network for segmentation              | Accurate and context-aware      | High         | IoU, mIoU         | COCO, ADE20K               |
| BEiT                             | Transformer model for semantic segmentation                    | Strong performance on diverse segmentation tasks | Very High    | IoU, mIoU         | COCO, ADE20K               |
| DANet                            | Dual attention network for semantic segmentation               | Context-aware with dual attention | High         | IoU               | COCO, Cityscapes           |
| FPN                              | Feature pyramid network for object detection and segmentation  | Accurate multi-scale segmentation | Medium       | IoU, mIoU         | COCO, VOC                  |

## Instance Segmentation
| SOTA Vision Models              | Description                                                    | Capability                     | Performance | Metrics           | Datasets                  |
|----------------------------------|----------------------------------------------------------------|---------------------------------|-------------|-------------------|---------------------------|
| Mask R-CNN                       | Combines object detection and instance segmentation            | Highly accurate                 | High         | AP (Average Precision) | COCO, LVIS               |
| SOLOv2                           | Segmenting objects with instance-level masks                   | Fast and accurate               | High         | AP, IoU            | COCO, Cityscapes          |
| YOLACT                           | Real-time instance segmentation                                | Extremely fast                  | Medium       | AP                | COCO, VOC                 |
| CondInst                         | Conditionally instantiated instance segmentation               | Efficient and high performance  | High         | AP, mAP            | COCO, Cityscapes          |
| PointRend                        | Rendering point-based instance segmentation                    | High-resolution segmentation    | High         | mAP               | COCO, LVIS               |
| TensorMask                       | Dense object instance segmentation using tensors               | High accuracy and efficiency    | High         | mAP               | COCO, OpenImages          |
| BlendMask                        | Blending masks for efficient instance segmentation             | Very efficient and accurate     | High         | mAP               | COCO, VOC                 |
| PolarMask                        | Instance segmentation using polar representation               | Fast and accurate               | Medium       | AP, IoU            | COCO, VOC                 |
| CenterMask                       | Combining instance segmentation with keypoint estimation       | Balanced speed and accuracy     | Medium       | mAP               | COCO, Cityscapes          |
| SCNet                            | Soft constraint instance segmentation model                    | High accuracy for edge cases    | Medium       | mAP, IoU          | COCO, Cityscapes          |

## Image Captioning
| SOTA Vision Models              | Description                                                    | Capability                     | Performance | Metrics           | Datasets                  |
|----------------------------------|----------------------------------------------------------------|---------------------------------|-------------|-------------------|---------------------------|
| BLIP                             | Vision-language pretraining for image-to-text generation        | Very accurate caption generation | High         | BLEU, CIDEr        | MS COCO, Flicker8k         |
| OFA                              | Unified model for visual and text tasks                        | Caption generation and reasoning | Very High    | BLEU, METEOR       | Visual Genome, MS COCO     |
| ViLBERT                          | Multi-modal pretraining for vision and language tasks          | Accurate and contextual captioning | High         | CIDEr, SPICE       | COCO, Flickr30k            |
| ClipCap                          | Clip-based image captioning model                              | Efficient and fast captioning   | High         | ROUGE, SPICE       | MS COCO, LSMDC             |
| XGPT                             | Cross-modal generative pretraining for text and images         | Cross-lingual and cross-modal captioning | Very High    | CIDEr, SPICE       | COCO, Multi30k             |
| VLP                              | Vision-language pretraining for contextual captioning          | Strong captioning capabilities  | High         | BLEU, ROUGE        | COCO, Visual Genome        |
| LXMERT                           | Learning cross-modality encoder representations from transformers | Context-aware caption generation | High         | CIDEr, BLEU        | COCO, VG                  |
| Unified VLP                      | Unified pretraining for vision and language tasks              | Multi-task vision-to-text generation | Very High    | CIDEr, METEOR      | COCO, VG                  |
| M4C                              | Multi-modal pretraining for captioning and visual question answering | Efficient and accurate captions | High         | SPICE, CIDEr       | COCO, Flickr30k            |
| Oscar                            | Object-Semantics Aligned Pretraining for caption generation    | Strong performance on image captioning | High         | CIDEr, SPICE       | COCO, Flickr30k
