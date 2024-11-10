# VGGNet - Very Deep Convolutional Networks for Large-Scale Image Recognition - Review

## Abstract
- The paper investigates the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting.
- The paper evaluates networks of increasing depth using an architecture with very small (3x3) convolution filters.
- The paper shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16-19 weight layers.
- The paper also shows that the representations learned by the deep networks generalize well to other datasets and tasks.

## Introduction
- The paper motivates the study of deep convolutional networks (ConvNets) for image recognition, as they have recently achieved state-of-the-art results on ImageNet and other benchmarks.
- The paper reviews the previous works on ConvNets and their design choices, such as filter sizes, pooling schemes, and activation functions.
- The paper states the main contribution as a thorough evaluation of networks of increasing depth, up to 19 weight layers, which leads to a substantial improvement over the existing models.

## Problem and Solution (Methodology)
- The paper defines the problem of image recognition as assigning a label to an image from a fixed set of categories, based on the visual content of the image.
- The paper proposes to use ConvNets as the solution, as they are composed of multiple layers of trainable filters and non-linearities, which can learn hierarchical representations of the images.
- The paper describes the methodology of training and testing the ConvNets on the ImageNet dataset, which consists of 1.2 million images for training and 50,000 images for validation, belonging to 1000 classes.
- The paper details the data augmentation, optimization, and regularization techniques used to improve the performance and reduce the overfitting of the ConvNets.

## System Architecture Pipeline
- The paper presents the architecture of the ConvNets used in the evaluation, which are based on the AlexNet model ¹, but with several modifications.
- The paper introduces two main types of ConvNets: A and B, which differ in the number and size of the convolutional layers.
- The paper also presents several variants of the ConvNets, such as C, D, and E, which have additional layers or modifications to the original architectures.
- The paper summarizes the configuration of each ConvNet in a table, showing the number of filters, filter size, stride, and padding for each layer.

## Findings
- The paper reports the results of the evaluation of the ConvNets on the ImageNet classification and localization tasks, using the top-1 and top-5 error rates as the metrics.
- The paper shows that the ConvNets with more layers outperform the shallower ones, and that the best performing model is the 19-layer ConvNet E, which achieves 7.3% top-5 error on the classification task and 25.3% top-5 error on the localization task.
- The paper also shows that the ConvNets with smaller filters perform better than the ones with larger filters, and that the use of 1x1 convolution layers helps to increase the non-linearity and reduce the number of parameters.
- The paper compares the results of the ConvNets with the previous state-of-the-art models, such as GoogLeNet ² and Clarifai ³, and shows that the ConvNets achieve competitive or superior results, especially when using a single model or a simple ensemble of two models.

## Conclusion
- The paper concludes that the depth of the ConvNets is crucial for their success in the image recognition setting, and that the use of very small convolution filters allows to build very deep networks that achieve state-of-the-art results.
- The paper also concludes that the representations learned by the ConvNets are general and transferable, as they can be used for other tasks, such as object detection, scene recognition, and fine-grained classification, with minimal or no fine-tuning.
- The paper suggests some future directions for improving the ConvNets, such as exploring different non-linearities, pooling schemes, and normalization methods, as well as applying them to other domains, such as video and speech recognition.

## Authors and organizations
- The paper is authored by Karen Simonyan and Andrew Zisserman, who are affiliated with the Visual Geometry Group at the University of Oxford.

