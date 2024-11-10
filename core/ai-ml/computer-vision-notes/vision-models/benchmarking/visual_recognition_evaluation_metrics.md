# Visual Recognition Evaluation Metrics

## Overview 

A comprehensive overview of the various ways performance is measured in different visual recognition applications.


| **Task Type**                | **Metric**                         | **Description**                                                                                               |
|------------------------------|------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **Image Classification**     | Accuracy                           | Proportion of correctly classified instances among the total instances.                                       |
|                              | Precision                          | Proportion of true positive instances among the instances classified as positive.                             |
|                              | Recall (Sensitivity)               | Proportion of true positive instances among the actual positive instances.                                    |
|                              | F1 Score                           | Harmonic mean of precision and recall.                                                                        |
|                              | Top-k Accuracy                     | Proportion of instances where the true label is within the top k predicted probabilities.                     |
| **Object Detection**         | Mean Average Precision (mAP)       | Average precision across different recall values, averaged over all classes.                                  |
|                              | Intersection over Union (IoU)      | Ratio of the intersection area to the union area of the predicted and ground truth bounding boxes.            |
|                              | Precision                          | Proportion of correctly detected objects among the detected objects.                                          |
|                              | Recall (Sensitivity)               | Proportion of correctly detected objects among the actual objects.                                            |
|                              | F1 Score                           | Harmonic mean of precision and recall.                                                                        |
| **Semantic Segmentation**    | Pixel Accuracy                     | Proportion of correctly classified pixels among the total pixels.                                             |
|                              | Mean Intersection over Union (mIoU)| Average IoU across all classes.                                                                                |
|                              | Frequency Weighted IoU (FWIoU)     | IoU weighted by the frequency of each class.                                                                  |
|                              | Dice Coefficient                   | Measure of overlap between the predicted and ground truth segmentation, similar to F1 score.                  |
| **Instance Segmentation**    | Mean Average Precision (mAP)       | Average precision of detected instances, considering both localization and classification accuracy.           |
|                              | Average Precision (AP)             | Precision averaged over different recall thresholds for each instance.                                        |
|                              | Intersection over Union (IoU)      | Ratio of the intersection area to the union area of the predicted and ground truth segments.                  |
|                              | Precision                          | Proportion of correctly segmented instances among the segmented instances.                                    |
|                              | Recall (Sensitivity)               | Proportion of correctly segmented instances among the actual instances.                                       |
| **Keypoint Detection**       | Percentage of Correct Keypoints (PCK)| Proportion of detected keypoints that are within a certain distance of the true keypoints.                   |
|                              | Average Precision (AP)             | Average precision across different recall values for detected keypoints.                                      |
|                              | Mean Average Precision (mAP)       | Average precision across different keypoint categories.                                                       |
| **Image Generation**         | Inception Score (IS)               | Evaluates the quality of generated images based on the diversity and meaningfulness of the generated content.  |
|                              | Fréchet Inception Distance (FID)   | Measures the similarity between the generated images and real images using statistics of features.            |
|                              | Perceptual Image Quality (PIQ)     | Evaluates the perceptual quality of generated images based on human vision models.                            |
|                              | Structural Similarity Index (SSIM) | Measures the structural similarity between the generated and real images.                                     |
| **Image Super-Resolution**   | Peak Signal-to-Noise Ratio (PSNR)  | Measures the ratio between the maximum possible power of a signal and the power of corrupting noise.          |
|                              | Structural Similarity Index (SSIM) | Measures the structural similarity between the super-resolved and original images.                            |
|                              | Mean Squared Error (MSE)           | Average squared difference between the super-resolved and original image pixels.                              |
| **Depth Estimation**         | Mean Absolute Error (MAE)          | Average of absolute differences between predicted and ground truth depth values.                              |
|                              | Root Mean Squared Error (RMSE)     | Square root of the average of squared differences between predicted and ground truth depth values.            |
|                              | Scale Invariant Log RMSE (SILog)   | Logarithmic measure that is invariant to changes in scale between predicted and ground truth depth values.    |
| **Optical Flow**             | End-Point Error (EPE)              | Average Euclidean distance between the predicted and ground truth optical flow vectors.                       |
|                              | Average Angular Error (AAE)        | Average angular difference between the predicted and ground truth optical flow vectors.                       |
| **Action Recognition**       | Accuracy                           | Proportion of correctly classified actions among the total instances.                                         |
|                              | Precision                          | Proportion of true positive actions among the instances classified as positive.                               |
|                              | Recall (Sensitivity)               | Proportion of true positive actions among the actual positive instances.                                      |
|                              | F1 Score                           | Harmonic mean of precision and recall.                                                                        |
| **Image Captioning**         | BLEU Score                         | Measures the precision of n-grams between generated and reference captions.                                   |
|                              | METEOR                             | Measures harmonic mean of unigram precision and recall, considering synonymy and stemming.                    |
|                              | ROUGE-L                            | Measures the longest matching sequence of words between generated and reference captions.                     |
|                              | CIDEr                              | Measures consensus between generated and reference captions, emphasizing consensus among humans.              |
| **3D Shape Reconstruction**  | Chamfer Distance                   | Measures the average distance between points in the predicted and ground truth 3D shapes.                     |
|                              | Earth Mover's Distance (EMD)       | Measures the distance between two probability distributions over a region (e.g., predicted and ground truth 3D shapes). |
|                              | Intersection over Union (IoU)      | Measures the overlap between predicted and ground truth 3D volumes.                                           |
