# Neural Nets Hackers' Notes

## Overview

Short Hackers' notes to neural nets, including how to build neural nets from scratch with large datasets, inferecing a pre-trained model using frameworks API, and finally, finetuning and deploy at scale. 

This short note was inspired from this great article: [Hacker's guide to Neural Networks](https://karpathy.github.io/neuralnets/) by [Andrej Karpathy](https://karpathy.ai/).

## Pipeline For Building Model From Scratch
- **Data Collection:** Gather relevant data for the task.
- **Data Preprocessing:** Clean, normalize, and augment data as needed.
- **Model Architecture Design:** Choose neural network architecture based on the problem.
- **Training Configuration:** Set hyperparameters and optimization techniques.
- **Model Training:** Train the model on the prepared data.
- **Evaluation:** Assess model performance and fine-tune if necessary.

### Tools & Frameworks
- TensorFlow or PyTorch for deep learning.
- Pandas and NumPy for data manipulation.
- Scikit-learn for preprocessing.

### Code Examples
```python
# Data Collection
data = load_data()

# Data Preprocessing
preprocessed_data = preprocess(data)

# Model Architecture Design
model = create_neural_network()

# Training Configuration
config = set_hyperparameters()

# Model Training
trained_model = train_model(model, preprocessed_data, config)

# Evaluation
evaluate_model(trained_model, test_data)
```

## Pipeline For Pre-Trained Models Inference
- **Model Loading:** Load pre-trained models.
- **Input Processing:** Prepare input data for model inference.
- **Inference:** Run the pre-trained model on input data.

### Tools & Frameworks
- TensorFlow Serving or ONNX for model serving.
- Transformers library for pre-trained models.
- Flask or FastAPI for API creation.

### Code Examples
```python
# Model Loading
loaded_model = load_pretrained_model()

# Input Processing
input_data = preprocess_input(raw_data)

# Inference
output = infer(loaded_model, input_data)
```

**Examples using OpenCV dnn API + Caffe pretrained model for Face Detection**

```python
#!/usr/bin/python

#                          License Agreement
#                         3-clause BSD License
## more details can be found: https://github.com/opencv/opencv/tree/4.x/samples/dnn

import cv2
import sys

"""
    @brief None
    1. The Pretrained model files ware already downloaded separately
    2. Load & create Caffe DNN object for the pre-trained model & some defaults parameters 
    3. Preprocessing based on the problem to solve
    4. Start inference phase
    5. Plot results
"""


s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)


# opencv deep neural net api to read compliant pre-trained models such as: caffe, tensorflow pytorch, darknet, onnx...
## 2. Load & create Caffe DNN object for the pre-trained model & some defaults parameters 
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

## 3. Preprocessing based on the problem to solve
# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123] 
conf_threshold = 0.7 # sensivility of the detections

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame,1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from a frame.
    """
    @brief Preprocessing of the image to put in the right format
    
    - frame: image frame from video stream
    - 1.0 : rescaled the img based on the model range
    - (in_width, in_height): input size 
    - mean: mean value subtracted from all the images 
    - swapRB = False : opencv & caffe use the same convention for channel camera 
    - crop = False: resize the image
    """
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB = False, crop = False)
    ## 4. Start inference phase
    # Run a model
    net.setInput(blob)
    # model inference => replay result scoring and prediction
    detections = net.forward()
    ## Debugging
    # Detections:[i,j,k,l] => k: numbers of row, l:number of columns 
    # print(f'detections: {detections}, detections-shape: {detections.shape}')
    # print(f'detections-size: {detections.size}, detections-ndim: {detections.ndim}')
    # break
    ## 5. Plot results
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            # drawing a box
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]),
                                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                                (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # time required to perform inference
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
```

## Pipeline For Pre-Trained Models Fine-Tuning
- **Model Selection:** Choose a pre-trained model suitable for the task.
- **Fine-Tuning Configuration:** Define fine-tuning hyperparameters.
- **Data Preparation:** Acquire additional task-specific data.
- **Fine-Tuning:** Fine-tune the pre-trained model on the new data.
- **Evaluation:** Assess the fine-tuned model's performance.

### Tools & Frameworks
- Hugging Face Transformers for pre-trained models.
- PyTorch or TensorFlow for fine-tuning.
- Scikit-learn for evaluation metrics.

### Code Examples
```python
# Model Selection
pretrained_model = select_pretrained_model()

# Fine-Tuning Configuration
fine_tuning_config = set_fine_tuning_hyperparameters()

# Data Preparation
additional_data = acquire_additional_data()

# Fine-Tuning
fine_tuned_model = fine_tune_model(pretrained_model, additional_data, fine_tuning_config)

# Evaluation
evaluate_fine_tuned_model(fine_tuned_model, validation_data)
```


## Pipeline For Models Deployment at Scale

First Method:

- **Model Serialization:** Save the trained model in a deployable format.
- **Server Setup:** Configure a server for model hosting.
- **API Creation:** Develop an API for model access.

### Tools & Frameworks
- Docker for containerization.
- Kubernetes for orchestration.
- Flask or FastAPI for API creation.

### Code Examples
```python
# Model Serialization
serialize_model(trained_model)

# Server Setup
setup_server()

# API Creation
create_api()
```

Second Method: Deployment at Scale

- **Containerization:** Package the model and its dependencies into containers.
- **Orchestration Configuration:** Define orchestration settings for deploying multiple instances.
- **Load Balancing:** Distribute incoming requests efficiently across deployed instances.
- **Monitoring Setup:** Implement monitoring for performance and resource usage.
- **Scalability Planning:** Design the system for easy horizontal scaling.

### Tools & Frameworks
- Docker for containerization.
- Kubernetes for orchestration and scaling.
- Nginx or HAProxy for load balancing.
- Prometheus and Grafana for monitoring.

### Code Examples
```python
# Containerization
dockerize_model(model)

# Orchestration Configuration
configure_kubernetes()

# Load Balancing
implement_load_balancer()

# Monitoring Setup
setup_monitoring(prometheus_config, grafana_config)

# Scalability Planning
design_for_scaling()
```

## References

- [Hacker's guide to Neural Networks -  karpathy.ai](https://karpathy.github.io/neuralnets/)
- [Neural Networks: Zero to Hero - karpathy.ai](https://karpathy.ai/zero-to-hero.html) 
- Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). Deep learning (Vol. 1). MIT press Cambridge.
- Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning. 
- Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
- Howard, J., et al. (2018). Universal language model fine-tuning for text classification. arXiv preprint arXiv:1801.06146.
- Burns, B., et al. (2017). Design Patterns for Container-Base Distributed Systems. Retrieved from https://www.docker.com/blog/design-patterns-for-container-based-distributed-systems/.
- Burns, B., & Vohra, A. (2016). Kubernetes: Up and Running. O'Reilly Media.


