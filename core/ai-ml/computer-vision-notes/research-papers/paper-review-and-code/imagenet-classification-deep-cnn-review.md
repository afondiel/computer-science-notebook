# ImageNet Classification with Deep Convolutional Neural Networks - Paper Review


**Introduction**

Deep convolutional neural networks (CNNs) have achieved state-of-the-art results on many image classification tasks. However, training such networks is challenging due to the large number of parameters involved and the difficulty of obtaining labeled training data.

**Problem and Methodology**

The authors propose a new method for training CNNs on large datasets. Their method uses a combination of techniques, including:

* **Data augmentation:** The authors use data augmentation to increase the size and diversity of the training dataset.
* **Dropout:** The authors use dropout to prevent the network from overfitting the training data.
* **GPU acceleration:** The authors use GPU acceleration to speed up the training process.

**Architecture pipeline**

The authors' CNN architecture consists of five convolutional layers, followed by two fully connected layers. The convolutional layers use rectified linear unit (ReLU) activations, and the fully connected layers use softmax activations.

**Findings**

The authors evaluate their CNN on the ImageNet classification task. Their CNN achieves a top-5 error rate of 15.3%, which is state-of-the-art at the time of publication.

**Conclusion**

The authors have presented a new method for training CNNs on large datasets. Their method achieves state-of-the-art results on the ImageNet classification task.

**Authors' names and organizations**

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton, University of Toronto


## References

- Paper: 
  - [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- Paper with code:
  - [Code & Benchmarks](https://paperswithcode.com/dataset/imagenet) 
