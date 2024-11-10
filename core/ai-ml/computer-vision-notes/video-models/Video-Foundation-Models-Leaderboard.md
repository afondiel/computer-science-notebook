# SOTA Video Foundation Model Leaderboard (08/Sep/2024)

**Table of Contents**

- [**Video Captioning Leaderboard**](#video-captioning-leaderboard)
- [**Object Tracking Leaderboard**](#object-tracking-leaderboard)
- [**Video Segmentation Leaderboard**](#video-segmentation-leaderboard)
- [**Action Recognition Leaderboard**](#action-recognition-leaderboard)
- [**Video Super-Resolution Leaderboard**](#video-super-resolution-leaderboard)
- [**Video Style Transfer Leaderboard**](#video-style-transfer-leaderboard)


### **Video Captioning Leaderboard**

| Rank | Model Name | Architecture | Video Input Type | BLEU | Params | Datasets | Release Date |
|--|--|--|--|--|--|--|--|
| 1 | ClipCap | Transformer + Vision-Language Pretraining | Frame Sequences | 62.5 | 340M | MSVD, MSR-VTT | 2021 |
| 2 | VideoBERT | BERT-based Vision-Language Model | Frame Sequences | 60.1 | 418M | HowTo100M, YouCook2 | 2020 |
| 3 | MAViL | Multimodal Attention Vision-Language | Frame Sequences | 58.4 | 392M | MSVD, YouCook2 | 2022 |
| 4 | Unified VLP | Visual and Language Pretraining | Frame Sequences | 59.8 | 313M | YouCook2, MSR-VTT | 2019 |
| 5 | HERO | Hierarchical Transformer | Frame Sequences | 56.7 | 380M | TVQA, HowTo100M | 2020 |
| 6 | Dense Video Captioning | Event-based Captioning | Frame Sequences | 55.4

### **Object Tracking Leaderboard**

| Rank | Model Name | Architecture | Tracking Type | MOTA | Params | FPS | Datasets | Release Date |
|--|--|--|--|--|--|--|--|--|
| 1 | TransTrack | Transformer-based | Online Multi-Object Tracking | 74.5 | 52M | 18 | MOT17 | 2021 |
| 2 | SiamRPN++ | Siamese Network | Single Object Tracking | 73.2 | 38M | 60 | GOT-10k, LaSOT | 2019 |
| 3 | FairMOT | Joint Detection and Tracking | Multi-Object Tracking | 72.3 | 57M | 30 | MOT17, MOT20 | 2020 |
| 4 | ByteTrack | Detection and Tracking | Multi-Object Tracking | 71.6 | 30M | 25 | MOT17 | 2021 |
| 5 | DeepSORT | Deep Learning + Kalman Filter | Multi-Object Tracking | 70.4 | 15M | 20 | MOT17 | 2017 |
| 6 | SiamMask | Siamese Network + Segmentation | Visual Object Tracking | 66.7 | 44M | 55 | VOT2018 | 2018 |
| 7 | CenterTrack | Center-based | Multi-Object Tracking | 69.8 | 38M | 22 | MOT17 | 2020 |
| 8 | TrackFormer | Transformer-based | Multi-Object Tracking | 71.2 | 78M | 16 | MOT20 | 2021 |
| 9 | MDNet | Multi-Domain Network | Single Object Tracking | 65.1 | 30M | 30 | OTB-100, VOT | 2016 |
| 10 | D3S | Discriminative Single-shot | Visual Object Tracking + Segmentation | 67.8 | 35M | 28 | GOT-10k, LaSOT | 2020 |

---

### **Video Segmentation Leaderboard**

| Rank | Model Name | Architecture | Backbone | mIoU | Params | Datasets | Release Date |
|--|--|--|--|--|--|--|--|
| 1 | Swin-L | Swin Transformer | Swin-L | 81.3 | 284M | Cityscapes, ADE20K | 2021 |
| 2 | Mask2Former | Transformer-based | ResNet-50 | 78.9 | 47M | COCO, ADE20K | 2022 |
| 3 | DeepLabV3+ | Atrous Convolution | Xception-71 | 79.2 | 43M | Cityscapes, Pascal VOC | 2018 |
| 4 | PSPNet | Pyramid Scene Parsing | ResNet-101 | 77.3 | 107M | Cityscapes, ADE20K | 2017 |
| 5 | HRNet | High-Resolution Networks | HRNetV2 | 80.4 | 65M | Cityscapes, COCO | 2020 |
| 6 | SegFormer | Transformer-based | Mix-Transformer | 81.6 | 78M | ADE20K, COCO | 2021 |
| 7 | EfficientNet-L | EfficientNet | EfficientNet-L | 77.5 | 62M | Cityscapes, ADE20K | 2021 |
| 8 | UPerNet | Unified Perceptual Parsing | ResNet-101 | 78.1 | 75M | ADE20K | 2019 |
| 9 | FCN | Fully Convolutional Networks | VGG-16 | 70.0 | 134M | PASCAL VOC | 2015 |
| 10 | PointRend | Point-based Segmentation | ResNet-50 | 77.1 | 46M | COCO, ADE20K | 2020 |

---

### **Action Recognition Leaderboard**

| Rank | Model Name | Architecture | Backbone | Top-1 Accuracy | Params | Datasets | Release Date |
|--|--|--|--|--|--|--|--|
| 1 | TimeSformer | Transformer-based | ViT-B | 80.7% | 121M | Kinetics-400, Something-Something | 2021 |
| 2 | SlowFast | 3D CNN | ResNet-50 | 78.2% | 59M | Kinetics-400, AVA | 2020 |
| 3 | VideoSwin | Swin Transformer | Swin-L | 79.5% | 284M | Kinetics-600, Something-Something | 2021 |
| 4 | X3D | Efficient 3D CNN | X3D-M | 77.2% | 3M | Kinetics-400 | 2020 |
| 5 | MViT | Multiscale Vision Transformer | MViT-B | 81.2% | 36M | Kinetics-400, SSv2 | 2021 |
| 6 | TSM | Temporal Shift Module | ResNet-50 | 76.5% | 24M | Kinetics-400, SSv2 | 2019 |
| 7 | SlowFast-R101 | Two-Stream CNN | ResNet-101 | 79.2% | 53M | Kinetics-600, AVA | 2020 |
| 8 | I3D | Inflated 3D CNN | InceptionV1 | 72.9% | 27M | Kinetics-400 | 2018 |
| 9 | R(2+1)D | 3D CNN | ResNet-34 | 75.0% | 32M | Kinetics-400 | 2018 |
| 10 | MoViNet | Efficient CNN | MoViNet-A0 | 72.1% | 2M | Kinetics-400 | 2021 |

---

### **Video Super-Resolution Leaderboard**

| Rank | Model Name | Architecture | Scale Factor | PSNR | Params | Datasets | Release Date |
|--|--|--|--|--|--|--|--|
| 1 | BasicVSR++ | Recurrent Network | x4 | 33.7 dB | 22M | REDS, Vimeo-90k | 2021 |
| 2 | EDVR | Enhanced Deformable ConvNet | x4 | 33.6 dB | 23M | REDS, Vimeo-90k | 2019 |
| 3 | TecoGAN | GAN-based | x4 | 31.9 dB | 16M | Vimeo-90k, REDS | 2020 |
| 4 | RBPN | Recurrent Network | x4 | 33.1 dB | 12M | Vimeo-90k, REDS | 2019 |
| 5 | VSRResNet | ResNet-based | x4 | 30.8 dB | 10M | Vimeo-90k | 2018 |
| 6 | DUF | Temporal Convolution | x4 | 32.1 dB | 12M | Vimeo-90k | 2018 |
| 7 | EDSR | Residual Network | x4 | 31.7 dB | 43M | DIV2K | 2017 |
| 8 | VESPCN | Spatio-Temporal Network | x4 | 30.5 dB | 2M | Vimeo-90k | 2017 |
| 9 | SPMC | Sub-Pixel Motion Comp. | x4 | 30.4 dB | 4M | Vimeo-90k | 2017 |
| 10 | SRGAN | GAN-based | x4 | 29.9 dB | 7M | DIV2K | 2017 |

---

### **Video Style Transfer Leaderboard**

| Rank | Model Name | Architecture | Style Input | Style Fidelity | Params | Datasets | Release Date |
|--|--|--|--|--|--|--|--|
| 1 | STROTSS | Optimization-based | Arbitrary | High | 43M | Video Artistic Dataset | 2020 |
| 2 | AdaIN | Adaptive Instance Norm | Arbitrary | Medium | 3M | Video Artistic Dataset | 2019 |
| 3 | ReCoNet | Real-time CNN | Predefined | Medium | 1M | Video Artistic Dataset | 2019 |
| 4 | MSTNet | Multi-Style Transfer | Predefined | Medium | 2M | Video Artistic Dataset | 2021 |
| 5 | VideoWCT | Wavelet CNN | Arbitrary | Medium | 10M | Video Artistic Dataset | 2020 |
| 6 | ArtisticVAE | Variational Autoencoder | Arbitrary | High | 15M | Video Artistic Dataset | 2021 |
| 7 | DeepVideoStyler | GAN-based | Arbitrary | High | 22M | Video Artistic Dataset | 2020 |
| 8 | FastStyleTransfer | Fast ConvNet | Predefined | Low | 1.5M | COCO | 2017 |
| 9 | CycleGAN | GAN-based | Arbitrary | Medium | 11M | COCO | 2017 |
| 10 | Artisto | Pretrained ConvNet | Predefined | Low | 3M | Video Artistic Dataset | 2017 |

