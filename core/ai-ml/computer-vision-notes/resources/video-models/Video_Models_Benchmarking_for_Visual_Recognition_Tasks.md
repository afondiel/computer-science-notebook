# Video Models Benchmarking for Visual Recognition Tasks

## Table of Contents
  - [Object Tracking](#object-tracking)
  - [Video Captioning](#video-captioning)
  - [3D Reconstruction from Video](#3d-reconstruction-from-video)


## Object Tracking

| SOTA Vision Models | Description | Capability | Performance | Metrics | Datasets |
|--|--|--|--|--|--|
| TransTrack | Transformer-based approach for tracking | Tracks objects across frames | High accuracy on multiple benchmarks | MOTA, IDF1 | MOT17, MOT20 |
| SiamRPN++ | Real-time object tracking using region proposal networks | Accurate and fast object tracking | Good real-time performance | Precision, Success rate | GOT-10k, LaSOT |
| CenterTrack | Uses center points to detect and track objects | Joint detection and tracking | Excellent speed and accuracy | MOTA, IDF1 | MOT17 |
| TrackFormer | Transformer-based framework for joint detection and tracking | Tracks objects directly | High accuracy on crowded scenes | MOTA, IDF1 | MOT17, MOT20 |
| FairMOT | Focuses on fairness in tracking across different object sizes | Joint detection and tracking | Excellent performance in diverse conditions | MOTA, IDF1 | MOT16, MOT20 |
| ByteTrack | Tracker for both high and low-confidence object detections | Handles diverse tracking cases | Good performance on crowded scenes | MOTA, IDF1 | MOT17, MOT20 |
| DeepSORT | Deep learning approach for Simple Online Realtime Tracking | Tracks multiple objects | Consistent in performance across different scenes | MOTA, IDF1 | MOT16, KITTI |
| SiamMask | Combines tracking and segmentation for visual object tracking | Tracks and segments objects | High success rate and precision | Success rate, Precision | VOT2018, LaSOT |
| D3S | Discriminative single-shot object tracker with segmentation | Real-time object tracking with segmentation | Competitive with state-of-the-art methods | Success rate, Precision | VOT2018, GOT-10k |
| MDNet | Multi-Domain Network for tracking across different tasks | Robust object tracking | High performance on multiple datasets | Success rate, Precision | VOT2015, OTB-100 |

---

## Video Captioning

| SOTA Vision Models | Description | Capability | Performance | Metrics | Datasets |
|--|--|--|--|--|--|
| ClipCap | Efficient video captioning using pre-trained vision models | Generates coherent captions | High-quality captions in multiple domains | BLEU, METEOR | MSVD, MSR-VTT |
| VideoBERT | Learns video-language representations | Generates accurate video descriptions | Strong performance in understanding complex actions | BLEU, METEOR | HowTo100M |
| Unified VLP | Joint training of visual and language representations | Generates captions and video descriptions | High quality across tasks | BLEU, CIDEr | MSR-VTT, YouCook2 |
| GPT2 + Visual Encoder | Integrates visual and textual information for video captioning | Accurate video captioning | Good performance on varied video datasets | BLEU, METEOR | MSVD, MSR-VTT |
| MAViL | Multimodal model for video captioning | Handles multimodal video inputs | State-of-the-art performance | BLEU, METEOR | MSR-VTT |
| End-to-End Video Captioning Transformer | Transformer-based model for video captioning | Generates contextual video descriptions | Excellent performance on complex video scenes | CIDEr, BLEU | ActivityNet, MSVD |
| Dense Video Captioning | Generates captions for densely occurring events | Captions multiple events in a single video | High performance on multi-event videos | CIDEr, BLEU | ActivityNet Captions |
| HERO | Hierarchical transformer for video event descriptions | Contextual understanding of video sequences | Top performance on long videos | BLEU, METEOR | TVQA, HowTo100M |
| M3L | Multimodal Multitask Learning for video description | Handles multiple tasks including captioning | High-quality video summaries | CIDEr, BLEU | YouCook2, MSVD |
| VIOLET | Combines vision and language transformers for video captioning | Generates captions with contextual richness | Strong benchmarks on several datasets | CIDEr, BLEU | YouCook2, MSVD |

---

## 3D Reconstruction from Video

| SOTA Vision Models | Description | Capability | Performance | Metrics | Datasets |
|--|--|--|--|--|--|
| DeepV2D | Monocular depth and pose estimation for 3D reconstruction | 3D reconstruction from monocular video | High reconstruction accuracy | RMSE, Depth Error | NYU-Depth, KITTI |
| COLMAP | Traditional multi-view stereo approach | Structure from motion for 3D models | Precise 3D reconstruction | Point accuracy | ETH3D, Tanks and Temples |
| NeuralRecon | Real-time 3D reconstruction from RGB-D video | Real-time performance with neural networks | High accuracy and fast processing | 3D IoU, F-Score | ScanNet, TUM-RGBD |
| BundleFusion | Fuses depth information from multiple frames | Dense 3D reconstruction | Excellent performance with depth sensors | Accuracy, F-Score | ScanNet, NYU-Depth |
| PixelNeRF | Neural Radiance Fields for reconstructing 3D models | Generates detailed 3D models | High-quality results on novel views | PSNR, SSIM | ShapeNet, ScanNet |
| MultiView StereoNet | Deep learning for multi-view stereo reconstruction | Produces dense 3D point clouds | High-quality reconstruction | 3D IoU, F-Score | DTU, Tanks and Temples |
| Voxblox | Real-time volumetric 3D reconstruction | Handles large-scale scenes in real time | Fast and accurate | RMSE, Depth Error | KITTI, TUM-RGBD |
| DISN | Deep Implicit Surface Network for 3D shape reconstruction | Detailed 3D surface reconstruction | High performance on fine details | Chamfer Distance | ShapeNet, KITTI |
| Occupancy Networks | Learning 3D shape representations for reconstruction | Generates high-fidelity 3D models | Good accuracy and robustness | Chamfer Distance, IoU | ShapeNet, KITTI |
| DepthFusion | Combines depth maps from multiple views for 3D reconstruction | Accurate multi-view 3D reconstruction | Excellent accuracy and consistency | RMSE, Depth Error | ScanNet, NYU-Depth |

