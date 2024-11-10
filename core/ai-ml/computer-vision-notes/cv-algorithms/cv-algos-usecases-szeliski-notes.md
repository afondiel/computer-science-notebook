# Computer Vision Algorithms and Applications


## Overview

This is a non-exhautive list of cv algorithms and some commun use cases from the book of  [Richard Szeliski](https://github.com/afondiel/cs-books/blob/main/computer-vision/Computer-Vision-Algorithms-and-Applications-Richard-Szeliski-2010.pdf).



## Table Of Contents

```
1. Introduction
1.1 What is computer vision?    
1.2 A brief history
1.3 Book overview  
1.4 Sample syllabus 
1.5 A note on notation
1.6 Additional reading
1. Image formation 
2.1 Geometric primitives and transformations
2.1.1 Geometric primitives 
2.1.2 2D transformations 
2.1.3 3D transformations
2.1.4 3D rotations
2.1.5 3D to 2D projections 
2.1.6 Lens distortions 
2.2 Photometric image formation
2.2.1 Lighting 
2.2.2 Reflectance and shading
2.2.3 Optics 
2.3 The digital camera
2.3.1 Sampling and aliasing 
2.3.2 Color 
2.3.3 Compression
xiv Computer Vision: Algorithms and Applications (September 3, 2010 draft)
2.4 Additional reading
2.5 Exercises  
1. Image processing 
3.1 Point operators   
3.1.1 Pixel transforms
3.1.2 Color transforms  
3.1.3 Compositing and matting 
3.1.4 Histogram equalization   
3.1.5 Application: Tonal adjustment  
3.2 Linear filtering   
3.2.1 Separable filtering   
3.2.2 Examples of linear filtering 
3.2.3 Band-pass and steerable filters  
3.3 More neighborhood operators   
3.3.1 Non-linear filtering   
3.3.2 Morphology  
3.3.3 Distance transforms   
3.3.4 Connected components   
3.4 Fourier transforms 
3.4.1 Fourier transform pairs   
3.4.2 Two-dimensional Fourier transforms  
3.4.3 Wiener filtering  
3.4.4 Application: Sharpening, blur, and noise removal   
3.5 Pyramids and wavelets  
3.5.1 Interpolation  
3.5.2 Decimation 
3.5.3 Multi-resolution representations  
3.5.4 Wavelets 
3.5.5 Application: Image blending  
3.6 Geometric transformations   
3.6.1 Parametric transformations 
3.6.2 Mesh-based warping   
3.6.3 Application: Feature-based morphing  
3.7 Global optimization 
3.7.1 Regularization  
3.7.2 Markov random fields   
3.7.3 Application: Image restoration  
Contents
3.8 Additional reading 192                                                      
3.9 Exercises   194
1. Feature detection and matching 205
4.1 Points and patches 207
4.1.1 Feature detectors  209
4.1.2 Feature descriptors   222
4.1.3 Feature matching  225
4.1.4 Feature tracking  235
4.1.5 Application: Performance-driven animation  237
4.2 Edges  238
4.2.1 Edge detection  238
4.2.2 Edge linking  244
4.2.3 Application: Edge editing and enhancement  249
4.3 Lines  250
4.3.1 Successive approximation 250
4.3.2 Hough transforms  251
4.3.3 Vanishing points  254
4.3.4 Application: Rectangle detection  257
4.4 Additional reading 257
4.5 Exercises   259
1. Segmentation 267
5.1 Active contours   270
5.1.1 Snakes   270
5.1.2 Dynamic snakes and CONDENSATION  276
5.1.3 Scissors   280
5.1.4 Level Sets 281
5.1.5 Application: Contour tracking and rotoscoping   282
5.2 Split and merge   284
5.2.1 Watershed 284
5.2.2 Region splitting (divisive clustering)  286
5.2.3 Region merging (agglomerative clustering)  286
5.2.4 Graph-based segmentation 286
5.2.5 Probabilistic aggregation 288
5.3 Mean shift and mode finding   289
5.3.1 K-means and mixtures of Gaussians  289
5.3.2 Mean shift 292
xvi Computer Vision: Algorithms and Applications (September 3, 2010 draft)
5.4 Normalized cuts   296
5.5 Graph cuts and energy-based methods  300
5.5.1 Application: Medical image segmentation  304
5.6 Additional reading 305
5.7 Exercises   306
1. Feature-based alignment 309
6.1 2D and 3D feature-based alignment  311
6.1.1 2D alignment using least squares  312
6.1.2 Application: Panography 314
6.1.3 Iterative algorithms   315
6.1.4 Robust least squares and RANSAC  318
6.1.5 3D alignment  320
6.2 Pose estimation   321
6.2.1 Linear algorithms  322
6.2.2 Iterative algorithms   324
6.2.3 Application: Augmented reality  326
6.3 Geometric intrinsic calibration   327
6.3.1 Calibration patterns   327
6.3.2 Vanishing points  329
6.3.3 Application: Single view metrology  331
6.3.4 Rotational motion   332
6.3.5 Radial distortion  334
6.4 Additional reading 335
6.5 Exercises   336
1. Structure from motion 343
7.1 Triangulation   345
7.2 Two-frame structure from motion 347
7.2.1 Projective (uncalibrated) reconstruction  353
7.2.2 Self-calibration  355
7.2.3 Application: View morphing  357
7.3 Factorization   357
7.3.1 Perspective and projective factorization  360
7.3.2 Application: Sparse 3D model extraction  362
7.4 Bundle adjustment 363
7.4.1 Exploiting sparsity   364
7.4.2 Application: Match move and augmented reality   368
Contents xvii
7.4.3 Uncertainty and ambiguities  370
7.4.4 Application: Reconstruction from Internet photos   371
7.5 Constrained structure and motion 374
7.5.1 Line-based techniques   374
7.5.2 Plane-based techniques   376
7.6 Additional reading 377
7.7 Exercises 
1. Dense motion estimation 381
8.1 Translational alignment  384
8.1.1 Hierarchical motion estimation  387
8.1.2 Fourier-based alignment 388
8.1.3 Incremental refinement   392
8.2 Parametric motion 398
8.2.1 Application: Video stabilization  401
8.2.2 Learned motion models   403
8.3 Spline-based motion  404
8.3.1 Application: Medical image registration  408
8.4 Optical flow   409
8.4.1 Multi-frame motion estimation  413
8.4.2 Application: Video denoising  414
8.4.3 Application: De-interlacing  415
8.5 Layered motion   415
8.5.1 Application: Frame interpolation  418                                     
8.5.2 Transparent layers and reflections  419
8.6 Additional reading 421
8.7 Exercises   422
1. Image stitching 427
9.1 Motion models   430
9.1.1 Planar perspective motion 431
9.1.2 Application: Whiteboard and document scanning   432
9.1.3 Rotational panoramas   433
9.1.4 Gap closing 435
9.1.5 Application: Video summarization and compression 436
9.1.6 Cylindrical and spherical coordinates  438
9.2 Global alignment 441
9.2.1 Bundle adjustment   441
xviii Computer Vision: Algorithms and Applications (September 3, 2010 draft)
9.2.2 Parallax removal  445
9.2.3 Recognizing panoramas 446
9.2.4 Direct vsfeature-based alignment  450
9.3 Compositing   450
9.3.1 Choosing a compositing surface  451
9.3.2 Pixel selection and weighting (de-ghosting)  453
9.3.3 Application: Photomontage  459
9.3.4 Blending 459
9.4 Additional reading 462
9.5 Exercises   463
1.   Computational photography 467
10.1 Photometric calibration  470
10.1.1 Radiometric response function  470
10.1.2 Noise level estimation   473
10.1.3 Vignetting 474
10.1.4 Optical blur (spatial response) estimation  476
10.2 High dynamic range imaging   479
10.2.1 Tone mapping  487
10.2.2 Application: Flash photography  494
10.3 Super-resolution and blur removal 497
10.3.1 Color image demosaicing 502
10.3.2 Application: Colorization 504
10.4 Image matting and compositing   505
10.4.1 Blue screen matting   507
10.4.2 Natural image matting   509
10.4.3 Optimization-based matting  513
10.4.4 Smoke, shadow, and flash matting  516
10.4.5 Video matting  518
10.5 Texture analysis and synthesis   518
10.5.1 Application: Hole filling and inpainting  521
10.5.2 Application: Non-photorealistic rendering  522
10.6 Additional reading 524
10.7 Exercises   526
1.    correspondence 533
11.1 Epipolar geometry 537
11.1.1 Rectification  538
Contents xix
11.1.2 Plane sweep 540
11.2 Sparse correspondence  543
11.2.1 3D curves and profiles   543
11.3 Dense correspondence  545
11.3.1 Similarity measures   546
11.4 Local methods   548
11.4.1 Sub-pixel estimation and uncertainty  550
11.4.2 Application: Stereo-based head tracking  551
11.5 Global optimization 552
11.5.1 Dynamic programming   554
11.5.2 Segmentation-based techniques  556
11.5.3 Application: Z-keying and background replacement 558
11.6 Multi-view stereo 558
11.6.1 Volumetric and 3D surface reconstruction  562
11.6.2 Shape from silhouettes   567
11.7 Additional reading 570
11.8 Exercises   571
1.   3D reconstruction 577
12.1 Shape from X   580
12.1.1 Shape from shading and photometric stereo  580
12.1.2 Shape from texture   583
12.1.3 Shape from focus  584
12.2 Active rangefinding 585
12.2.1 Range data merging   588
12.2.2 Application: Digital heritage  590
12.3 Surface representations  591
12.3.1 Surface interpolation   592
12.3.2 Surface simplification   594
12.3.3 Geometry images  594
12.4 Point-based representations   595
12.5 Volumetric representations   596
12.5.1 Implicit surfaces and level sets  596
12.6 Model-based reconstruction   598
12.6.1 Architecture 598
12.6.2 Heads and faces  601
12.6.3 Application: Facial animation  603
12.6.4 Whole body modeling and tracking  605
xx Computer Vision: Algorithms and Applications (September 3, 2010 draft)
12.7 Recovering texture maps and albedos  610
12.7.1 Estimating BRDFs   612
12.7.2 Application: 3D photography  613
12.8 Additional reading 614
12.9 Exercises   616
1.   Image-based rendering 619
13.1 View interpolation 621
13.1.1 View-dependent texture maps  623
13.1.2 Application: Photo Tourism  624
13.2 Layered depth images  626
13.2.1 Impostors, sprites, and layers  626
13.3 Light fields and Lumigraphs   628
13.3.1 Unstructured Lumigraph 632
13.3.2 Surface light fields   632
13.3.3 Application: Concentric mosaics  634
13.4 Environment mattes 634
13.4.1 Higher-dimensional light fields  636
13.4.2 The modeling to rendering continuum  637
13.5 Video-based rendering  638
13.5.1 Video-based animation   639
13.5.2 Video textures  640
13.5.3 Application: Animating pictures  643
13.5.4 3D Video 643
13.5.5 Application: Video-based walkthroughs  645
13.6 Additional reading 648
13.7 Exercises   650
1.   Recognition 655
14.1 Object detection   658
14.1.1 Face detection  658
14.1.2 Pedestrian detection   666
14.2 Face recognition   668
14.2.1 Eigenfaces 671
14.2.2 Active appearance and 3D shape models  679
14.2.3 Application: Personal photo collections  684
14.3 Instance recognition 685
14.3.1 Geometric alignment   686
Contents xxi
14.3.2 Large databases  687
14.3.3 Application: Location recognition  693
14.4 Category recognition  696
14.4.1 Bag of words  697
14.4.2 Part-based models   701
14.4.3 Recognition with segmentation  704
14.4.4 Application: Intelligent photo editing  709
14.5 Context and scene understanding 712
14.5.1 Learning and large image collections  714
14.5.2 Application: Image search 717
14.6 Recognition databases and test sets 718
14.7 Additional reading 722
14.8 Exercises   725
1.  Conclusion 731
A Linear algebra and numerical techniques 735
A.1 Matrix decompositions  736
A.1.1 Singular value decomposition  736
A.1.2 Eigenvalue decomposition 737
A.1.3 QR factorization  740
A.1.4 Cholesky factorization   741
A.2 Linear least squares 742
A.2.1 Total least squares   744
A.3 Non-linear least squares  746
A.4 Direct sparse matrix techniques   747
A.4.1 Variable reordering   748
A.5 Iterative techniques 748
A.5.1 Conjugate gradient   749
A.5.2 Preconditioning  751
A.5.3 Multigrid 753
B Bayesian modeling and inference 755
B.1 Estimation theory 757
B.1.1 Likelihood for multivariate Gaussian noise  757
B.2 Maximum likelihood estimation and least squares  759
B.3 Robust statistics   760
B.4 Prior models and Bayesian inference  762
B.5 Markov random fields  763
xxii Computer Vision: Algorithms and Applications (September 3, 2010 draft)
B.5.1 Gradient descent and simulated annealing
B.5.2 Dynamic programming  
B.5.3 Belief propagation  
B.5.4 Graph cuts
B.5.5 Linear programming  
B.6 Uncertainty estimation (error analysis) 
C Supplementary material
C.1 Data sets 
C.2 Software 
C.3 Slides and lectures
C.4 Bibliography                                                      
References
```

## References

- Book : [Computer Vision Algorithms and Applications by Richard Szeliski 2010](https://github.com/afondiel/cs-books/blob/main/computer-vision/Computer-Vision-Algorithms-and-Applications-Richard-Szeliski-2010.pdf)


