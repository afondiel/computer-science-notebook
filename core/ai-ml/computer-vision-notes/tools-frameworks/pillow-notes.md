# Pillow (PIL) - The Python Imaging Library Notes

## Table of Contents (ToC)
  - [Introduction](#introduction)
    - [What's Pillow (PIL)?](#whats-pillow-pil)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [Pillow (PIL) Architecture Pipeline](#pillow-pil-architecture-pipeline)
    - [How Pillow (PIL) works?](#how-pillow-pil-works)
    - [Some hands-on examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)

## Introduction
Pillow (PIL) is a Python library used for opening, manipulating, and saving various image file formats.

### What's Pillow (PIL)?
- A fork of the original Python Imaging Library (PIL).
- Provides extensive file format support.
- Simplifies image processing tasks in Python.

### Key Concepts and Terminology
- **Image Object**: Core object representing an image.
- **Modes**: Represent image types (e.g., RGB, RGBA, L, etc.).
- **Filters**: Operations that can be applied to images (e.g., BLUR, CONTOUR).

### Applications
- Web development (e.g., image manipulation for web applications).
- Data analysis (e.g., processing images for machine learning).
- Graphics design (e.g., automating repetitive image editing tasks).

## Fundamentals
### Pillow (PIL) Architecture Pipeline
- Image Loading
  - Open images using `Image.open()`.
  - Supported formats: JPEG, PNG, BMP, etc.
- Image Processing
  - Apply transformations (resize, crop, rotate).
  - Enhance images (adjust color, contrast, brightness).
- Image Saving
  - Save images using `Image.save()`.
  - Export formats: JPEG, PNG, GIF, etc.

### How Pillow (PIL) works?
- Import the library: `from PIL import Image`.
- Load an image: `image = Image.open('example.jpg')`.
- Perform operations: `image = image.resize((100, 100))`.
- Save the image: `image.save('example_resized.jpg')`.

### Some hands-on examples 
- Opening an image: `Image.open('example.jpg')`.
- Converting image mode: `image.convert('L')`.
- Applying filters: `image.filter(ImageFilter.BLUR)`.
- Cropping an image: `image.crop((10, 10, 100, 100))`.

## Tools & Frameworks
- **Pillow**: Core library for image processing.
- **Jupyter Notebook**: For interactive coding and testing.
- **NumPy**: Often used alongside for numerical operations on images.
- **Matplotlib**: For visualizing images and results.

## Hello World!
```python
from PIL import Image

# Open an image file
image = Image.open('example.jpg')

# Resize the image
image = image.resize((200, 200))

# Save the image
image.save('example_resized.jpg')

# Display the image
image.show()
```

## Lab: Zero to Hero Projects
- **Basic Image Editor**: Create a GUI application to load, edit, and save images.
- **Thumbnail Generator**: Automate the creation of thumbnails for a collection of images.
- **Image Filters Application**: Develop an application to apply various filters and transformations to images.
- [**OCR Tool**: Build a tool that uses Pillow and Tesseract to extract text from images.](https://github.com/afondiel/computer-vision-challenge/blob/main/L0_07_Optical_Character_Recognition_OCR/notebooks/OCR_Pytesseract.ipynb)

## References
- [Python_Imaging_Library - Wikipedia](https://en.wikipedia.org/wiki/Python_Imaging_Library)
- [Pillow (PIL Fork) 10.3.0 documentation](https://pillow.readthedocs.io/en/stable/index.html#)
- [Pillow Tutorial Handbook](https://pillow.readthedocs.io/en/stable/handbook/index.html)
- [Pillow GitHub Repository](https://github.com/python-pillow/Pillow)
- [Pillow Tutorial by Real Python](https://realpython.com/working-with-images-in-python/)
- [Pillow: Working with Images - Python crash courses](https://ehmatthes.github.io/pcc_2e/beyond_pcc/pillow/)
- [Python pillow tutorial - Geeks For Geeks](https://www.geeksforgeeks.org/python-pillow-tutorial/)
  - [Python – Edge Detection using Pillow](https://www.geeksforgeeks.org/python-edge-detection-using-pillow/)
