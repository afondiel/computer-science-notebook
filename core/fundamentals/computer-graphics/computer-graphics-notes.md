# Computer Graphics - Notes

## Table of Contents (ToC)

  - [Introduction](#introduction)
  - [Key Concepts](#key-concepts)
    - [Common Misconceptions](#common-misconceptions)
  - [Why It Matters / Relevance](#why-it-matters--relevance)
  - [Learning Map (Architecture Pipeline)](#learning-map-architecture-pipeline)
  - [Framework / Key Theories or Models](#framework--key-theories-or-models)
  - [How Computer Graphics Works](#how-computer-graphics-works)
  - [Methods, Types \& Variations](#methods-types--variations)
  - [Self-Practice / Hands-On Examples](#self-practice--hands-on-examples)
  - [Pitfalls \& Challenges](#pitfalls--challenges)
  - [Feedback \& Evaluation](#feedback--evaluation)
  - [Tools, Libraries \& Frameworks](#tools-libraries--frameworks)
    - [Comparison of Tools](#comparison-of-tools)
  - [Hello World! (Practical Example)](#hello-world-practical-example)
  - [Advanced Exploration](#advanced-exploration)
  - [Zero to Hero Lab Projects](#zero-to-hero-lab-projects)
  - [Continuous Learning Strategy](#continuous-learning-strategy)
  - [References](#references)

---

## Introduction
Computer graphics involves creating, manipulating, and rendering visual images using computational methods.

---

## Key Concepts
- **Rendering**: Process of generating a 2D image from a 3D model.
- **Rasterization**: Conversion of vector graphics to a pixel grid.
- **Shading**: Technique used to add depth and realism to objects.
- **Transformation**: Applying mathematical operations to objects (rotation, translation, scaling).
- **Texture Mapping**: Applying a 2D image to a 3D object’s surface.

### Common Misconceptions
- People often confuse **2D graphics** with **3D graphics**. The former focuses on flat, two-dimensional images, while the latter builds and manipulates objects in three dimensions.
- **Rendering time** is often underestimated, especially for complex scenes.

---

## Why It Matters / Relevance
- Used in **movies** and **video games** for realistic visual effects.
- Critical in **virtual reality** and **augmented reality** applications.
- Supports **scientific visualization**, which helps researchers visualize complex data.

---

## Learning Map (Architecture Pipeline)

```mermaid
graph LR;
    Input(Data)--->Processing(Graphics Algorithms)-->Rendering-->Output(Images/Frames)
```
Description:

- **Input**: Data such as 3D models, textures, and lighting information.
- **Processing**: Algorithms like rasterization, shading, and lighting.
- **Rendering**: The final process to convert the data into a 2D image.
- **Output**: The rendered frames displayed on the screen.

---

## Framework / Key Theories or Models
- **Phong Shading Model**: A technique for adding highlights and shading to give 3D objects a smooth appearance.
- **Ray Tracing**: A rendering technique that simulates light paths to create highly realistic images.
- **Bezier Curves**: Used in 2D graphics for smooth curve representation.

---

## How Computer Graphics Works
1. 3D models are created using geometric shapes like triangles and polygons.
2. These models are processed with lighting and texture algorithms.
3. The **renderer** translates the 3D model into a 2D image, adjusting for perspective, lighting, and materials.
4. The result is displayed as a 2D image on the screen, often in real-time for applications like video games.

---

## Methods, Types & Variations
- **Raster Graphics**: Images created from a grid of pixels.
- **Vector Graphics**: Images formed using paths defined by mathematical formulas.
- **2D vs. 3D Graphics**: Two-dimensional images vs. models that occupy three-dimensional space.
- **Real-time Rendering**: Used in games for fast frame rendering.
- **Pre-rendering**: Used in film, where rendering can take longer for higher quality.

---

## Self-Practice / Hands-On Examples
- **Create a 3D model** of a simple object (e.g., a cube) and render it using basic shading.
- **Explore texture mapping** by applying textures to different 3D models.
- **Implement transformations** such as scaling, rotating, or translating objects in a scene.

---

## Pitfalls & Challenges
- **Long rendering times** for complex models.
- **Lighting issues** can cause unrealistic visuals if not handled properly.
- **Hardware limitations** can affect real-time rendering performance.

---

## Feedback & Evaluation
- **Self-explanation test**: Explain how ray tracing works to a beginner.
- **Peer review**: Share a 3D model and its rendered output with peers for critique.
- **Real-world simulation**: Try building a simple video game scene with real-time rendering.

---

## Tools, Libraries & Frameworks
- **OpenGL**: A cross-language, cross-platform API for rendering 2D and 3D vector graphics.
- **Blender**: A powerful, free 3D modeling, animation, and rendering software.
- **Unreal Engine**: Used for game development and real-time graphics rendering.

### Comparison of Tools
- **OpenGL**: Low-level, flexible but requires more code.
- **Blender**: Full-featured but has a steep learning curve.
- **Unreal Engine**: Excellent for real-time rendering but resource-intensive.

---

## Hello World! (Practical Example)
```python
# Simple OpenGL program to render a triangle
import OpenGL.GL as gl
import OpenGL.GLUT as glut

def draw_triangle():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glBegin(gl.GL_TRIANGLES)
    gl.glVertex2f(-0.5, -0.5)
    gl.glVertex2f(0.5, -0.5)
    gl.glVertex2f(0.0, 0.5)
    gl.glEnd()
    gl.glFlush()

glut.glutCreateWindow('Hello Triangle')
glut.glutDisplayFunc(draw_triangle)
glut.glutMainLoop()
```

---

## Advanced Exploration
- **Real-time Ray Tracing**: NVIDIA RTX documentation and tutorials.
- **GPU Programming**: Learn about parallel computing with CUDA and OpenCL.
- **Procedural Generation**: Explore techniques for creating textures, models, and worlds dynamically.

---

## Zero to Hero Lab Projects
- Build a **3D rendering engine** that takes a simple scene description and renders it using basic lighting and shading.
- Create a **simple video game** that involves real-time rendering of 3D models and animations.
- Model and render an **interactive virtual environment** with texture mapping and lighting effects.

---

## Continuous Learning Strategy
- Next steps:
  - Learn about **GPU programming** for faster rendering.
  - Study advanced rendering techniques like **global illumination**.
- Explore related topics:
  - **Animation**: Adding movement to 3D models.
  - **Augmented Reality**: Merging computer graphics with real-world environments.

---

## References

- **"Computer Graphics: Principles and Practice" by Foley et al.**: The standard textbook on computer graphics.
- **OpenGL SuperBible**: Comprehensive guide for OpenGL.
- **Blender Documentation**: In-depth guides and tutorials for Blender users.
- **NVIDIA RTX**: Resources and tutorials on ray tracing and advanced rendering techniques.

Lecture & Tutorials:
- [OpenGL Course - Create 3D and 2D Graphics With C++](https://www.youtube.com/watch?v=45MIykWJ-C4)


