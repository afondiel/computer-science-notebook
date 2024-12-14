# Visual Prompting - Notes

## Table of Contents
  - [Introduction](#introduction)
  - [Fundamentals](#fundamentals)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)
  - [References](#references-1)

## Introduction
Visual Prompting is a novel method for adapting large-scale models in computer vision with minimal human supervision.

## Fundamentals

- Visual Prompting is inspired by the success of text prompting in natural language processing (NLP), where a single API call can prompt a pre-trained model to perform a new task.
- Visual Prompting leverages the power of pre-trained vision transformers, which can learn from a few visual prompts (such as regions of interest or labels) provided by the user .
- Visual Prompting is faster and easier than conventional labeling, which requires annotating every image in the training set.
- Visual Prompting can be applied to various computer vision tasks, such as object detection, segmentation, classification, and captioning.

## Tools & Frameworks

- Landing AI is a platform that offers Visual Prompting as a capability for building computer vision applications.
- Landing AI's Visual Prompting allows users to specify a visual prompt by painting over object classes they wish the system to detect, using just one or a few images.
- Landing AI's Visual Prompting can transform an unlabeled dataset into a deployed model in minutes, resulting in a simplified, faster, and more user-friendly workflow.

## Hello World!

```python
# Import Landing AI's Visual Prompting module
from landingai import visual_prompting

# Load a pre-trained vision transformer model
model = visual_prompting.load_model("clip")

# Create a visual prompt for detecting cats and dogs
prompt = visual_prompting.create_prompt(["cat", "dog"])

# Apply the prompt to an image
image = visual_prompting.load_image("cat_dog.jpg")
result = model(image, prompt)

# Display the result
visual_prompting.show_result(result)
```

## Lab: Zero to Hero Projects

- Try Visual Prompting on different computer vision tasks and datasets, such as face recognition, scene understanding, or medical imaging.
- Compare the performance of Visual Prompting with conventional labeling and fine-tuning methods.
- Explore the properties of the downstream dataset, prompt design, and output transformation in regard to adaptation performance .
- Experiment with different pre-trained vision transformer models and visual prompt types.

## References

- [INTRODUCING - Visual Prompting - @LandingAI](https://landing.ai/)
- [What is Visual Prompting? - LandingAI](https://landing.ai/blog/what-is-visual-prompting/)
- [Visual Prompting Livestream With Andrew Ng](https://www.youtube.com/watch?v=FE88OOUBonQ)
- [CVPR 2023 Tutorial on Prompting in Vision](https://prompting-in-vision.github.io/)
- [Exploring Visual Prompts for Adapting Large-Scale Models.](https://arxiv.org/abs/2203.17274) .
- [Visual Prompting | Papers With Code.](https://paperswithcode.com/task/visual-prompting) .
- [Andrew Ng’s Landing AI makes it easier to create ... - VentureBeat.](https://venturebeat.com/ai/andrew-ngs-landing-ai-makes-it-easier-to-create-computer-vision-apps-with-visual-prompting/)
- https://prompting-in-vision.github.io/index_cvpr24.html
- https://paperswithcode.com/task/visual-prompting
- https://medium.com/@tenyks_blogger/cvpr-2024-foundation-models-visual-prompting-are-about-to-disrupt-computer-vision-026f2c1c3a2f
- https://support.landing.ai/docs/visual-prompting
- https://research.ibm.com/projects/visual-prompting
- https://github.com/JindongGu/Awesome-Prompting-on-Vision-Language-Model
- https://www.saxifrage.xyz/post/vision-prompting


Lectures & Online Courses: 
- [Prompt Engineering for Vision Models Crash Course - Notes (DeepLearningAI)](https://github.com/afondiel/Prompt-Engineering-for-Vision-Models-DeepLearningAI)
