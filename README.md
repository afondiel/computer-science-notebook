![GitHub repo size](https://img.shields.io/github/repo-size/afondiel/research-notes) ![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/t/afondiel/research-notes/master) ![Packagist Stars](https://img.shields.io/github/stars/afondiel/research-notes.svg) ![Packagist forks](https://img.shields.io/github/forks/afondiel/research-notes.svg) 

# Research Notes

## Overview

As a Software Engineer Consultant, I have had the opportunity to work on a variety of topics in the **automotive** and **agriculture** industries, including electrification and sustainability, safety and comfort, and more.

So, I decided to compile some resources from my previous and ongoing work that software engineers looking to get into any tech industry might find useful, including core skills, projects, and tools.

## Motivation

The big picture behind this project is to create a "Hello World" Notebook, sort of [Whole "Software Engineering" Catalog](https://en.wikipedia.org/wiki/Whole_Earth_Catalog), that Software Engineers, Developers and Computer Science Students can use as an aide-memoire or a "prompt" for their daily work, interview preps and study checklist, etc.

## Why Use This Notebook?

- 🚀 Accelerate Your Learning: Dive into a treasure trove of knowledge curated by passionate software engineers for fellow developers.
- 📦 Wide-Ranging Resources: From algorithms to data structures, design patterns to coding best practices, our notebook covers it all.
- 📝 Interactive Learning: Discover engaging code examples, real-world applications, and interactive challenges to solidify your understanding.
- 🌐 Community-Driven: Join a vibrant community of developers like yourself, all working together to shape the future of this invaluable resource.


## Topics 

Topics covered include requirements, design, programming languages, and testing. You may also find some research papers and notebooks about AI, [Prompt Engineering & LLMs](https://github.com/afondiel/ChatGPT-Prompt-Engineering-DeepLearningAI), ML, DL, NLP, IoT, robotics, and more.

- [agriculture](https://github.com/afondiel/research-notes/tree/master/agriculture)
- [ai](https://github.com/afondiel/research-notes/tree/master/ai)
- [automotive](https://github.com/afondiel/research-notes/tree/master/automotive)
- [computer-vision-notes](https://github.com/afondiel/research-notes/tree/master/computer-vision-notes)
- [control-law](https://github.com/afondiel/research-notes/tree/master/control-law)
- [courses](https://github.com/afondiel/research-notes/tree/master/control-law)
- [cyber-security](https://github.com/afondiel/research-notes/tree/master/courses)
- [datascience-notes](https://github.com/afondiel/research-notes/tree/master/datascience-notes)
- [design-patterns-notes](https://github.com/afondiel/research-notes/tree/master/design-patterns-notes)
- [devops-notes](https://github.com/afondiel/research-notes/tree/master/devops-notes)
- [embedded-systems](https://github.com/afondiel/research-notes/tree/master/embedded-systems)
- [iot](https://github.com/afondiel/research-notes/tree/master/iot)
- [job-interview-notes](https://github.com/afondiel/research-notes/tree/master/job-interview-notes)
- [outro](https://github.com/afondiel/research-notes/tree/master/outro)
- [programming](https://github.com/afondiel/research-notes/tree/master/programming)
- [project-management](https://github.com/afondiel/research-notes/tree/master/project-management)
- [quantum-computing](https://github.com/afondiel/research-notes/tree/master/quantum-computing)
- [regex-notes](https://github.com/afondiel/research-notes/tree/master/regex-notes)
- [robotics](https://github.com/afondiel/research-notes/tree/master/robotics)
- [signal-processing](https://github.com/afondiel/research-notes/tree/master/signal-processing)
- [sw-design-architecture](https://github.com/afondiel/research-notes/tree/master/sw-design-architecture)
- [sw-documentation-convention](https://github.com/afondiel/research-notes/tree/master/sw-documentation-convention)
- [sw-methodology](https://github.com/afondiel/research-notes/tree/master/sw-methodology)
- [sw-standards](https://github.com/afondiel/research-notes/tree/master/sw-standards)
- [sw-testing](https://github.com/afondiel/research-notes/tree/master/sw-testing)
- [vcs](https://github.com/afondiel/research-notes/tree/master/vcs)
- [web](https://github.com/afondiel/research-notes/tree/master/web)

## Industry

- [Automotive](https://github.com/afondiel/research-notes/tree/master/automotive)
- [Agriculture](https://github.com/afondiel/research-notes/tree/master/agriculture)

## Note & Content Generation

**Generate template files and folders with [hello_world.py](hello_world.py) tool**. 

The tool generate the note structure in 3 steps:

1. create a note template 

```python
def note_content_template(topic):
    pass
```
2. Then creates the repos `docs` & `lab` 

```python
def create_repo(repo_name):
    pass
```
3. Finally, it generates the entire template files

```python
def note_gen(topic_name):
    pass
```

Try it out! by running the command below. Choose a `topic`, and create your first note:

```python
python hello_world.py topic-name
```

output:

```
E:.
│   topic-notes.md
│
├───docs
└───lab
```

**ChatGPT/Bard Prompt for Content Generation**

After generating your first (empty) note you can fill it out the content by using the prompt below (if you're using Copilot, It might do trick as well)

```
I'm going to train you write to a short-form summarization.

The summarization has the following format: 
`
# {Topic} - Notes
## Table of Contents (ToC)
## Overview
## Applications
## Tools & Frameworks
## Hello World!
## References
`
Here are the rules: 
- The `##Overview` heading has to be one line sentence
- Organize the following headings in a list of 5-6 succinct bullets 
- Do not use hashtags and emojis. Ever.
- The `## Hello World! ` has to be a code
- replace `{Topic}` by the topic name 

I am going to give you a topic and you will summarize the topic, and output the summarization in markdown format.

Do you understand?

```

Transition:

```
Topic={Topic}
``` 

The final output should be a something like [this](#)!

Notice: this prompt might not work in the attempt depending on the what AI chat you're using. Just hit `regenerate` to the LLM the time to "think" if that the case

## Contributing

There are still many topics and industry to be added, so feel free to submit a pull request if you have any suggestions or cool ideas that may help this project grow.

Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) to see how you can help this project grow and make learning accessible to all.


`Some to-do tasks:`

- Add guidelines of "essential resources to get into a specific industry/company"
- Add MIT LICENSE
- Sort notes based on tech industries : Telecommunications & Network, Aerospace, Defense, Naval-Maritime ...
- Create a pipeline to generate a 'minimum' note content using [hello_world_prompt.md](./hello_world_prompt.md) + [OpenAI API](https://openai.com/blog/openai-api) / [Bard API ](https://www.googlecloudcommunity.com/gc/AI-ML/Google-Bard-API/m-p/538517#M1526)  

## LICENSE

[MIT](https://en.wikipedia.org/wiki/MIT_License)
--

>As a lifelong learner and someone who believes that **free education** will make the world a better a place, I hope you find this project useful and inspiring as I do.
>
>Cheers,
>
>[@Muntu](https://github.com/afondiel)


