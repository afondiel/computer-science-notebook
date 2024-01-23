# Contributor Guidelines

**Contribute to Excellence**

- Are you passionate about sharing your expertise and making learning accessible to all? Join us in our mission to empower software engineers worldwide. Whether you're a seasoned pro or just starting your journey, your contributions are invaluable. Together, we can create a resource that will benefit countless aspiring and experienced developers.

**How to Contribute**

- Getting started is easy! Dive into our open issues, propose new topics, enhance existing content, or collaborate with our community of like-minded contributors. Your contributions will not only help others but also establish your presence in the ever-expanding world of computer science education.

Join us in shaping the future of software engineering education and be a part of something truly remarkable. Start today and become an integral part of this exciting journey!

You can contribute in many ways :

## Notes Structure

**Note Template**

Your note folder should have the following structure:

```
E:.
│   topic-notes.md
│   readme.md
│
├───docs
└───lab
```

To generate/create a note files, you can either:

**Customize your note manually**

- Create a new folder with the "topic-name"
- Inside the root folder (topic-name), add: 
  - 2 other subfolders: `docs`, `lab`
  - 1 markdown file for the note: `topic-notes.md`
  - 1 `readme.md`for notes with slight description

**Or use the [hello_world.py](hello_world.py) tool** (Please refer to the section [Note & Content Generation](#note--content-generation), for more details).


### `topic-notes.md` Format

```
# {Topic} - Notes
## Table of Contents (ToC)
## Introduction
### What's `topic`?
### Applications
## Fundamentals
### How `topic` works?
#### Topic Architecture Pipeline (image/mermaid diagram, ...)
## Tools & Frameworks
## Hello World!
## Lab: Zero to Hero Projects
## References
```

### `readme.md` Format

```
# Machine Learning Notes

![](img)

(Src: link)

## Overview

This is <topic> "Hello World" resources.

## Contributing

Please refer to this [CONTRIBUTING.md](../../CONTRIBUTING.md) file.

> ### “Any funny/interesting quote/citation about the topic” — @Author
```

### `Lab` Content

Notebooks, source code, coding playground link such: leetcode, replit ...

If you are submiting a source code refer to the file below : 

- [CODING_CONVENTIONS.md](./sw-documentation-convention/CODING_CONVENTIONS.md)
- [CODING_CONVENTIONS_C++.md](./sw-documentation-convention/CODING_CONVENTIONS_C++.md)

### `Docs` Content

Documentation, research papers ...


## Note & Content Generation

**Generate files and folders with [hello_world.py](hello_world.py) tool**. 

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
**Usage**
```
Usage: python hello_world.py [options] <NotePath> <NoteName>

<NotePath>
        Specify a path if you want to generate the note in a different location
<NoteName>
        Note/Topic name
Options:
        -v, --verbose: print debug informationn
        -a, --all: display the list of current notes
        -h, --help: see usage
        -f, --file: add new file
```

Try it out! by running the command below. Choose a `topic`, and create your first note:

```python
python hello_world.py topic-name
```

output:

```
E:.
│   topic-notes.md
│   readme.md
│
├───docs
└───lab
```

**ChatGPT/Bard Prompt for Content Generation**

After generating your first (empty) note you can fill it out the content by using the prompt below (if you're using Copilot, It might do trick as well)

```
I'm going to train you how to write a short-form summary in academic style.

The summary has the following format:

"""
# {Topic} - Notes
## Table of Contents (ToC)
## Introduction
### What's {topic}?
### Key Concepts and Terminology
### Applications
## Fundamentals
### {Topic} Architecture Pipeline (image/mermaid diagram, ...)
### How {topic} works?
### Some hands-on examples 
## Tools & Frameworks
## Hello World!
## Lab: Zero to Hero Projects
## References
"""

Here are the rules:
- Generate a table of contents based on summary format given above
- Replace `{Topic}` with the name of the topic I will give you
- The `Introduction` section shall be a short one-line sentence
- Use 3-5 succinct bullet lists for each section of the summary.
- The `Hello World!` section shall be a code snippet
- Do not use hashtags and emojis. Ever.
- output the summary in markdown format
- add the references in `References` section
- this summary is to be shared with people who want to explore {topic} realm, so make it concise and easy to understand
- 

I am going to give you a topic name and you write a summary based on the topic.

Do you understand?
```

Transition:

```
Topic={Topic}
``` 

The final output should be a something like [this](./ai/agi-notes/)!

Notice: this prompt might not work in the first attempt depending on the what AI chat you're in. Just hit `regenerate` to give the LLM the time to "think" if that the case

## Steps to Contribute

1. Fork it!
2. Branch off of `master`: `git checkout master`
3. Create your note branch: `git checkout -b my-new-note`
4. Make changes
5. Commit your changes: `git commit -m 'Add topic notes/Add new changes, in case you add new change to an existant note'`
6. Push to the branch: `git push origin my-new-note`
7. Submit a pull request. Make sure it is based off of the `master` branch when submitting! :D
 
`Notice`: You might come across to the following commit message patterns `topic-notes: Hello World!`. This because when I started the project, I was the only contributor, so instead of creating a new branch + PR, for each note, I just made my life easier working directly on master branch  (not a good practice, and I apologize to [Linus Torvalds](github.com/Torvalds) If He reads this one day though :D)


## Note Taking resources

Here you'll find some best practices, methods and techniques for note taking:

- [Note-taking](https://en.wikipedia.org/wiki/Note-taking)
- [Zettelkasten - The art of taking notes](https://en.wikipedia.org/wiki/Zettelkasten)
- [A Beginner’s Guide to the Zettelkasten Method](https://zenkit.com/en/blog/a-beginners-guide-to-the-zettelkasten-method/)
- [Note Taking Structures - Moving Beyond Bullets and Dashes By T. Milligan](https://www.dunwoody.edu/pdfs/Elftmann-Note-Taking-Structures.pdf)
- [Common Note-taking Methods | University of Tennessee](https://www.utc.edu/enrollment-management-and-student-affairs/center-for-academic-support-and-advisement/tips-for-academic-success/note-taking)

The Feynman Technique
- [Learning From the Feynman Technique](https://evernote.com/blog/learning-from-the-feynman-technique/)



