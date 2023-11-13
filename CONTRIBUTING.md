# Contributor Guidelines

**Contribute to Excellence**

- Are you passionate about sharing your expertise and making learning accessible to all? Join us in our mission to empower software engineers worldwide. Whether you're a seasoned pro or just starting your journey, your contributions are invaluable. Together, we can create a resource that will benefit countless aspiring and experienced developers.

**How to Contribute**

- Getting started is easy! Dive into our open issues, propose new topics, enhance existing content, or collaborate with our community of like-minded contributors. Your contributions will not only help others but also establish your presence in the ever-expanding world of computer science education.

Join us in shaping the future of software engineering education and be a part of something truly remarkable. Start today and become an integral part of this exciting journey!

You can contribute in many ways :

## Notes Structure

**Note Template**

Two ways you can generate/create your first note:

**Using [hello_world.py](hello_world.py) tool**

By running the command below:

```python
python hello_world.py topic-name
```

**Customize your note nanually**

- Create a new folder with the <topic name>
- Inside the root folder <topic-name>, create: 
  - 2 other folders: `docs`, `lab`
  - 1 markdown file entitled: `topic-notes.md`
  - 1 `readme.md`for notes description

Your final note folder should have the following structure:

```
E:.
│   topic-notes.md
│   readme.md
│
├───docs
└───lab
```

### `topic-notes.md` Format

```
# Title - Notes
## Table of Contents (ToC)
## Overview
## Applications
## Tools & FrameWorks
## Hello World!
## References
```

### ChatGPT/Bard Prompt for Content Generation

```
I'm going to train you write a short-form summary.

The summary has the following format: 
"""
# {Topic} - Notes
## Table of Contents (ToC)
## Overview
## Applications
## Tools & Frameworks
## Hello World!
## References
"""
Here are the rules: 
- replace `{Topic}` by the name of the topic I will give you 
- The `ToC` has to be link-based to each heading
- The `## Overview` heading has to be one-line sentence
- Organize the headings in a list of 5-6 succinct bullets except for `Overview`  
- The `## Hello World! ` has to be a code snippet
- Do not use hashtags and emojis. Ever.
- output the summary in markdown format including the `## References`

I am going to give you a topic name and you will summarize the topic.

Do you understand?
```

Transition: 

```
Topic={Topic}
```

Magic !

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

## Your Contribution 

For this example I suppose you already performed the followng tasks
```
- git clone https://github.com/afondiel/research-notes.git
- git branch -b <branch-name> 
``` 
After made some changes  and finally have your note ready. Let's say your first note is ready and it's called: `agi-notes`.   

**First Commit!**

1. Add your current files/changes & cd

```
E:\research-notes\agi-notes> git add .
```

3. Commit them

```
E:\research-notes\agi-notes> git commit -m"agi-notes: Hello World!"
``` 

4. Push to remote

```
E:\research-notes\agi-notes> git push origin <branch-name> 
``` 

5. Open PR on github

It's done! 

PS : Your `first commit message` doesn't need to be boring like "initial commit" or "first commit", instead make it look funny:  git commit -m"agi-notes: Hello World!"

## References

Note Taking resources

- [Note-taking](https://en.wikipedia.org/wiki/Note-taking)
- [Zettelkasten - The art of taking notes](https://en.wikipedia.org/wiki/Zettelkasten)
- [A Beginner’s Guide to the Zettelkasten Method](https://zenkit.com/en/blog/a-beginners-guide-to-the-zettelkasten-method/)
- [Note Taking Structures - Moving Beyond Bullets and Dashes By T. Milligan](https://www.dunwoody.edu/pdfs/Elftmann-Note-Taking-Structures.pdf)
- [Common Note-taking Methods | University of Tennessee](https://www.utc.edu/enrollment-management-and-student-affairs/center-for-academic-support-and-advisement/tips-for-academic-success/note-taking)

The Feynman Technique

- [Learning From the Feynman Technique](https://evernote.com/blog/learning-from-the-feynman-technique/)



