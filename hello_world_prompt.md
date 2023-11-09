# Hello World Prompt

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
