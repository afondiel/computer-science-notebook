# Hello World Prompt

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
