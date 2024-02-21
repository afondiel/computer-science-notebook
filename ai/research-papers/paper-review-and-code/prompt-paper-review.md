# Prompt for Paper Review and Summarization  

## Overview

Prompt for paper review and summarization. The summary can be generated at once or interactively , i.e section by section.

## Setup Config

### ChatGPT

Config:

`gpt-3.5-turbo, temperature : default:`

- Notice ChatGPT gpt-3 model does not access external pdf files or url, but it may provide the results based on general understanding of the topic

- gpt-4 may not encounter this issue(to be tested)

### MS Copilot

- No additional config need
 
### Gemini (former Bard)

Config: `default`

- No issues with bard


## Prompts: 

`prompt1: summarize import insights`

```
Act as a machine learning expert. I want you to give a short summary of the transformers model from the paper "Attention is all you need ", focusing on the methodology and findings. I need you give the summarization in the following order in a markdown format: 
- Introduction
- architecture pipeline
- How transformers model works ? 
- Machine translation
- Pros & cons
- Conclusion
- Authors names
```

`prompt2: summarize import insights`

```
Act as an academician. I will provide you a sequence of scientific papers and you will be responsible for reviewing them and providing a concise summary in APA style. The first paper is entitled "<paper title, authors/organization, year> / <url>". Organize the contents into the following markdown format: 
- Introduction
- Problem and Methodologies
- Architecture pipeline
- Findings 
- Conclusion
- Authors' names and organizations
```

`prompt2.1: summarize import insights`

```
Act as an academician. I will provide you a sequence of reseach papers and you will do a review and provide a summary(APA style). 

The summary has the following format:
'''
- Abstract
- Introduction
- Problem and Solution (Methodologies)
- System Architecture pipeline
- Findings
- Conclusion
- Authors' and organizations
'''

Here are some rules:
- Make it concise and succinct to read
- Do not use hashtag or emojis. Ever.

I am going to give the name of the research paper and you will write a summary and output in markdown format.

Do you understand?
```

`prompt3: Extract Relevant information/insights`

```
Your task is to extract relevant information from the research paper: "Levels of AGI: Operationalizing Progress on the Path to AGI"  from `google deepmind`

Extract the information relevant to `robot` and `AGI embodiment`
``` 


`p4 - last update`

```
I am going to train you to write short-form paper summary (in APA style).

The summary has the following markdown structure:
"""
- Abstract
- Introduction
- Problem and Solution (Methodology)
- System Architecture Pipeline
- Findings
- Conclusion
- Authors and organizations
"""

Here are the rules:
- The summary structure is `H2` level
- Create a title for the summary
- Use 3-5 succinct bullet lists for each section of the summary.
- generate shorter and concise content
- Do not use hashtags or emojis. Ever.

I am going to give you the paper name and you are going to write a short-form piece based on structure given at the start.

Do you understand?
```
`p4-transition`: the paper= {title} : {link}


