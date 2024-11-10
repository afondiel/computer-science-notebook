# Large Language Models (LLMs) - Notes

## Overview

A large language model (LLM) is a language model consisting of a neural network with many parameters (typically billions of weights or more), trained on large quantities of unlabeled text using self-supervised learning or semi-supervised learning + NLP techniques.

## Applications

- Chatbots (ChatGPT, Bard ...)
- Search Engines (Being AI, Google ...)
- Multi-language Translator
- Image & videos generation (AI art...) 

## Tools & Frameworks

- LangChain
- Cohere
- GPT3.5 / GPT-4 (OpenAI)
- LLaMA (Meta)
- ...

## Hello World!

LLMs `Hello World` example using LangChain bash process to perform simple filesystem commands.

```sh
from langchain_experimental.llm_bash.base import LLMBashChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

text = "Please write a bash script that prints 'Hello World' to the console."

bash_chain = LLMBashChain.from_llm(llm, verbose=True)

bash_chain.run(text)
```

Prompt sequences:

```
    
    
    > Entering new LLMBashChain chain...
    Please write a bash script that prints 'Hello World' to the console.
    
    ```bash
    echo "Hello World"
    ```
    Code: ['echo "Hello World"']
    Answer: Hello World
    
    > Finished chain.





    'Hello World\n'
```

For more details, check the entire notebook [here](https://python.langchain.com/docs/use_cases/more/code_writing/llm_bash).


## Vision LLMS vs Vision Language Models

- OpenAI DALLE 2
- Flamingo (DeepMind)
- GPT-3 (OpenAI)
- LaMDA (Google)
- PaLM (Google)
- Llama2 (Meta AI)

- Src: [Most Powerful 7 Language (LLM) and Vision Language Models (VLM) Transforming AI in 2023](https://www.topbots.com/top-language-models-transforming-ai-in-2023/)

## References

- [Large Language Model - Wikipedia](https://en.wikipedia.org/wiki/Large_language_model)
- [What developers need to know about generative AI](https://github.blog/2023-04-07-what-developers-need-to-know-about-generative-ai/)

Lectures & Tutorials:

- [[1hr Talk] Intro to Large Language Models - karpathy.ai](https://www.youtube.com/watch?v=zjkBMFhNj_g)
- [State of GPT - MS Build w/ Andrej Karpathy](https://www.youtube.com/watch?v=bZQun8Y4L2A)

