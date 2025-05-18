# LangChain Hello World

## Table of Contents

- [Overview](#overview)
- [Applications](#applications)
- [Tools & Frameworks](#tools-frameworks)
- [Hello World!](#hello-world)
- [References](#references)

## Overview

LangChain is a powerful framework that aims to help developers build end-to-end applications using large language models (LLMs). It provides a set of tools, components, and interfaces that simplify the process of creating applications powered by LLMs and chat models.

## Applications

- LangChain can be used to build context-aware, reasoning applications with flexible abstractions and AI-first toolkit¹.
- LangChain can also be used to create and deploy LLM apps with confidence using LangSmith, an all-in-one developer platform for every step of the application lifecycle².
- LangChain can leverage new cognitive architectures and battle-tested orchestration to improve the performance and scalability of LLM apps¹².
- LangChain can integrate with various domains and languages, and allow users to customize LLMs with their own data¹⁴.
- LangChain can establish best practices and standards for LLM app development and evaluation²⁵.

## Tools & Frameworks

- LangChain offers a standard interface for chains, which are sequences of LLMs and chat models that perform different tasks on the input and output data¹⁴.
- LangChain provides lots of integrations with other tools, such as Hugging Face, PyTorch, TensorFlow, and GPT-3¹⁴⁵.
- LangChain supports end-to-end chains for common applications, such as chatbots, Q&A, summarization, copilots, workflow automation, document analysis, and custom search¹².
- LangChain also provides a web-based playground where users can try out different models and use cases interactively¹².

## Hello World!

Here is a code snippet that shows how to use the LangChain Python SDK to create a simple chain that generates a summary of a text document:

```python
import langchain

# Initialize a chain object
chain = langchain.Chain()

# Add a LLM component that extracts the main points from the document
chain.add_component(
    name="extractor",
    model="gpt-3",
    prompt="Given the following document, extract the main points in bullet points:\n{input}\n\n-",
    max_tokens=100,
    stop="-"
)

# Add a LLM component that generates a summary from the main points
chain.add_component(
    name="summarizer",
    model="gpt-3",
    prompt="Given the following main points, generate a summary in one sentence:\n{input}\n\nSummary:",
    max_tokens=50,
    stop="."
)

# Run the chain on a sample document
document = "LangChain is a framework designed to simplify the creation of applications using large language models (LLMs). It provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications."
output = chain.run(document)

# Print the output
print(output)
```

## References

- [LangChain](https://www.langchain.com/)
- [LangSmith - langchain.com](https://www.langchain.com/langsmith).
- [LangChain - Wikipedia](https://en.wikipedia.org/wiki/LangChain).
- [Introduction to Langchain - GeeksforGeeks](https://www.geeksforgeeks.org/introduction-to-langchain/).
- [LangChain 完整指南：使用大语言模型构建强大的应用程序 - 知乎.](https://zhuanlan.zhihu.com/p/620529542).
- [en.wikipedia.org](https://en.wikipedia.org/wiki/LangChain).

