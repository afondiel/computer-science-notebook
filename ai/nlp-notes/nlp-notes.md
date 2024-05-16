# Natural Language Processing (NLP) - Notes

## Overview

NLP is a computer science, artificial Intelligence subsets which deals with Human Language.


## Application

- Sentimental Analysis: (ex: fb emojis)
- Chatbot (costumer assistance)
- Speech Recognition (voice assistance, like windows cortana)
- Machine Translation (google translate)
- Spell Checking
- Information Extraction
- Keyword Searching
- Advertisement 

## Components of NLP

Natural Language Understanding (NLU)
- Mapping input to useful representations
- Analyzing different aspects of the language
- Ambiguity lexical
- Ambiguitysyntactic 
- Ambiguity referential
 
Natural Language Generation (NLG)
- Text planning
- Sentence planning
- Text realization

TEXT MINING: analysation of information from natural language text.


## NLP Pipeline

![alt text](https://miro.medium.com/v2/resize:fit:828/format:webp/1*CbzCcP3XFtYVJmWowZLugQ.png)

Src: [Basic Steps In Natural Language Processing Pipeline](https://monicamundada5.medium.com/basic-steps-in-natural-language-processing-pipeline-763cd299dd99)

## NLP Techniques

1. **SYNTACTIC ANALYSIS** : understanding the grammar of the text
   - **Segmentation** : split document into contituent units (one or more sentences)
   - **Tokenization** : split a phrase into a small part od token
   - **Stemming** : normalize words into its base form or root form
   - **Lemmatization**: morphological analyse of the phrase (based on dictionary : meaning and synonym)
   - **POS tagging** : classify a part of the speech : verb, adj, noun 
   - **Named Entity Recognition (NER)**: classify a group of word in a group : movie, monetary value, organizatio, location, quantities, person
   - **Chunking**: picking individual pieces of informations and grouping them into bigger Pieces
   - **Parsing(tree)**: grammatical analysis of the sentence

2. **SEMANTIC ANALYSIS** : understanding the literal meaning of the text.
   - Text Correction
   - Text generation
   - Machine Translation
   - Word sense disambiguation
   - Vector/Embeddings (high-dimensional vectors)

3. **PRAGMATIC ANALYSIS**: understanding of what the text is trying to achieve

## Content Generation

- Next Sentence Prediction (NSP) : Check if the sentence B, follows the sentence A.

Tools:
- [Deep Learning](../deep-learning-notes/)
- [GenAI and LLM](../generative-ai-notes/llm-notes/LLM-notes.md)
- [Multimodal AI](../Multimodal-AI/)
- [RAG (Retrieval-Augmented Generation)](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)
- Embeddings (high-dimensional vectors)


## Machine Translation

![](./docs/mt-group.png)

Src: [Study tonight](https://www.studytonight.com/post/different-machine-translation-models-in-nlp)

## Large Language Models (LLMs) 

|Feature|NLP|LLM|
|--|--|--|
|Focus|	Broad range of techniques for specific tasks|Specific type of NLP model, broad capabilities|
|Strength|Accuracy in specific tasks|Versatility, open ended tasks|
|Underlying Techniques|Rule-based, Machine Learning, Deep Learning|Deep Learning (Transformers)|
|Applications|	Chatbots (specific functions), sentiment analysis, named entity recognition|Creative text generation, summarization, open ended chatbots|
||||

- More about LLMs [here](../generative-ai-notes/llm-notes/LLM-notes.md).


## Tools & Frameworks

- [NLTK](https://www.nltk.org/)
- [Spacy](https://spacy.io/)
- Facebook AI XLM/mBERT
- PyTorch  
- TensorFlow
- Keras
- Chainer
- [Gensim](https://radimrehurek.com/gensim/index.html)
- [Fasttext](https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html#sphx-glr-auto-examples-tutorials-run-fasttext-py)
- [Word2Vec Model](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py)
- [OpenAI Embeddings tools](https://platform.openai.com/docs/guides/embeddings)
- [GloVe - stanford](https://nlp.stanford.edu/projects/glove/)
- [LangChain](https://python.langchain.com/docs/integrations/toolkits/openapi_nla)


## References

Wikipedia

  - Natural Language Processing: https://en.wikipedia.org/wiki/Natural_language_processing
  - MT: https://en.wikipedia.org/wiki/Machine_translation
  - Neural MT: https://en.wikipedia.org/wiki/Neural_machine_translation
  - NLP: https://en.wikipedia.org/wiki/Natural_language_processing
  - semantic wiki: https://en.wikipedia.org/wiki/Semantics

Lecture and Online Courses:
- ibm: 
  - [What is NLP (Natural Language Processing)?](https://www.youtube.com/watch?v=fLvJ8VdHLA0&t=0s)
  - ibm nlp vs nlu vs nlg: https://www.youtube.com/watch?v=1I6bQ12VxV0&t=12s
  - Machine Translation - ibm lstm: https://www.youtube.com/watch?v=b61DPVFX03I
- npl vs nlu: https://www.kdnuggets.com/2019/07/nlp-vs-nlu-understanding-language-processing.html
- Google Research: https://research.google/research-areas/natural-language-processing/
- [NLP for Developers - Full course Rasa](https://www.youtube.com/watch?v=hJ1hzEJE16c&list=PL75e0qA87dlFJiNMeKltWImhQxfFwaxvv&index=1)

- Stanford Univ School of Engineering:
  - [Lecture Collection | Natural Language Processing with Deep Learning](https://www.youtube.com/watch?v=OQQ-W_63UgQ&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6) 
- [OpenAI Embeddings and Vector Databases Crash Course](https://www.youtube.com/watch?v=ySus5ZS0b94)

Frameworks: 
- https://odsc.medium.com/10-notable-frameworks-for-nlp-ce8c4196bfd6
- NLP w/ python & nltk: 
  - https://www.youtube.com/watch?v=U8m5ug9Q54M  
  - https://www.youtube.com/watch?v=05ONoGfmKvA&t=0s

- Nlp with spy: 
  - https://www.youtube.com/watch?v=dIUTsFT2MeQ
  - Spacy: https://www.youtube.com/watch?v=dIUTsFT2MeQ&t=1025s
- [Transformer Neural Networks - EXPLAINED! (Attention is all you need) - CodeEmporium](https://www.youtube.com/watch?v=TQQlZhbC5ps)
- Transformer: https://devopedia.org/transformer-neural-network-architecture


MT: 
- https://www.studytonight.com/post/different-machine-translation-models-in-nlp

Word Embedding and Vector Space Model
- [Word embedding](https://en.wikipedia.org/wiki/Word_embedding)
- [Vector Space Models](https://towardsdatascience.com/vector-space-models-48b42a15d86d)

- [Word Embedding and Vector Space Models](https://medium.com/analytics-vidhya/word-embedding-and-vector-space-models-11c9b76f58e)
- [Exploring Qdrant: A Guide to Vector Databases](https://medium.com/@bilalhanif848/exploring-qdrant-a-guide-to-vector-databases-68dc6a405be4)
- [Getting Started With Embeddings - HuggingFace](https://huggingface.co/blog/getting-started-with-embeddings)

NLP vs DL vs GenAI vs LLM Tools:
- [Unleashing the Power of Langchain: Advancing NLP with Chained Language Models](https://anote-ai.medium.com/introducing-langchain-revolutionizing-nlp-with-advanced-language-modeling-ef4f8f2fc18a)
- [NLP vs LLM: A Comprehensive Guide to Understanding Key Differences](https://medium.com/@vaniukov.s/nlp-vs-llm-a-comprehensive-guide-to-understanding-key-differences-0358f6571910)
- [RAG vs semantic search, embedding, vector... ](https://www.youtube.com/watch?v=orLGv2LgWDE)
- [RAG vs Finetuning — Which Is the Best Tool to Boost Your LLM Application?](https://towardsdatascience.com/rag-vs-finetuning-which-is-the-best-tool-to-boost-your-llm-application-94654b1eaba7)

Projects:
- [Self-Driving Taxi Chatbot with Pytorch](https://github.com/diesimo-ai/self-driving-taxi-chatbot)

Research papers: 
- Attention is all you need (paper): https://research.google/pubs/pub46201/
- analyticsindiamag : https://analyticsindiamag.com/10-must-read-technical-papers-on-nlp-for-2020/
- slator : https://slator.com/here-are-the-best-natural-language-processing-papers-from-acl-2022/
- paperdigest: https://www.paperdigest.org/category/nlp/
- paper with code: https://paperswithcode.com/task/language-modelling



