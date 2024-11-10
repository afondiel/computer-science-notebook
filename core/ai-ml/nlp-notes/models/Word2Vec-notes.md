# Word2Vec - Notes

## Table of Contents (ToC)
  - [Introduction](#introduction)
    - [What's Word2Vec?](#whats-word2vec)
    - [Key Concepts and Terminology](#key-concepts-and-terminology)
    - [Applications](#applications)
  - [Fundamentals](#fundamentals)
    - [Word2Vec Architecture Pipeline](#word2vec-architecture-pipeline)
    - [How Word2Vec Works](#how-word2vec-works)
    - [Types of Word2Vec Models](#types-of-word2vec-models)
    - [Some Hands-On Examples](#some-hands-on-examples)
  - [Tools \& Frameworks](#tools--frameworks)
  - [Hello World!](#hello-world)
  - [Lab: Zero to Hero Projects](#lab-zero-to-hero-projects)
  - [References](#references)


## Introduction
Word2Vec is a popular word embedding technique that represents words in a continuous vector space, capturing semantic similarities.

### What's Word2Vec?
- Introduced by Google in 2013 for creating word embeddings.
- Converts words into dense vectors based on context within large corpora.
- Captures semantic relationships between words in a continuous vector space.

### Key Concepts and Terminology
- **Word Embedding**: Representation of words in a dense vector space where similar words have similar vectors.
- **Skip-gram Model**: Predicts surrounding words given a target word.
- **Continuous Bag of Words (CBOW)**: Predicts a target word based on its surrounding words.
- **Context Window**: The number of words around the target word used for prediction.

### Applications
- **Text Similarity**: Measuring similarity between words, sentences, or documents.
- **Sentiment Analysis**: Enhancing feature representations for NLP models.
- **Machine Translation**: Improving translation quality by understanding word relationships.
- **Named Entity Recognition (NER)**: Identifying and classifying proper nouns in text.

## Fundamentals

### Word2Vec Architecture Pipeline
- **Text Preprocessing**: Tokenization, removal of stopwords, and lowercasing.
- **Training Objective**: Choose between Skip-gram or CBOW to train the model.
- **Context Window**: Define the size of the context window.
- **Output**: Generate word embeddings where each word is represented by a dense vector.

### How Word2Vec Works
- **Skip-gram Model**: Maximizes the probability of predicting context words given a target word.
- **CBOW Model**: Maximizes the probability of predicting a target word given surrounding context words.
- **Training Process**: Uses shallow neural networks to generate word vectors based on context.
- **Word Vector Space**: Words with similar meanings end up with similar vector representations.

### Types of Word2Vec Models
- **Skip-gram**: Focuses on predicting context words, good for small datasets.
- **CBOW**: Predicts target words based on context, faster and suitable for large datasets.
- **GloVe (Global Vectors for Word Representation)**: An alternative to Word2Vec, considers word co-occurrence statistics.

### Some Hands-On Examples
- **Word Similarity**: Identifying similar words using cosine similarity between word vectors.
- **Analogy Reasoning**: Performing analogy tasks like "king - man + woman = queen."
- **Clustering Words**: Grouping similar words together using K-means clustering on word vectors.
- **Text Classification**: Enhancing feature extraction for classification tasks using word embeddings.

## Tools & Frameworks
- **Gensim**: Python library for training and using Word2Vec models.
- **TensorFlow & PyTorch**: Frameworks that support custom implementations of Word2Vec.
- **NLTK**: Provides utilities for text preprocessing before Word2Vec training.
- **FastText**: An extension of Word2Vec by Facebook for learning word vectors.

## Hello World!
```python
from gensim.models import Word2Vec

# Example corpus
sentences = [["this", "is", "a", "sample", "sentence"],
             ["word2vec", "is", "great", "for", "text", "similarity"]]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

# Example usage: Finding similar words
similar_words = model.wv.most_similar("sample")
print(similar_words)
```

## Lab: Zero to Hero Projects
- **Word Similarity Tool**: Build an app to find similar words using Word2Vec embeddings.
- **Text Classification with Embeddings**: Enhance text classification using Word2Vec embeddings.
- **Document Clustering**: Group documents by topic using word embeddings and clustering algorithms.
- **Word Embedding Visualization**: Visualize word vectors using PCA or t-SNE.

## References
- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space."
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality."
- Pennington, J., Socher, R., & Manning, C. D. (2014). "GloVe: Global Vectors for Word Representation."
- Rong, X. (2014). "Word2Vec Parameter Learning Explained."



