# Prompt Template for Paper Review and Summarization    

## Prompts: 

`prompt1: `

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

`prompt2:`

```
Act as an academician. I will provide you a sequence of scientific papers and you will be responsible for reviewing them and providing a concise summary in APA style. The first paper is entitled "<paper title, authors/organization, year> / <url>". Organize the contents into the following markdown format: 
- Introduction
- Problem and Methodologies
- Architecture pipeline
- Findings 
- Conclusion
- Authors' names and organizations
```

## Setup Config

### ChatGPT

Config:

`gpt-3.5-turbo, temperature : default:`

- Notice ChatGPT gpt-3 model does not access external pdf files or url, but it may provide the results based on general understanding of the topic

- gpt-4 may not encounter this issue(to be tested)
 
### Bard

Config: `default`

- No issues with bard


## Example using transformer paper:


## Introduction
The paper "Attention is all you need" introduces the Transformers model, a novel architecture for sequence-to-sequence tasks in natural language processing. This model eliminates the need for recurrent or convolutional layers traditionally used in sequence models and replaces them with attention mechanisms.

## Architecture Pipeline
The architecture of the Transformers model consists of an encoder-decoder framework. The encoder takes an input sequence and generates a sequence of continuous representations, while the decoder takes these representations and generates the output sequence. Both the encoder and decoder are composed of multiple identical layers, and each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.

## How Transformers Model Works?
The key concept in the Transformers model is self-attention. Self-attention allows the model to focus on different parts of the input sequence while computing the representation for each position. It assigns weights to each input token based on its relevance to other tokens in the sequence, capturing dependencies and relationships effectively. Self-attention is performed multiple times in parallel, with each instance attending to different parts of the sequence. This parallelization improves the model's ability to capture long-range dependencies.

Additionally, the Transformers model incorporates positional encoding to consider the order of the tokens in the input sequence. Positional encoding is added to the input embeddings and provides the model with information about the token's position in the sequence.

## Machine Translation
One of the main applications explored in the paper is machine translation. The Transformers model achieves state-of-the-art performance on the WMT 2014 English-German and English-French translation tasks, surpassing previous models based on recurrent or convolutional architectures. The model's ability to capture long-range dependencies and the parallelization of self-attention contribute to its success in machine translation.

## Pros & Cons
The Transformers model brings several advantages compared to traditional sequence models. It avoids the computational constraints of recurrent networks by parallelizing the attention mechanism. This parallelization enables more efficient training and inference, making the Transformers model faster. Additionally, the model achieves superior performance on various natural language processing tasks, including machine translation.

However, the Transformers model requires a significant amount of computational resources due to the self-attention mechanism's quadratic complexity with respect to the input sequence length. This scalability issue makes it challenging to process very long sequences. Furthermore, the self-attention mechanism may struggle to capture local dependencies as effectively as recurrent models in certain cases.

## Conclusion
The paper "Attention is all you need" presents the Transformers model, which revolutionizes sequence-to-sequence tasks in natural language processing. By leveraging self-attention mechanisms and parallelization, the model achieves state-of-the-art results in machine translation and other NLP tasks. Although the Transformers model has some drawbacks, its performance and efficiency advancements make it a significant breakthrough in the field of deep learning.

## Authors' Names
The authors of the paper "Attention is all you need" are Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; Uszkoreit, Jakob; Jones, Llion; Gomez, Aidan N.; Kaiser, Łukasz; and Polosukhin, Illia.

