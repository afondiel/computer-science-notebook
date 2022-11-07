# Natural Language Processing (NLP) - Notes

NLP is a Computer science, artificial Intelligence subsets which deals with Human Language

- Subset : 
Text mining : analysation of information from natural language text.

## Application

* Sentimental Analysis : like -> fb emojis
* Chatbot : costumer assistance
* Speech Recognition : voice assistance, like windows cortana
* Machine Translation : google translate
* Spell Checking
* Information Extraction
* Keyword Searching
* Advertisement 

## Components of NLP :
### Natural Language Understanding (NLU)
  - Mapping input to useful representations
  - Analyzing different aspects of the language
  - ambiguity lexical
  - ambiguitysyntactic 
  - ambiguity referential
 
### Natural Language Generation (NLG)
  - text planning
  - sentence planning
  - text Realization

## NLP Techniques

1. SYNTACTIC ANALYSIS : understanding the grammar of the text
  - **Segmentation** : split document into contituent units (one or more sentences)
  - **Tokenization** : split a phrase into a small part od token
  - **Stemming** : normalize words into its base form or root form
  - **Lemmatization**: morphological analyse of the phrase (based on dictionary : meaning and synonym)
  - **POS tagging** : classify a part of the speech : verb, adj, noun 
  - **Chunking**: picking individual pieces of informations and grouping them into bigger Pieces
  - <b>Parsing (tree):</b> grammatical analysis of the sentence

2. SEMANTIC ANALYSIS : understanding the literal meaning of the text.

  - **Named Entity recognition (NER)**: classify a group of word in a group : movie, monetary value, organizatio, location, quantities, person
  - Text generation
  - Machine translation
  - Word sense disambiguation

3. PRAGMATIC ANALYSIS : understanding of what the text is trying to achieve


## Tools / frameworks

- PyTorch  
- spacy  :  https://spacy.io/
- Facebook AI XLM/mBERT
- TensorFlow
- Keras
- Chainer
- Gensim 
- nltk : https://www.nltk.org/


# References : 

- Wikipedia : https://en.wikipedia.org/wiki/Natural_language_processing
- ibm : https://www.youtube.com/watch?v=fLvJ8VdHLA0&t=42s
- ibm nlp vs nlu vs nlg : https://www.youtube.com/watch?v=1I6bQ12VxV0&t=12s
- Machine Translation - ibm lstm : https://www.youtube.com/watch?v=b61DPVFX03I
- npl vs nlu : https://www.kdnuggets.com/2019/07/nlp-vs-nlu-understanding-language-processing.html
- NLTK : https://www.youtube.com/watch?v=05ONoGfmKvA&t=0s
- Spacy : https://www.youtube.com/watch?v=dIUTsFT2MeQ&t=1025s
- Google Research : https://research.google/research-areas/natural-language-processing/
- Transformer : https://devopedia.org/transformer-neural-network-architecture
- Attention is all you need (paper)  : https://research.google/pubs/pub46201/
- Rasa - full - tuto : https://www.youtube.com/watch?v=hJ1hzEJE16c&list=PL75e0qA87dlFJiNMeKltWImhQxfFwaxvv&index=1
frameworks : https://odsc.medium.com/10-notable-frameworks-for-nlp-ce8c4196bfd6
- NLP w/ python & nltk : 
https://www.youtube.com/watch?v=U8m5ug9Q54M
- nlp with spy : 
https://www.youtube.com/watch?v=dIUTsFT2MeQ
- YT course Stanford :
https://www.youtube.com/watch?v=OQQ-W_63UgQ&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6 
https://www.youtube.com/watch?v=OQQ-W_63UgQ

Research papers : 
- analyticsindiamag :
https://analyticsindiamag.com/10-must-read-technical-papers-on-nlp-for-2020/
- slator :
https://slator.com/here-are-the-best-natural-language-processing-papers-from-acl-2022/
- paperdigest:
https://www.paperdigest.org/category/nlp/
- paper with code:
https://paperswithcode.com/task/language-modelling
