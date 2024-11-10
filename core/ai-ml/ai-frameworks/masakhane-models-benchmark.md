# Masakhane Hugging Face models Benchmarks

## Overview

Masakhane Hugging Face models are a collection of models for natural language processing (NLP) tasks in African languages, created by the Masakhane NLP community¹. 

Masakhane is a grassroots organization whose mission is to strengthen and spur NLP research in African languages, for Africans, by Africans¹. 

## Masakhane Hugging Face models

- **masakhaner**²: A named entity recognition (NER) dataset consisting of PER, ORG, LOC, and DATE entities annotated by Masakhane for ten African languages: Amharic, Hausa, Igbo, Kinyarwanda, Luganda, Luo, Nigerian-Pidgin, Swahili, Wolof, and Yoruba.
- **masakhaner2**³: An updated version of masakhaner, with more languages, more data, and more annotations. It covers 21 African languages and 6 entity types: PER, ORG, LOC, DATE, TIME, and MISC.
- **masakhanews**⁴: A machine translation dataset for low-resource African languages, consisting of news articles from the BBC and Africanews, covering topics such as politics, sports, health, and entertainment. It includes 13 languages: Amharic, Akan, Hausa, Igbo, Kinyarwanda, Lingala, Luganda, Luo, Ndebele, Nigerian-Pidgin, Swahili, Wolof, and Yoruba.
- **masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0**⁵: A NER model for 21 African languages, fine-tuned on an aggregation of masakhaner and masakhaner2 datasets. It is based on the Davlan/afro-xlmr-large model, which is a multilingual model trained on 100 languages, including several African languages.
- **masakhane/afriqa**⁶: A question answering dataset for African languages, consisting of questions and answers extracted from Wikipedia articles. It includes 8 languages: Amharic, Hausa, Igbo, Swahili, Wolof, Yoruba, Somali, and Oromo.
- **masakhane/afri-mbart50**: A machine translation model for 50 African languages, fine-tuned on the JW300 dataset. It is based on the facebook/mbart-large-50 model, which is a multilingual model trained on 50 languages, including several African languages.
- **masakhane/afri-mt5-base**: A text-to-text generation model for African languages, fine-tuned on the JW300 dataset. It is based on the google/mt5-base model, which is a multilingual model trained on 101 languages, including several African languages.

## References

- [masakhane.io](https://www.masakhane.io/)
- [masakhaner · Datasets at Hugging Face](https://huggingface.co/datasets/masakhaner).
- [masakhane (Masakhane NLP) - Hugging Face](https://huggingface.co/masakhane).
- [masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0 · Hugging Face](https://huggingface.co/masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0).
- [masakhane/masakhaner2 · Datasets at Hugging Face](https://huggingface.co/datasets/masakhane/masakhaner2).
- [MasakhaNER: Named Entity Recognition for African Languages - 2021](https://arxiv.org/abs/2103.11811)

