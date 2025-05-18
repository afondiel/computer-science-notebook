# Masakhane Hello World!

## Overview

To use the Masakhane models to create a simple "Hello World" program, you can follow these steps:

- Install the Hugging Face Transformers library by running the command `pip install transformers` in your terminal or command line.
- Choose a model from the Masakhane Hugging Face models collection¹ that supports your language and task. For example, you can use the `masakhane/afri-mbart50` model for machine translation or the `masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0` model for named entity recognition.
- Load the model and the tokenizer using the `AutoModelForSeq2SeqLM` and `AutoTokenizer` classes from the Transformers library. For example, you can use the following code to load the `masakhane/afri-mbart50` model and the tokenizer:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and the tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("masakhane/afri-mbart50")
tokenizer = AutoTokenizer.from_pretrained("masakhane/afri-mbart50")
```

- Encode your input text using the tokenizer and generate the output text using the model. You can specify the source and target languages using the `src_lang` and `tgt_lang` arguments. For example, you can use the following code to translate the text "Hello, world!" from English to Swahili:

```python
# Encode the input text
input_ids = tokenizer.encode("Hello, world!", return_tensors="pt", src_lang="en_XX")

# Generate the output text
output_ids = model.generate(input_ids, max_length=20, tgt_lang="sw_KE")

# Decode the output text
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
```

- Run the code by entering `python masakhane-test.py` in your terminal or command line, where `masakhane-test.py` is the name of your Python file. You should get an output text that resembles the following:

```
Habari, dunia!
```

## References

- [masakhane-io/masakhane-mt: Machine Translation for Africa - GitHub](https://github.com/masakhane-io/masakhane-mt).
- [GitHub - dsfsi/masakhane-web: Masakhane Web is a translation web .... ](https://github.com/dsfsi/masakhane-web).
- [Masakhane](https://www.masakhane.io/).
- [Masakhane - create-pull-request-github](https://opensource.com/article/19/7/create-pull-request-github).

