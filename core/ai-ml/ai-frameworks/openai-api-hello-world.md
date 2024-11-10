# OpenAI API Hello World!

## Getting started 

To use the OpenAI API to create a simple "Hello World" program, you can follow these steps:

- Install the OpenAI Python library by running the command `pip install openai` in your terminal or command line.
- Get your secret API key from your [API Keys page](^1^) and save it somewhere safe. Do not share it with anyone or expose it in any client-side code.
- Set up your API key in your Python code by importing the `os` module and using the `openai.api_key = os.getenv("OPENAI_API_KEY")` statement. Make sure to replace `"OPENAI_API_KEY"` with the name of the environment variable that stores your API key.
- Send a request to the OpenAI API using the `openai.Completion.create` method. You can specify the engine, the prompt, and other parameters to customize the request. For example, you can use the following code to generate a completion for the prompt "Hello, world!" using the `gpt-3.5-turbo` engine and limiting the output to 10 tokens:

```python
import openai
import os

# Set up your API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Send a request to the OpenAI API
response = openai.Completion.create(
    engine="gpt-3.5-turbo",
    prompt="Hello, world!",
    max_tokens=10
)

# Print the response text
print(response.choices[0].text)
```

- Run the code by entering `python openai-test.py` in your terminal or command line, where `openai-test.py` is the name of your Python file. You should get a response text that resembles the following:

```
Hello, world!
How are you today?
```


## References

- [API Reference - OpenAI API](https://platform.openai.com/docs/api-reference).
- [OpenAI Platform](https://platform.openai.com/docs/quickstart).
- [Need help with hello world - API - OpenAI Developer Forum](https://community.openai.com/t/need-help-with-hello-world/125707).
- [Hello World In ChatGPT - C# Corner](https://www.c-sharpcorner.com/article/hello-world-in-chatgpt/).
- [api.openai.com/v1/models](https://api.openai.com/v1/models).
- [api.openai.com/v1/chat/completions](https://api.openai.com/v1/chat/completions).

