# Content Creator Chatbot

## Task Klimbb @Listed

The Content Creator Chatbot is an interactive chatbot designed to generate concise and appealing summaries for potential investors based on the description of a content creator. It leverages natural language processing (NLP) techniques and the OpenAI GPT-3 language model to provide personalized responses.

```bash
pip install -r requirements.txt
```

## Usage

### 1. Clone the repository:

```bash
git clone https://github.com/khushpatel2002/ContentCreatorChatbot.git
 ```

### 2. Create a virtual environment (optional but recommended):

```bash
make venv 
```
or 
```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment:

```bash
source .venv/bin/activate
```

### 4. Install the dependencies:
```bash 
make install 
```
or 
```bash 
pip install -r requirements.txt
```

### 5. Open the .env file and replace <OPENAI_API_KEY> with your actual OpenAI API key.

### 6. Run the chatbot: 
```bash 
make spawn 
```
or 
```bash 
python3 ContentCreatorChatbot.py
```
### Follow the prompts and provide the necessary information about the content creator. The chatbot will generate a concise and appealing summary based on the input.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
- NLTK - Natural Language Toolkit library
- Transformers
- Optimum - Library for optimizing models.
- OnnxRuntime - Open Neural Network Exchange (ONNX) runtime 
- OpenAI - GPT-3

