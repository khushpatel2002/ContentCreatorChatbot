# Content Creator Chatbot

## Dependencies

- nltk
- torch
- sentencepiece
- transformers 
- optimum
- onnxruntime
- onnx
- langchain 
- openai
- python-dotenv

```Python
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

