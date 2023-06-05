venv:
	python3 -m venv .venv

install:
	pip install -r requirements.txt

spawn:
	python3 ContentCreatorChatbot.py