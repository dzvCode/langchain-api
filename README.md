# FastAPI LangChain API

This is a simple FastAPI application with two POST endpoint that uses LangChain to interact with OpenAI's gpt-3.5-turbo model for Cash Me app.

## Setup

1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

```bash
uvicorn app.main:app --reload
```