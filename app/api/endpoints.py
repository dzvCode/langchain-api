from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from transformers import pipeline
from openai import OpenAI

app = FastAPI()
router = APIRouter()

# Initialize once on startup
loader = TextLoader("./data/data.txt")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(data)
vector = ObjectBox.from_documents(documents, OpenAIEmbeddings(), embedding_dimensions=768)

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = hub.pull("rlm/rag-prompt")
qa_chain = RetrievalQA.from_chain_type(
  llm,
  retriever=vector.as_retriever(),
  chain_type_kwargs={"prompt": prompt}
)

class Query(BaseModel):
    query: str

@router.post("/query/")
async def create_query(query: Query):
  try:
      result = qa_chain({"query": query.query})
      return {"query": query, "answer": result["result"]}
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))


# Endpoint to interpret the message
@router.post("/interpret")
async def interpret_message(query: Query):
  client = OpenAI()
  allowed_responses = ["moneyReceived", "moneyGiven", "moneyBalance", "invalidQuery"]
  
  # Make the request to OpenAI
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You must interpret the user's request and respond with only one of these four words (moneyReceived, moneyGiven, moneyBalance, invalidQuery). You cannot respond with any other word. Ensure that the response pertains only to the user's query and does not address questions about any other person. If the user's query is unclear or does not match any of the categories, respond with 'invalidQuery'."},
        {"role": "user", "content": query.query},
    ],
    max_tokens=10  # Limit response to one word
  )

  # Validate and process the response
  response = completion.choices[0].message.content.strip()

  if response in allowed_responses:
    return {"query": query, "answer": response}

  else:
    raise HTTPException(status_code=400, detail=f"Respuesta no válida: {response}")


app.include_router(router)
