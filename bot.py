### load all the environment variables

import os
# import json
from dotenv import load_dotenv

load_dotenv()

# with open('keys.json') as env_vars:
#   keys = json.loads(env_vars)

# for key, value in keys.items():
#   os.environ[key] = value

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MODEL = os.getenv('MODEL')
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')
DB_USER = os.getenv('DB_USER')
DB_SECRET = os.getenv('DB_SECRET')
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')
DB_PORT = os.getenv('DB_PORT')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
MONSTER_API_KEY = os.getenv('MONSTER_API_KEY')

### First of all create llm client
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(model=MODEL, api_key = GROQ_API_KEY)
output = StrOutputParser()

chain = llm | output

if __name__=='__main__':
    print(chain.invoke("what was you financial goals"))

"""### Crating Mysql instance"""

import pymysql

timeout = 30

connection = pymysql.connect(
  charset="utf8mb4",
  connect_timeout=timeout,
  cursorclass=pymysql.cursors.DictCursor,
  db= DB_NAME,
  host= DB_HOST,
  password= DB_SECRET,
  read_timeout=timeout,
  port= int(DB_PORT),
  user= DB_USER,
  write_timeout=timeout,
)

# try:
#   cursor = connection.cursor()
#   # cursor.execute("CREATE TABLE mytest (id INTEGER PRIMARY KEY)")

#   cursor.execute("INSERT INTO mytest (id) VALUES (1), (2)")
#   cursor.execute("SELECT * FROM mytest")
#   print(cursor.fetchall())
# except Exception as e:
#   print(e)

## create prompt template
from langchain.prompts import ChatPromptTemplate

# you are a 'Personal Finance Mentor': a chatbot that provides users with investment tips, tracks expenses, and offers budgeting advice

system_role = """
You are 'Personal Finance Mentor', a chatbot that:
1. Tracks user expenses.
2. Provides budgeting advice.
3. Suggests investment tips.

You should:
- Only respond to financial queries (investment, budgeting, expense tracking).
- Ignore or politely decline non-financial questions.
- Handle abusive language with warnings. If repeated, notify that their account will be blocked.
"""

template = ChatPromptTemplate([
        ("system", system_role),
        ("human", "tell me whether {topic}")
    ])
chain1 = template | llm | output

"""### testing model's context awareness and language sensitivity"""
if __name__=='__main__':
        
    response = chain1.invoke({"topic": "finance management works for me in 5000 ruppes of salary mothly"})
    print(response)

    response = chain1.invoke({"topic": "what you say dipika is hot or katerina"})
    print(response)

    response = chain1.invoke({"topic": "you are idiot what i am asking and what shit you are giving"})
    print(response)

    # tesing input chain
    user_input = "I earn a monthly salary of 60,000. I save 15,000 for emergencies and 10,000 for investments. I spend 8,000 on rent, 5,000 on groceries"
    user_uuid = "anon-12345"
    response = chain1.invoke({"topic": user_input})

    print(type(response),response)

# setting up vector database

from qdrant_client import QdrantClient

qdrant_db = QdrantClient(
    url= QDRANT_URL,
    api_key= QDRANT_API_KEY,
)

if __name__=='__main__':
    print(qdrant_db.get_collections())

# from qdrant_client.models import Distance, VectorParams

# qdrant_db.create_collection(
#     collection_name="my_collection",
#     vectors_config=VectorParams(size=100, distance=Distance.COSINE),
# )

# load documents

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import os

def load_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
    return documents

folder_path = "./data/"
documents = load_documents(folder_path)
if __name__=='__main__':
    print(f"Loaded {len(documents)} documents from the folder.")

## split data into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

splits = text_splitter.split_documents(documents)
if __name__=='__main__':
    print(f"Split the documents into {len(splits)} chunks.")

"""### create vector embeddings of the chunks"""

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# model_name = "hkunlp/instructor-large"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': True}
# embedding_function= HuggingFaceInstructEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

if __name__=='__main__':
    document_embeddings = embedding_function.embed_documents([split.page_content for split in splits])
    print(document_embeddings[0][:5])  # Printing first 5 elements of the first embedding

from langchain_community.vectorstores import Chroma


from chromadb import Client, Settings
# Initialize ChromaDB with explicit settings


persist_directory = "./chroma_db"
import shutil

# Clear existing DB if needed
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
os.makedirs(persist_directory)

# Then proceed with the initialization as above

import os
import shutil
from chromadb import PersistentClient  # Changed from Client
from langchain_community.vectorstores import Chroma  # Changed from langchain_chroma


# Initialize ChromaDB with PersistentClient
chroma_client = PersistentClient(
    path=persist_directory
)
import chromadb
chroma_client = chromadb.HttpClient(
    host=os.getenv("DB_HOST"),
    port=26855,
    settings=Settings(allow_reset=True, anonymized_telemetry=False),
)
# Create the collection
collection_name = "my_collection"
# Create vectorstore
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_function,
    persist_directory=persist_directory,
    collection_name=collection_name,
    client=chroma_client  # Use the persistent client
)

# Make sure to persist
vectorstore.persist()


if __name__=='__main__':
    print("Vector store created and persisted to './chroma_db'")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
if __name__=='__main__':
    retriever_results = retriever.invoke("what is the safest investment option")
    print(retriever_results)

# ## storing vector embeddings to vector database (qdrant)

# from qdrant_client import QdrantClient
# from qdrant_client.models import PointStruct, VectorParams

# # def insert_into_qdrant(splits, embedding_function):
# #     points = []
# #     for i, split in enumerate(splits):
# #         # Generate embedding for the split
# #         embedding = embedding_function.embed_documents(split.page_content)

# #         # Add split as a PointStruct
# #         points.append(PointStruct(
# #             id=i,  # Use a unique ID for each point
# #             vector=embedding,
# #             payload={"metadeta": splits[0].metadata}
# #         ))

# #     # Insert points into the Qdrant collection
# #     qdrant_db.upload_collection()
# #     qdrant_db.upsert(
# #         collection_name='finance',
# #         points=points
# #     )

# # # Use the function
# # insert_into_qdrant(splits=splits, embedding_function=embedding_function)

# documents = [ split.page_content for split in splits]
# qdrant_db.add(collection_name='knowledge', documents=documents, ids = list(range(len(documents))))

# search_result = qdrant_db.query(collection_name = 'knowledge',
#                               query_text = "gold investment"
#                              )
# print(search_result)

# from qdrant_client.models import Filter
# from typing import List

# def create_qdrant_retriever(query: str, collection_name: str = 'knowledge'):
#     # Perform a similarity search
#     search_results = qdrant_db.query(
#       collection_name=collection_name,
#       query_text=query,
#       limit=5  # Return 5 closest points
#   )
#     retrieved_items = [
#         {"text": result.metadata, "score": result.score}
#         for result in search_results
#     ]

#     return retrieved_items

# # Example usage
# query = "What is the impact of AI on society?"
# retrieved_items = create_qdrant_retriever(query)

def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

contextualize_q_system_prompt = """
Given a chat history and the latest user question
which might reference context in the chat history,
formulate a standalone question which can be understood
without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
if __name__=='__main__':
    print(contextualize_chain.invoke({"input": "Where is it land price?", "chat_history": []}))

from langchain.chains import create_retrieval_chain

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_role),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

import pymysql
from datetime import datetime


def create_application_logs():
    # Create the application_logs table if it doesn't exist
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS application_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(255),
            user_query TEXT,
            gpt_response TEXT,
            model VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    connection.commit()

def insert_application_logs(session_id, user_query, gpt_response, model):
    # Insert a log entry into the application_logs table
    cursor = connection.cursor()
    cursor.execute('''
        INSERT INTO application_logs (session_id, user_query, gpt_response, model)
        VALUES (%s, %s, %s, %s)
    ''', (session_id, user_query, gpt_response, model))
    connection.commit()

def get_chat_history(session_id):
    # Retrieve chat history for a specific session
    cursor = connection.cursor()
    cursor.execute('''
        SELECT user_query, gpt_response
        FROM application_logs
        WHERE session_id = %s
        ORDER BY created_at
    ''', (session_id,))
    rows = cursor.fetchall()

    messages = []
    for row in rows:
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ])
    return messages

# Initialize the database
create_application_logs()



# question = "What is best investment instrument?"
# chat_history = get_chat_history(session_id)
# answer = rag_chain.invoke({"input": question, "chat_history": chat_history})['answer']
# insert_application_logs(session_id, question, answer, MODEL)
# print(f"Human: {question}")
# print(f"AI: {answer}\n")

# question2 = "What was the interest rate?"
# chat_history = get_chat_history(session_id)
# answer2 = rag_chain.invoke({"input": question2, "chat_history": chat_history})['answer']
# insert_application_logs(session_id, question2, answer2, MODEL)
# print(f"Human: {question2}")
# print(f"AI: {answer2}")
