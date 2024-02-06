from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
load_dotenv()
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
import langchain

langchain.debug = True

embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory='emb',
    embedding_function=embeddings
)

chat = ChatOpenAI()

retriever = RedundantFilterRetriever(embedding = embeddings, chroma= db)

chain = RetrievalQA.from_chain_type(
    llm =chat,
    retriever = retriever,
    chain_type='stuff'
)

result = chain.run('What is an interesting fact about the English Language?')
print(result)