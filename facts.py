from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 200,
    chunk_overlap = 0
)

embeddings = OpenAIEmbeddings()
emb = embeddings.embed_query('hi there')

loader = TextLoader('facts.txt')
docs = loader.load_and_split(
    text_splitter=text_splitter
)

db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory='emb'
)


results = db.similarity_search_with_score("What is a fact about ostriches?")

for result in results:

    print('\n')
    print(result[1])
    print(result[0].page_content)