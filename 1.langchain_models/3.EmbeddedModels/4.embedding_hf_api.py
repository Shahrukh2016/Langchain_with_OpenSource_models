from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Defining text
text = 'Hi , Machine Learning is impressive!'

# Initializing embedding model
embedding = HuggingFaceEndpointEmbeddings(model= 'sentence-transformers/all-MiniLM-L6-v2')

# Sending to LLM
result = embedding.embed_query(text=text)

# Printing result
print(result)
print(len(result))