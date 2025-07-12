# Importing the HuggingFaceEmbeddings class for generating text embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Importing function to load environment variables from a .env file
from dotenv import load_dotenv

# Importing cosine_similarity function to measure similarity between vectors
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from a .env file, if present
load_dotenv()

# A list of technical documents (sentences) to be embedded for similarity comparison
docs = [
    "Neural networks learn to approximate complex functions by adjusting weights through backpropagation.",
    "Docker containers provide a lightweight way to package and deploy applications across environments.",
    "A relational database uses structured query language (SQL) to manage and retrieve data efficiently.",
    "The time complexity of binary search is O(log n), making it faster than linear search for sorted arrays.",
    "Random forest is an ensemble learning method that combines multiple decision trees to improve accuracy.",
    "RESTful APIs use HTTP methods to perform CRUD operations and support stateless communication.",
    "Cloud computing allows scalable, on-demand access to computing resources over the internet.",
    "Principal Component Analysis (PCA) is a dimensionality reduction technique used to capture variance in data.",
    "Version control systems like Git help track changes in code and enable collaborative development.",
    "Anomaly detection algorithms identify data points that significantly deviate from the expected pattern."
]

# The user's search query to find the most relevant document
user_query = "What algorithms are being used in Deep Learning?"

# Initialize the HuggingFace embedding model (MiniLM)
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings for all the documents
doc_embeeding = embedding.embed_documents(docs)

# Generate embedding for the user's query
query_embeeding = embedding.embed_query(user_query)

# Compute cosine similarity between the query embedding and all document embeddings
scores = cosine_similarity([query_embeeding], doc_embeeding)[0]

# Find the document index with the highest similarity score
index, score = sorted(list(enumerate(scores)), key = lambda x: x[1], reverse=True)[0]

# Print the most relevant document along with the similarity score
print(f"The maching document of the user query: '{user_query}' is: '{docs[index]}' with the similarity score of '{score}'")