from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Documents
doc = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills",
    "Sachin Tendulkar is known for his elegant batting and record-breaking double centuries",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers"
]

# Query
query = "Tell me about dhoni"

# Generate embeddings
doc_embeddings = embedding.embed_documents(doc)
query_embedding = embedding.embed_query(query)

# Compute cosine similarity
similarity = cosine_similarity([query_embedding], doc_embeddings)[0]

# Output similarity scores
index,score=sorted(list(enumerate(similarity)),key=lambda x:x[1])[-1]

print("USER QUERY:",query)
print(doc[index])
print("SIMILARITY:",score)



