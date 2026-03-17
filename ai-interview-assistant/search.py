from sentence_transformers import SentenceTransformer
import requests

model = SentenceTransformer("all-MiniLM-L6-v2")

query = input("Enter your search query: ")

embedding = model.encode(query).tolist()

response = requests.post(
    "http://localhost:8080/api/v1/search",
    json={
        "vector": embedding,
        "top_k": 3
    }
)

print("Search Results:")
print(response.json())