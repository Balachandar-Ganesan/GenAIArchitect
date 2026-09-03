import lancedb
import pyarrow as pa
from contextgem import ContextGem  # Structured data extraction framework
from openai import OpenAI

# 1. Initialize Clients
client = OpenAI()
gem = ContextGem(api_key="your_contextgem_key")

# 2. Extract Narrative Structural Metadata using ContextGem
# We use ContextGem to parse 'movies.txt' and confidently isolate character behaviors.
with open("movies.txt", "r") as f:
    movie_corpus = f.read()

extraction_prompt = """
Analyze the text and extract all mappings of Actors to their Characters. 
Pay specific attention to plot twists:
1. Did the actor play distinct multiple characters?
2. Did the character simply wear a disguise or fake an identity? 
Provide a clear, comma-separated mapping with plot context.
"""

# Extracting high-fidelity context blocks to prevent chunk fragmentation
structured_contexts = gem.extract(
    text=movie_corpus,
    instruction=extraction_prompt,
    response_format="list" # Formatted for downstream embedding ingestion
)

# 3. Setup LanceDB Vector Database
db = lancedb.connect("./movie_rag_db")

# Define schema enforcing strict plot context segregation
schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 1536)), # OpenAI text-embedding-3-small dimensions
    pa.field("text", pa.string()),
    pa.field("actor", pa.string())
])

# Utility function to generate embeddings
def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Prepare and insert data into LanceDB
data_to_insert = []
for item in structured_contexts:
    embedding = get_embedding(item)
    data_to_insert.append({
        "vector": embedding,
        "text": item,
        "actor": item.split("–")[0].strip() # Assuming standard 'Actor - Context' format
    })

table = db.create_table("movie_plots", schema=schema, mode="overwrite")
table.add(data_to_insert)

# 4. RAG Query & Inference Function
def ask_rag(query_str):
    # Retrieve relevant context from LanceDB
    query_vector = get_embedding(query_str)
    search_results = table.search(query_vector).limit(3).to_list()
    
    retrieved_context = "\n".join([res["text"] for res in search_results])
    
    # System prompt forces the model to carefully analyze structural character deceit
    system_prompt = (
        "You are an expert film analyst. Use the provided context to answer the user's question. "
        "Pay meticulous attention to whether an actor plays two entirely separate physical entities, "
        "or if the character is intentionally faking an identity/wearing a disguise within the plot story."
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{retrieved_context}\n\nQuestion: {query_str}"}
        ]
    )
    return response.choices[0].message.content

# 5. Execution Test
answer = ask_rag("Did Christian Bale play multiple characters or fake his identity?")
print(answer)
