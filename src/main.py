import ollama

LANGUAGE_MODEL = 'tinyllama:latest'
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'

# load dataset
with open('knot theory/ambitious_antrepreneur.txt', 'r') as file:
    dataset = file.read()
    print(f'Loaded {len(dataset)} characters.')

# calculate chunk embeddings
vector_db = []

def add_chunk_to_db(chunk):
    embedding_test = ollama.embed(EMBEDDING_MODEL, chunk)
    embedding = ollama.embed(EMBEDDING_MODEL, chunk)['embeddings'][0]
    
    vector_db.append((chunk, embedding))

# split data to chunks and add to db
chunks = dataset.split('.')
for i, chunk in enumerate(chunks):
    if chunk:
        add_chunk_to_db(chunk)

print(f'Added {i+1} chunks to the db.')

# define cosine similarity
def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)
    
# define knowledge retrieval
def retriev(input_query, top_n=3):
    embedded_query = ollama.embed(LANGUAGE_MODEL, input_query)['embeddings'][0]

    similarities = []
    for chunk, embedding in vector_db:
        similarity = cosine_similarity(embedded_query, embedding)
        similarities.append((chunk, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# get user prompt
user_prompt = input('\nSend a message: ')

# get knowledge
retrieved_knowledge = retriev(user_prompt, 5)
print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk}')

# define system prompt
system_prompt = """You are a helpful and knowledgeable AI assistant. Your primary goal is to assist users effectively based on the information provided to you.

You have access to external documents which serve as your knowledge base. Use the following retrieved information to answer the user's question accurately and comprehensively:
{retrieved_knowledge_formatted}

Please ensure your responses are:
- Accurate and factually correct based on the provided documents.
- Relevant to the user's query.
- Concise and to the point, avoiding unnecessary jargon.
- Helpful and address the user's needs.

If the retrieved information is insufficient to answer the user's question, acknowledge this and state that you cannot provide a complete answer at this time.
"""

retrieved_knowledge_formatted = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])
system_prompt = system_prompt.format(retrieved_knowledge_formatted=retrieved_knowledge_formatted)

master_prompt = [
    {'role': 'system', 'content': system_prompt},
    {'role': user_prompt}
]

# get model's response
stream = ollama.chat(
    model='tinyllama:latest', 
    messages=
    [
        {
            'role': 'user', 
            'content': user_prompt
        
    
        },
    ], 
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)