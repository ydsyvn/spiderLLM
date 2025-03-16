from llama_cpp import Llama

# Load the model
llm = Llama(model_path="gemma-2b.Q4_K_M.gguf", n_ctx=2048)

# Generate a response
response = llm("What is Retrieval-Augmented Generation (RAG)?", max_tokens=200)
print(response["choices"][0]["text"])
