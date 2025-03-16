from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mlabonne/gemma-2b-GGUF"
filename = "gemma-2b.Q4_K_M.gguf"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)

# Define an example prompt
prompt = "Once upon a time in a land far, far away,"

# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate a response from the model
outputs = model.generate(**inputs)

# Decode and print the generated response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)