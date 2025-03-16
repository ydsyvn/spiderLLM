from ollama import chat

print("Send a message: ", end='')
prompt = input()

stream = chat(
    model='tinyllama:latest', 
    messages=
    [
        {
            'role': 'user', 
            'content': prompt
        },
    ], 
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)