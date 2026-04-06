import os
from groq import Groq

#client = Groq(api_key=os.environ["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"])
from groq import Groq

client = Groq(api_key="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")
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ],
)

print(response.choices[0].message.content)