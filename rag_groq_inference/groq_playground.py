from groq import Groq
import os 
from dotenv import load_dotenv

load_dotenv()

# Create Groq client
client = Groq(
    # This is the default and can be omitted
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Create the config for how the chatbot should generate response
completion = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {
            "role": "user",
            "content": "Hi there"
        }
    ],
    temperature=1,
    max_tokens=2014,
    top_p=1,
    stream=True,
    stop=None
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
# print(completion)
    