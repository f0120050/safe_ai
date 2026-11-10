
from litellm import completion

response = completion(
    model="ollama_chat/qwen2.5:3b",
    api_base="http://localhost:11434",
    api_key="ollama",
    messages=[
        {
            "role": "user",
            "content": "Multiply 17 by 23 using the multiply tool."
        }
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                }
            }
        }
    ],
    tool_choice="auto",
    temperature=0,
)

print("Content:", response.choices[0].message.content)
print("Tool calls:", response.choices[0].message.tool_calls)
