
from smolagents import LiteLLMModel, tool

# 1. Define the tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer.
        b: Second integer.
    """
    return a * b


# 2. Configure the local LLM
model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5:3b",
    api_base="http://localhost:11434",
    api_key="ollama",
    temperature=0,
)


# 3. Define the user message
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Multiply 17 by 23 using the multiply tool."
            }
        ]
    }
]


# 4. Generate a tool call
response = model.generate(
    messages=messages,
    tools_to_call_from=[multiply],
)


# 5. Inspect the response
print("\nCONTENT:")
print(response.content)

print("\nTOOL CALLS:")
print(response.tool_calls)

print("\nRAW RESPONSE:")
print(getattr(response, "raw", "Not available"))

