
from smolagents import ToolCallingAgent, LiteLLMModel, tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer.
        b: Second integer.
    """
    return a * b

model = LiteLLMModel(
    model_id="ollama_chat/qwen3:4b",
    api_base="http://localhost:11434",
    api_key="ollama",
)

agent = ToolCallingAgent(
    tools=[multiply],
    model=model,
    max_steps=3,
    add_base_tools=False,
)

result = agent.run("What is 17 multiplied by 23? Use the multiply tool.")
print("Final result:", result)
