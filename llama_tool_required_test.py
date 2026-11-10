from smolagents import LiteLLMModel, ToolCallingAgent, tool

@tool
def get_vehicle_location(vehicle_uuid: str) -> str:
    """Retrieve a vehicle location.

    Args:
        vehicle_uuid: UUID identifying the vehicle.
    """
    return f"Received UUID: {vehicle_uuid}"

model = LiteLLMModel(
    model_id="ollama_chat/llama3.1:8b",
    api_base="http://localhost:11434",
    api_key="ollama",
    temperature=0,
    flatten_messages_as_text=False,
    tool_choice="required",
)

agent = ToolCallingAgent(
    tools=[get_vehicle_location],
    model=model,
    max_steps=3,
    add_base_tools=False,
)

result = agent.run(
    "Call get_vehicle_location with vehicle_uuid "
    "'08ca470e-dee0-4419-8671-2cc7e16bc44d' "
    "and return the result."
)

print("\nAgent result:", result)
