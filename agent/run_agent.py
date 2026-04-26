import re
from typing import Callable, Optional, Tuple

try:
    # Legacy OpenClaw API used by older tutorials.
    from openclaw import Agent as LegacyOpenClawAgent
except ImportError:
    LegacyOpenClawAgent = None

from glassbox_tool import GLASSBOX_TOOL_SCHEMA, execute_glassbox


class FallbackAgent:
    """Small local fallback that preserves the chat/register_tool interface."""

    def __init__(self, name: str, system_prompt: str) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self._tool_schema: Optional[dict] = None
        self._tool_executable: Optional[Callable[..., str]] = None

    def register_tool(self, schema: dict, executable: Callable[..., str]) -> None:
        self._tool_schema = schema
        self._tool_executable = executable

    def _extract_inputs(self, user_input: str) -> Tuple[Optional[str], Optional[str]]:
        csv_match = re.search(r"([\w.-]+\.csv)", user_input)
        target_match = re.search(
            r"target(?:\s+column)?\s*(?:=|is|:)?\s*([A-Za-z_][\w]*)",
            user_input,
            flags=re.IGNORECASE,
        )

        csv_filename = csv_match.group(1) if csv_match else None
        target_column = target_match.group(1) if target_match else None
        return csv_filename, target_column

    def chat(self, user_input: str) -> str:
        if not self._tool_executable:
            return "No tool is registered."

        csv_filename, target_column = self._extract_inputs(user_input)
        if not csv_filename or not target_column:
            return (
                "I can run glassbox_autofit for you. Please include both the CSV file and "
                "target column, for example: analyze test_model.csv target=target"
            )

        return self._tool_executable(
            csv_filename=csv_filename, target_column=target_column
        )


def build_agent(name: str, system_prompt: str):
    if LegacyOpenClawAgent is not None:
        return LegacyOpenClawAgent(name=name, system_prompt=system_prompt)
    return FallbackAgent(name=name, system_prompt=system_prompt)


# 1. Initialize the OpenClaw Agent
agent = build_agent(
    name="GlassBox Assistant",
    system_prompt="""You are an expert Data Scientist. You have access to the 'glassbox_autofit' tool. 
    When a user asks you to build a model, call the tool. Once the tool returns the JSON report, 
    summarize the best model, its accuracy, and any outliers found during EDA in a friendly way. Do not output raw JSON to the user.""",
)

# 2. Register your Docker tool with the Agent
agent.register_tool(schema=GLASSBOX_TOOL_SCHEMA, executable=execute_glassbox)

# 3. Start the chat interface
if __name__ == "__main__":
    print("GlassBox AI initialized. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        # The agent processes the prompt and decides if it needs to trigger Docker
        response = agent.chat(user_input)
        print(f"\nAgent: {response}")
