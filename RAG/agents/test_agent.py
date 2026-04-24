from agents.agent import Agent
from agents.tool_registry import ToolRegistry
from agents.react_loop import ReActLoop
from agents.tools import rag_search, calculator


registry = ToolRegistry()

registry.register(
    "rag_search",
    rag_search,
    "Search knowledge base using RAG"
)

registry.register(
    "calculator",
    calculator,
    "Evaluate math expressions"
)


agent = Agent(registry)

loop = ReActLoop(agent, registry)


question = "Hướng dẫn tôi chụp hình audit"

result = loop.run(question)

print("\nFINAL ANSWER:\n", result)