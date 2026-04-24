class ToolRegistry:

    def __init__(self):
        self.tools = {}

    def register(self, name, func, description):

        self.tools[name] = {
            "function": func,
            "description": description
        }

    def get_tool(self, name):

        return self.tools.get(name)

    def list_tools(self):

        return {
            name: tool["description"]
            for name, tool in self.tools.items()
        }