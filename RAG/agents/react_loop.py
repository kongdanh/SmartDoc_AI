class ReActLoop:

    def __init__(self, agent, registry):

        self.agent = agent
        self.registry = registry

    def run(self, question, max_steps=5):

        scratchpad = ""

        for step in range(max_steps):

            response = self.agent.think(question, scratchpad)

            print("Agent:", response)

            if "Action:" not in response:
                return response

            lines = response.split("\n")
            action = None
            action_input = None

            for line in lines:

                if line.startswith("Action:"):
                    action = line.replace("Action:", "").strip()

                if line.startswith("Action Input:"):
                    action_input = line.replace("Action Input:", "").strip()

            tool = self.registry.get_tool(action)

            if tool is None:
                observation = "Tool not found"

            else:
                observation = tool["function"](action_input)

            scratchpad += f"""
{response}
Observation: {observation}
"""

        return "Max steps reached"