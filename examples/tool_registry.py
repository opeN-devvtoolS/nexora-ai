
import tools

print("Registered tool names:")

for tool in tools.TOOL_LIST:
    print(tool["function"]["name"])

print("\nChecking Python function mapping:")

for name, function in tools.TOOLS.items():
    print(name, "->", function.__name__)
