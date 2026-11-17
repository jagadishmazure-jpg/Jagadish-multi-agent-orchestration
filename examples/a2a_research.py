"""Publish the research crew on the in-process A2A mesh and send one topic."""

from orchestrator.a2a import AgentMesh, publish_research_crew
from orchestrator.frameworks.mock import ResearchMockLLM

mesh = AgentMesh()
server = publish_research_crew(mesh, llm=ResearchMockLLM(), backend="builtin")
print("mesh agents:", ", ".join(mesh.names()))
print("card:", server.card["name"])
print("skill:", server.card["skills"][0]["id"])
response = mesh.send("Research Crew", "Research: The history of Python")
task = response["result"]
by_name = {item["name"]: item["parts"][0]["text"] for item in task["artifacts"]}
report = by_name["research-report"]
review = by_name["critic-review"]
print("state:", task["status"]["state"])
print(
    "approved:",
    "APPROVED" in review and "Guido van Rossum" in report and "1991" in report,
)
