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
report = task["artifacts"][0]["parts"][0]["text"]
print("state:", task["status"]["state"])
print(
    "approved:",
    "APPROVED" in report and "Guido van Rossum" in report and "1991" in report,
)
