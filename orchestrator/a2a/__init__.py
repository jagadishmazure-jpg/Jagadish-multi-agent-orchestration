"""A2A agent card, JSON-RPC handler, and the in-process mesh the crew joins."""

from .mesh import AgentMesh, default_mesh, publish_research_crew, reset_default_mesh
from .server import (
    AGENT_CARD_PATHS,
    ResearchCrewServer,
    research_crew_card,
    serve,
)

__all__ = [
    "AGENT_CARD_PATHS",
    "AgentMesh",
    "ResearchCrewServer",
    "default_mesh",
    "publish_research_crew",
    "research_crew_card",
    "reset_default_mesh",
    "serve",
]
