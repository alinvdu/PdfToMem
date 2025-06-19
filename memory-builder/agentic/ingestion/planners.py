from agentic.ingestion.ingestion_agents import run_ingestion_agents_graph
from agentic.ingestion.ingestion_planner import run_ingestion_planner_graph
from configs import MCPConfig
from typing import Dict, Any

class StructureState:
    def __init__(self, envelope: dict):
        self.envelope = envelope
        self.plan = None

def run_ingestion_pipeline(pdf_bytes: bytes, config: MCPConfig) -> Dict[str, Any]:
    # Phase 1: Document structure extraction
    structure_envelope = run_ingestion_agents_graph(pdf_bytes, config)
    
    # Phase 2: Index planning based on structured envelope (includes hierarchy)
    ingestion_plan = run_ingestion_planner_graph(structure_envelope, config)

    return {
        "structure_envelope": structure_envelope,
        "ingestion_plan": ingestion_plan
    }
