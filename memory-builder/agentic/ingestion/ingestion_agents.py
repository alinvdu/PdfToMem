from langgraph.graph import StateGraph, END
from langgraph.types import Command as LGCommand
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
import json

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List
from configs import MCPConfig
import openai

from langchain.globals import set_debug
from typing_extensions import Annotated

from openai import OpenAI

from trustcall import create_extractor

client = OpenAI()

set_debug(True)

# ────────────────────────────────────────────────
# STATE DEFINITION
# ────────────────────────────────────────────────

def merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}

class StructureState(BaseModel):
    envelope: Annotated[dict, merge_dicts]
    config: MCPConfig

# ────────────────────────────────────────────────
# EXTRACTOR AGENT with create_react_agent
# ────────────────────────────────────────────────

class PDFBytesInput(BaseModel):
    pdf_bytes: bytes = Field(description="Raw content of the PDF file.")

@tool(args_schema=PDFBytesInput)
def parse_text(pdf_bytes: bytes) -> list:
    """
    Extracts raw, continuous text from a digital PDF.
    
    Returns:
        A single string containing the parsed text from all pages.
    """
    return [
        {
            "page": 1,
            "text": "This is the text from page 1 of the PDF.",
        }
    ]

class ExtractTablesInput(BaseModel):
    pdf_bytes: bytes = Field(..., description="Raw content of the PDF file.")

@tool(args_schema=PDFBytesInput)
def extract_tables(pdf_bytes: bytes) -> list:
    """
    Extracts structured tables from the PDF.

    Returns:
        A list of table representations in structured format (e.g., Markdown or CSV).
    """
    return [{
        "page": 1,
        "tables": [{
            "name": "Expenses",
            "data": [
                ["Category", "Amount"],
                ["Food", "$200"],
                ["Transport", "$150"],
                ["Utilities", "$100"]
            ]
        }]
    }]

@tool(args_schema=PDFBytesInput)
def extract_layout(pdf_bytes: bytes) -> dict:
    """
    Extracts the layout of the PDF.

    Returns:
        A dictionary containing the layout of the PDF.
    """
    return {"headers": ["Header A"], "footers": ["Footer B"]}

@tool(args_schema=PDFBytesInput)
def extract_images_from_pdf(pdf_bytes: bytes) -> list:
    """
    Extract images from an in-memory PDF (bytes object).

    Args:
        pdf_bytes (bytes): Raw content of a PDF file.

    Returns:
        List[dict]: Each dict contains page number and image data or references.
    """
    # Example stub output
    return [
        {"page": 1, "image_id": "fig1", "base64": "data:image/png;base64,..."},
        {"page": 3, "image_id": "fig2", "base64": "data:image/png;base64,..."}
    ]
    
class TakeScreenshotsInput(BaseModel):
    pdf_bytes: bytes = Field(description="Raw content of the PDF file.")
    pages: List[int] = Field(description="List of page numbers to screenshot.")
    
@tool(args_schema=TakeScreenshotsInput)
def take_screenshots(pdf_bytes: bytes, pages: list) -> list:
    """
    Take screenshots of specific pages from an in-memory PDF (bytes object).

    Args:
        pdf_bytes (bytes): Raw content of a PDF file.
        pages (list): List of page numbers to screenshot.

    Returns:
        List[dict]: Each dict contains page number and image data or references.
    """
    # Example stub output
    return [
        {"page": 1, "image_id": "fig1", "base64": "data:image/png;base64,..."},
        {"page": 3, "image_id": "fig2", "base64": "data:image/png;base64,..."}
    ]

@tool(args_schema=PDFBytesInput)
def ocr(pdf_bytes: bytes) -> list:
    """
    Extracts text from the PDF using OCR.

    Returns:
        A list of text extracted from the PDF.
    """
    return [{
        "page": 1,
        "text": "This is the OCR text extracted from page 1 of the PDF.",
    }]

extractor_llm = ChatOpenAI(model="gpt-3.5-turbo")
extractor_agent = create_react_agent(extractor_llm, [parse_text, extract_tables, extract_layout, ocr])

def extractor_node(state: StructureState) -> LGCommand[str]:
    pdf_data = state.envelope.get("pdf")
    screenshots = state.envelope.get("screenshots", [])
    cfg = state.config

    input_prompt = f"""
Analyze the PDF. Use tools to extract structure and content. At least one tool must be returned.
Available tools:
- parse_text
- ocr
- extract_layout
{('- extract_tables' if cfg.look_for_queryable_tables else '')}
{('- extract_images_from_pdf' if cfg.extract_and_embed_images else '')}
{('- take_screenshots' if cfg.extract_and_embed_images else '')}

Screenshots (preview only): {screenshots}
"""

    messages = [{"role": "user", "content": input_prompt}]
    result = extractor_agent.invoke(
        {"messages": messages}
    )

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    tool_outputs = {}
    for tm in tool_msgs:
        try:
            tool_outputs[tm.name] = json.loads(tm.content)
        except json.JSONDecodeError:
            tool_outputs[tm.name] = tm.content

    extracted = {
        "text":        tool_outputs.get("parse_text", []),
        "ocr":         tool_outputs.get("ocr", []),
        "layout":      tool_outputs.get("extract_layout", {}),
        "tables":      tool_outputs.get("extract_tables", []),
        "images":      tool_outputs.get("extract_images_from_pdf", []),
        "screenshots": tool_outputs.get("take_screenshots", []),
    }

    steps = state.envelope.get("steps_completed", []) + ["extractor"]
    print('Update for extractor is: ', extracted)
    new_envelope = {
        **state.envelope.copy(),
        "extractor": extracted,
        "steps_completed": steps
    }
    return LGCommand(goto="orchestrator_node", update={"envelope": new_envelope})


# ────────────────────────────────────────────────
# SEGMENTER using trust_call
# ────────────────────────────────────────────────

segmenter_llm = ChatOpenAI(model="gpt-3.5-turbo")

class SegmenterNode(BaseModel):
    sections: List[str]

bound_segmenter = create_extractor(
    segmenter_llm,
    tools=[SegmenterNode],
    tool_choice="SegmenterNode",
)

def segmenter_node(state: StructureState) -> LGCommand[str]:
    """Segment the document into high-level sections based on extracted text and layout."""
    prompt = f"""
You are a document segmenter. Based on the extracted text and layout, split the content into high-level sections.
Text: {state.envelope.get("extractor", {}).get("text", [])}
Layout: {state.envelope.get("extractor", {}).get("layout", [])}
"""
    messages = [{"role": "user", "content": prompt}]
    response = bound_segmenter.invoke(
        messages
    )
    steps = state.envelope.get("steps_completed", []) + ["segmenter"]
    new_envelope = {
        **state.envelope.copy(),
        "segment_output": response["responses"][0].dict(),
        "steps_completed": steps
    }
    return LGCommand(goto="orchestrator_node", update={"envelope": new_envelope})

# ────────────────────────────────────────────────
# SEMANTIC NODE using trust_call
# ────────────────────────────────────────────────

semantic_llm = ChatOpenAI(model="gpt-3.5-turbo")

class SemanticNode(BaseModel):
    entities: List[str]
    summary: List[str]

bound_semantic = create_extractor(
    semantic_llm,
    tools=[SemanticNode],
    tool_choice="SemanticNode",
)

def semantic_node(state: StructureState) -> LGCommand[str]:
    """Summarize sections and extract entities from the document."""
    prompt = f"""
Summarize each section and extract entities.
Sections: {state.envelope.get("segment_output", {}).get("sections", [])}
Text: {state.envelope.get("extractor", {}).get("text", [])}
"""
    messages = [{"role": "user", "content": prompt}]
    response = bound_semantic.invoke(
        messages
    )
    steps = state.envelope.get("steps_completed", []) + ["semantic"]
    new_envelope = {
        **state.envelope.copy(),
        "semantic_output": response["responses"][0].dict(),
        "steps_completed": steps
    }
    return LGCommand(goto="orchestrator_node", update={"envelope": new_envelope})

# ────────────────────────────────────────────────
# RELATIONSHIP MAPPER using direct LLM
# ────────────────────────────────────────────────

# Initialize the LangChain LLM interface
relationship_llm = ChatOpenAI(model="gpt-3.5-turbo")

def relationship_mapper(state: StructureState) -> LGCommand[str]:
    """Map relationships across document content using entities and summaries."""
    
    prompt = f"""
You are mapping relationships across document content.
Use section summaries and entities to infer relationships.
Entities: {state.envelope.get("semantic_output", {}).get("entities", [])}
Summaries: {state.envelope.get("semantic_output", {}).get("summary", [])}
"""
    messages = [{"role": "system", "content": prompt}]
    
    response = relationship_llm.invoke(messages)

    steps = state.envelope.get("steps_completed", []) + ["relationship_mapper"]
    new_envelope = {
        **state.envelope.copy(),
        "relationships": response.content,
        "steps_completed": steps
    }
    return LGCommand(
        goto="orchestrator_node",
        update={"envelope": new_envelope}
    )
# ────────────────────────────────────────────────
# FINAL AGGREGATOR NODE
# ────────────────────────────────────────────────

def aggregator_node(state: StructureState) -> LGCommand[str]:
    """Aggregate all extracted data into final structured format."""
    envelope = state.envelope
    final_data = {
        **envelope,
        "final_structured_output": {
            "text": envelope.get("text"),
            "tables": envelope.get("tables"),
            "sections": envelope.get("sections"),
            "summary": envelope.get("summary"),
            "entities": envelope.get("entities"),
            "relationships": envelope.get("relationships"),
        }
    }
    print("✅ Final structured data ready for ingestion.")
    return LGCommand(goto=END, update={"envelope": final_data})

# ────────────────────────────────────────────────
# ORCHESTRATOR NODE
# ────────────────────────────────────────────────

def orchestrator_node(state: StructureState) -> LGCommand[str]:
    """Orchestrate the flow between different processing nodes based on completion status."""
    envelope = state.envelope or {}
    steps_completed = envelope.get("steps_completed", [])

    print('step completed so far: ', steps_completed)

    if "extractor" not in steps_completed:
        print('running extractor agent')
        next_node = "extractor_agent"
    elif "segmenter" not in steps_completed:
        print('running segmenter')
        next_node = "segmenter_node"
    elif "semantic" not in steps_completed:
        print('running semantic')
        next_node = "semantic_node"
    elif "relationship_mapper" not in steps_completed:
        print('running relationship')
        next_node = "relationship_mapper"
    else:
        next_node = "aggregator_node"

    return LGCommand(goto=next_node, update={})

# ────────────────────────────────────────────────
# BUILD STRUCTURE GRAPH
# ────────────────────────────────────────────────

def build_ingestion_agents_graph() -> StateGraph:
    graph = StateGraph(StructureState)

    graph.add_node("orchestrator_node", orchestrator_node)
    graph.add_node("extractor_agent", extractor_node)
    graph.add_node("segmenter_node", segmenter_node)
    graph.add_node("semantic_node", semantic_node)
    graph.add_node("relationship_mapper", relationship_mapper)
    graph.add_node("aggregator_node", aggregator_node)

    graph.set_entry_point("orchestrator_node")
    graph.set_finish_point("aggregator_node")

    return graph.compile()

def run_ingestion_agents_graph(pdf_bytes: bytes, config: MCPConfig) -> dict:
    """Run the ingestion agents graph with proper initial state."""
    graph = build_ingestion_agents_graph()
    
    initial_state = StructureState(
        envelope={"pdf": pdf_bytes},
        config=config
    )
    
    result = graph.invoke(initial_state, {"recursion_limit": 25})
    return result['envelope']
