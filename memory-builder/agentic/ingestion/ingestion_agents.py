import base64
from langgraph.graph import StateGraph, END
from langgraph.types import Command as LGCommand
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
import json

from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional
from configs import MCPConfig
from langchain.globals import set_debug
from typing_extensions import Annotated
from openai import OpenAI
import fitz
import camelot
import tempfile
from io import BytesIO
import os
import pytesseract
from PIL import Image
import io

client = OpenAI()

set_debug(True)

def save_base64_images_to_disk(base64_images: list, output_dir: str = "screenshots"):
    os.makedirs(output_dir, exist_ok=True)
    for i, img_b64 in enumerate(base64_images, 1):
        img_data = base64.b64decode(img_b64)
        with open(os.path.join(output_dir, f"screenshot_page_{i}.png"), "wb") as f:
            f.write(img_data)

def cap_text_length(text_list: list, max_chars: int = 8000) -> str:
    """Concatenate list of page texts and cap total character length."""
    combined = "\n\n".join((page["text"] or "") for page in text_list)
    return combined[:max_chars]

def cap_text_length_raw(text_list: list, max_chars: int = 8000) -> str:
    """Concatenate list of page texts and cap total character length."""
    combined = "\n\n".join((elem or "") for elem in text_list)
    return combined[:max_chars]

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
    Extracts raw, continuous text from a digital PDF. This is a good idea to use for nicely structured PDF document with clear text that can be extracted.
    If the PDF can only be parsed by OCR this tool will not work.
    
    Returns:
        A single string containing the parsed text from all pages.
    """
    results = []
    
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            results.append({
                "page": page_num + 1,
                "text": text.strip(),
            })
    
    return results

class ExtractTablesInput(BaseModel):
    pdf_bytes: bytes = Field(..., description="Raw content of the PDF file.")

@tool(args_schema=PDFBytesInput)
def extract_tables(pdf_bytes: bytes) -> list:
    """
    Extracts structured tables from the PDF itself using PDF extractor tools. If the PDF can only be parsed by OCR this tool will not work.

    Returns:
        A list of table representations in structured format (e.g., Markdown or CSV).
    """
    tables_data = []
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_file.flush()

        tables = camelot.read_pdf(tmp_file.name, pages='all')
        
        for i, table in enumerate(tables):
            tables_data.append({
                "page": table.page,
                "table_index": i,
                "data": table.df.values.tolist()  # returns the table as list of rows
            })
    
    return tables_data

# @tool(args_schema=PDFBytesInput)
# def extract_layout(pdf_bytes: bytes) -> dict:
#     """
#     Extracts the layout of the PDF using PDF extractor tools. If the PDF can only be parsed by OCR this tool will not work.

#     Returns:
#         A dictionary containing the layout of the PDF.
#     """
#     return {"headers": ["Header A"], "footers": ["Footer B"]}

@tool(args_schema=PDFBytesInput)
def extract_images_from_pdf(pdf_bytes: bytes) -> list:
    """
    Extract images from an in-memory PDF (bytes object).

    Args:
        pdf_bytes (bytes): Raw content of a PDF file.

    Returns:
        List[dict]: Each dict contains page number and image data or references.
    """
    results = []
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_index in range(len(pdf_doc)):
        page = pdf_doc[page_index]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]   
            image_ext = base_image["ext"]  # 'png', 'jpeg', etc.

            base64_str = base64.b64encode(image_bytes).decode("utf-8")
            results.append({
                "page": page_index + 1,
                "image_id": f"page{page_index + 1}_img{img_index + 1}",
                "base64": f"data:image/{image_ext};base64,{base64_str}"
            })

    return results
    
class TakeScreenshotsInput(BaseModel):
    pdf_bytes: bytes = Field(description="Raw content of the PDF file.")
    pages: List[int] = Field(description="List of page numbers to screenshot.")
    
@tool(args_schema=TakeScreenshotsInput)
def take_screenshots(pdf_bytes: bytes) -> list:
    """
    Take screenshots of specific pages from an in-memory PDF (bytes object).

    Args:
        pdf_bytes (bytes): Raw content of a PDF file.
        pages (list): List of page numbers to screenshot.

    Returns:
        List[dict]: Each dict contains page number and image data or references.
    """
    results = []
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    for index, page in enumerate(pdf):
        page_number = index + 1  # 1-indexed page number
        pix = page.get_pixmap(dpi=150)
        img_bytes = BytesIO(pix.tobytes("png"))
        base64_str = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

        results.append({
            "page": page_number,
            "image_id": f"screenshot_page_{page_number}",
            "base64": f"data:image/png;base64,{base64_str}"
        })

    return results

@tool(args_schema=PDFBytesInput)
def ocr(pdf_bytes: bytes) -> list:
    """
    Extracts text from the PDF using OCR. This tool is useful for when the PDF has hand written text, text that doesn't seem extractable with standard PDF tools.

    Returns:
        A list of text extracted from the PDF.
    """
    results = []
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_index, page in enumerate(pdf):
            pix = page.get_pixmap(dpi=300)  # Higher DPI improves OCR accuracy
            image = Image.open(io.BytesIO(pix.tobytes("png")))

            text = pytesseract.image_to_string(image)
            results.append({
                "page": page_index + 1,
                "text": text.strip()
            })

        pdf.close()
    except Exception as e:
        results.append({"error": str(e)})

    return results

tool_node = ToolNode([parse_text, extract_tables, ocr, extract_images_from_pdf, take_screenshots])

### Utils for taking screenshots used for processing the tool usage
def pdf_bytes_to_base64_images(pdf_bytes: bytes, num_of_screenshots: int):
    base64_images = []
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    num_pages = len(pdf)
    pages_to_render = min(num_of_screenshots, num_pages)

    for page in pdf[:pages_to_render]:
        pix = page.get_pixmap(dpi=150)
        img_bytes = BytesIO(pix.tobytes("png"))

        # Encode to base64
        base64_str = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        base64_images.append(base64_str)

    return base64_images

# Take screenshot of the first 2-3 pages to understand the type of content
def take_initial_screenshots(pdf_bytes: bytes, num_of_screenshots: int):
    images = pdf_bytes_to_base64_images(pdf_bytes, num_of_screenshots)
    save_base64_images_to_disk(images)
    return images

extractor_llm = ChatOpenAI(model="gpt-4o", model_kwargs={"tool_choice": "required"}).bind_tools([parse_text, extract_tables, ocr])
# extractor_agent = create_react_agent(extractor_llm,
#             [parse_text, extract_tables, ocr],
#             version="v1",
#             prompt="You are a multi modal agent that decides tool usage.")

def extractor_node(state: StructureState) -> LGCommand[str]:
    pdf_data = state.envelope.get("pdf")
    screenshots = take_initial_screenshots(pdf_data, 1)
    cfg = state.config

    input_prompt = f"""
    You are given one or more screenshots from a native PDF document as a visual reference only.These screenshots are provided to help you assess the type and structure of the actual PDF document, which may contain either selectable text or scanned images.

📌 Your task:
1. Determine whether the PDF pages are primarily composed of:
    * Selectable (native) text, or
    * Scanned images (e.g., photos of text, image-only pages)
        * Look for deformities in the picture that comes from the process of taking the picture, colour misalignment, bending of the pages and so on.
2. Based on this assessment, choose the appropriate tools to extract structured data for agentic ingestion.

🛠️ Tool Selection Rules
* ✅ If the document contains selectable text:→ Based on visual structure in the screenshot(s), choose from:
    * parse_text – for raw text extraction
    {('& extract_tables if tables are visually present' if cfg.look_for_queryable_tables else '')}
{("* Additionally, use as needed:"
    "* extract_images_from_pdf"
    "* take_screenshots" if cfg.extract_and_embed_images else '')}
* ✅ If the document contains scanned images:→ Use:
    * ocr – to extract text from images
    {('* take_screenshots – if needed for visual processing or context' if cfg.extract_and_embed_images else '')}
* (Do not use parse_text, extract_tables, or extract_images_from_pdf in this case.)
"""

    messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": input_prompt},
            *[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img}"
                    }
                }
                for img in screenshots
            ]
        ]
    }]

    result = extractor_llm.invoke(
        messages
    )

    tool_calls = result.tool_calls
    tool_outputs = {}

    tool_funcs = {
        "parse_text": parse_text,
        "ocr": ocr,
        "extract_tables": extract_tables,
        "extract_images_from_pdf": extract_images_from_pdf,
        "take_screenshots": take_screenshots,
    }

    for call in tool_calls:
        name = call["name"]
        args = call.get("args", {})

        # Always ensure the real bytes are passed, not the placeholder
        args["pdf_bytes"] = state.envelope.get('pdf')

        tool = tool_funcs.get(name)
        if not tool:
            continue

        try:
            out = tool.func(**args)
            tool_outputs[name] = out
        except Exception as e:
            tool_outputs[name] = {"error": str(e)}

    extracted = {
        "text":        tool_outputs.get("parse_text", []),
        "ocr":         tool_outputs.get("ocr", []),
        "tables":      tool_outputs.get("extract_tables", []),
        "images":      tool_outputs.get("extract_images_from_pdf", []),
        "screenshots": tool_outputs.get("take_screenshots", []),
    }

    steps = state.envelope.get("steps_completed", []) + ["extractor"]
    new_envelope = {
        **state.envelope.copy(),
        "extractor": extracted,
        "steps_completed": steps
    }
    return LGCommand(goto="orchestrator_node", update={"envelope": new_envelope})


# ────────────────────────────────────────────────
# SEGMENTER using trust_call
# ────────────────────────────────────────────────

class SegmenterNode(BaseModel):
    sections: list[str] = Field(..., description="The sections of the document.")

segmenter_llm = ChatOpenAI(model="gpt-3.5-turbo").with_structured_output(SegmenterNode)

def segmenter_node(state: StructureState) -> LGCommand[str]:
    """Segment the document into high-level sections based on extracted text and layout."""
    prompt = f"""
You are a document segmenter. Based on the extracted text and layout, split the content into high-level sections.
Text: {cap_text_length(state.envelope.get("extractor", {}).get("text", []))}
Layout: {cap_text_length(state.envelope.get("extractor", {}).get("layout", []))}
"""
    messages = [{"role": "user", "content": prompt}]
    response: AIMessage = segmenter_llm.invoke(
        messages
    )

    segment_text = response.dict()
    steps = state.envelope.get("steps_completed", []) + ["segmenter"]
    new_envelope = {
        **state.envelope.copy(),
        "segment_output": segment_text,
        "steps_completed": steps
    }
    return LGCommand(goto="orchestrator_node", update={"envelope": new_envelope})

# ────────────────────────────────────────────────
# SEMANTIC NODE using trust_call
# ────────────────────────────────────────────────

class SemanticNode(BaseModel):
    entities: List[str]
    summary: Optional[List[str]] = None


semantic_llm = ChatOpenAI(model="gpt-3.5-turbo").with_structured_output(SemanticNode)

def semantic_node(state: StructureState) -> LGCommand[str]:
    """Summarize sections and extract entities from the document."""
    print('Segment output is', state.envelope.get("segment_output", {}))
    prompt = f"""
Summarize each section and extract entities.
Sections: {cap_text_length_raw(state.envelope.get("segment_output", {}).get("sections", []))}
Text: {cap_text_length(state.envelope.get("extractor", {}).get("text", []))}
"""
    messages = [{"role": "user", "content": prompt}]
    response = semantic_llm.invoke(
        messages
    )
    semantic_text = response.dict()
    steps = state.envelope.get("steps_completed", []) + ["semantic"]
    new_envelope = {
        **state.envelope.copy(),
        "semantic_output": semantic_text,
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
Entities: {cap_text_length_raw(state.envelope.get("semantic_output", {}).get("entities", []))}
Summaries: {(lambda s: cap_text_length_raw(s) if s else None)(state.envelope.get("semantic_output", {}).get("summary"))}
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
            "text": envelope.get("extractor").get("text"),
            "tables": envelope.get("extractor").get("tables"),
            "sections": envelope.get("segment_output").get("sections"),
            "summary": envelope.get("semantic_output").get("summary"),
            "entities": envelope.get("semantic_output").get("entities"),
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
