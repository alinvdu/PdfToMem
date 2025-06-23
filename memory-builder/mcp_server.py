from fastmcp import FastMCP
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64

# Load environment variables
load_dotenv()

# Your existing configuration and function definitions
from configs import MCPConfig
from agentic.ingestion.planners import run_ingestion_pipeline

mcp = FastMCP("pdf_ingest", stateless_http=True)

@mcp.tool()
async def determine_memory_architecture(pdf: str, extract_and_embed_images: bool, look_for_queryable_tables: bool) -> dict:
    """Determine the architecture for PDF ingestion.
    This function kicks off an multi-agent pipeline to determine the best architecture for ingesting a PDF file and storing it.
    It returns an envelope with the metadata needed for ingestion such as: text_chunks, tables, images, screenshots, OCR text, hierarchy graph
    All build smartly by the agents based on the PDF content.

    It also returns a strategy that can be used for ingestion and memory persistence based on Llama-Index Query Engines.

    Args:
        pdf (str): The PDF file content as base64 encoded string.
        extract_and_embed_images (bool): Whether to extract and embed images from the PDF.
        look_for_queryable_tables (bool): Whether to look for queryable tables in the PDF.
    """
    cfg = MCPConfig(
        extract_and_embed_images=extract_and_embed_images,
        look_for_queryable_tables=look_for_queryable_tables
    )

    if not pdf or not isinstance(pdf, str):
        raise ValueError("Expected a base64-encoded PDF string for pdf")

    try:
        # Decode base64 string to bytes
        pdf_bytes = base64.b64decode(pdf)
    except Exception as e:
        raise ValueError("Invalid base64 PDF string") from e


    ingestion = run_ingestion_pipeline(pdf_bytes, cfg)

    structure_envelope = ingestion['structure_envelope']
    keys_to_remove = ['pdf', 'screenshots']
    new_envelope = {k: v for k, v in structure_envelope.items() if k not in keys_to_remove}


    return {
        "structure_envelope": new_envelope,
        "ingestion_plan": ingestion['ingestion_plan'],
    }

# Get the underlying ASGI app from FastMCP
app = mcp.http_app()

# Define the allowed origins for CORS
# For development, you might use a wildcard, but for production,
# it's recommended to list your specific frontend domains.
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    # Add the origin of your web application here
]

# Add the CORS middleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

if __name__ == "__main__":
    # Run the application using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)