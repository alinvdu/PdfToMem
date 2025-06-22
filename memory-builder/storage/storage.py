from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import traceback
from llama_index.core import (
    VectorStoreIndex
)

from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceWindowNodeParser, SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.query_engine import RouterQueryEngine, RetrieverQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.schema import TextNode
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Document
from typing import Any
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever
import logging
import llama_index
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global router engine instance
router_engine: RouterQueryEngine = None
router_tools: List[QueryEngineTool] = []

class Extractor(BaseModel):
    text: List = []
    ocr: List = []
    tables: List = []
    images: List = []
    screenshots: List = []

class StructureEnvelope(BaseModel):
    extractor: Extractor
    steps_completed: List[str]
    segment_output: Any
    semantic_output: Any
    relationships: str
    final_structured_output: Any

class IngestionPlanParams(BaseModel):
    window_size: Optional[int] = None

class IngestionPlanItem(BaseModel):
    strategy: str
    query_node_description: str
    params: Optional[IngestionPlanParams] = None
    reasoning: str

class RequestBody(BaseModel):
    structure_envelope: StructureEnvelope
    ingestion_plan: List[IngestionPlanItem]

Settings.llm = OpenAI(model="gpt-3.5-turbo")
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
Settings.embed_model = embed_model

def build_sentence_index(document, chunk_size="512", save_dir="simple_sentence_index"):
    splitter = SentenceSplitter(chunk_size)
    nodes = splitter.get_nodes_from_documents([document])
    print(len(nodes))
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=save_dir)

    return index

def get_sentence_query_engine(index, similarity_top_k=6):
    return index.as_query_engine(similarity_top_k=similarity_top_k)

def build_sentence_window_index(document, window_size = 3,
                                save_dir="sentence_index"):
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )

    Settings.node_parser = node_parser
    
    sentence_index = VectorStoreIndex.from_documents(
        [document]
    )
    sentence_index.storage_context.persist(persist_dir=save_dir)

    return sentence_index

def get_sentence_window_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model="BAAI/bge-reranker-base")
    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postproc, rerank],
        verbose=True
    )
    
    return sentence_window_engine

from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever

def build_automerging_index(documents,
                            save_dir="merging_index", chunk_sizes=None):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    Settings.node_parser = node_parser
    
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(docstore=docstore)
    merging_index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context
    )
    merging_index.storage_context.persist(persist_dir=save_dir)
        
    return merging_index

def get_automerging_query_engine(automerging_index, similarity_top_k=12,
                                 rerank_top_n=2):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever,
        automerging_index.storage_context,
        verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine(
        retriever,
        node_postprocessors=[rerank]
    )
    
    return auto_merging_engine

def build_semantic_index(documents):
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)

    return index

def get_semantic_query_engine(index, similarity_top_k=12):
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)

    return query_engine


@app.post("/storage_ingest")
async def storage_ingest(request: RequestBody):
    global router_engine
    global router_tools

    logging.info("Starting ingestion")

    envelope = request.structure_envelope
    text_chunks = [entry['text'] for entry in envelope.extractor.text]
    document = Document(text="\n".join(text_chunks))

    ingestion_plan = request.ingestion_plan
    query_tools = []

    indices = []

    for plan in ingestion_plan:
        strategy = plan.strategy
        params = plan.params or {}
        index = None
        query_engine = None

        print('params are', params)

        if strategy == "SimpleSentenceIndex":
            # the most basic representation:
            index = build_sentence_index(document, save_dir="simple_sentence_index")
            query_engine = get_sentence_query_engine(index)
        elif strategy == "SentenceWindowIndex":
            window_size = 3
            if 'window_size' in params:
                window_size = params.get('window_size')
            index = build_sentence_window_index(document, window_size, save_dir="sentence_index")
            query_engine = get_sentence_window_query_engine(index)
        elif strategy == "AutoMergingIndex":
            chunk_sizes = [2048, 512, 128]
            if 'chunk_sizes' in params:
                chunk_sizes = params.get('chunk_sizes')
            index = build_automerging_index(
                [document],
                chunk_sizes=chunk_sizes,
                save_dir="merging_index",
            )
            query_engine = get_automerging_query_engine(index)
        elif strategy == "SemanticIndex":
            index = build_semantic_index([document])
            query_engine = get_semantic_query_engine(index)
        else:
            raise ValueError(f"Unsupported parser: {strategy}")

        tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name=strategy,
            description=plan.query_node_description
        )
        query_tools.append(tool)

    router_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=query_tools,
        verbose=True
    )

    router_tools = query_tools

    print('All the query tools requested', query_tools)

    tools_info = [
        {
            "name": t.metadata.name,
            "description": t.metadata.description
        }
        for t in query_tools
    ]

    return {"status": "success", "tools_initialized": len(query_tools), "tools": tools_info}

class QueryRequest(BaseModel):
    query: str

# @app.post("/query_router")
# async def query_router(request: QueryRequest):
#     global router_engine

#     if router_engine is None:
#         return {"error": "Router engine not initialized. Please ingest data first."}

#     query = request.query
#     try:
#         response = router_engine.query(query)
#         print('response selection is', response.metadata["selector_result"])
#         print('chunks are', response.source_nodes)
#         print('response is', str(response))
#         return {"query": query, "response": str(response)}
#     except Exception as e:
#         traceback.print_exc()
#         return {"error": str(e)}

@app.post("/query_router")
async def query_router(request: QueryRequest):
    global router_engine, router_tools

    if router_engine is None:
        return {"error": "Router engine not initialized. Please ingest data first."}

    raw_resp = router_engine.query(request.query)
    resp_obj = raw_resp

    sel = resp_obj.metadata.get("selector_result")
    if hasattr(sel, "selections"):
        indices = [s.index for s in sel.selections]
    else:
        indices = getattr(sel, "inds", [getattr(sel, "ind")])
    selected_queries = [router_tools[i].metadata.name for i in indices]
    chunks = [node.text for node in resp_obj.source_nodes]
    answer = resp_obj.response

    return {
        "response": answer,
        "selected_queries": selected_queries,
        "chunks": chunks,
    }