from langgraph_reflection import create_reflection_graph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command as LGCommand
from langchain_openai import ChatOpenAI
from openevals.llm import create_llm_as_judge
from typing import Dict, Any, Optional, TypedDict, List, Literal, Union
from configs import MCPConfig
from pydantic import BaseModel, RootModel
from langchain_core.messages import AIMessage


# ────────────────────────────────────────────────
# STATE DEFINITION
# ────────────────────────────────────────────────

class PlannerState(BaseModel):
    envelope: dict
    config: MCPConfig
    planner_steps: Optional[List] = None
    chunking_plan: Optional[str] = None
    kg_plan: Optional[str] = None
    visual_plan: Optional[str] = None
    plan: Optional[str] = None

class SentenceWindowIndexParams(BaseModel):
    window_size: int

class SimpleSentenceIndexParams(BaseModel):
    chunk_size: int

class AutoMergingIndexParams(BaseModel):
    chunk_sizes: List[int]

class SemanticIndexParams(BaseModel):
    pass  # No params for this one

class StrategyOutput(BaseModel):
    strategy: Literal[
        "SimpleSentenceIndex",
        "SentenceWindowIndex",
        "AutoMergingIndex",
        "SemanticIndex"
    ]
    query_node_description: str
    params: Union[
        SentenceWindowIndexParams,
        SimpleSentenceIndexParams,
        AutoMergingIndexParams,
        SemanticIndexParams
    ]
    reasoning: str

class ChunkingStrategies(BaseModel):
    strategies: List[StrategyOutput]

### Reflection utils, assistant
# Define the main assistant model that will generate responses
llm_model = ChatOpenAI(model="o3-mini").with_structured_output(ChunkingStrategies)

def call_model(state):
    """Process the user query with a large language model."""
    response = llm_model.invoke(state["messages"]).dict()
    return {"messages": {
        "role": "assistant",
        "content": json.dumps(response)}}

assistant_graph = (
    StateGraph(MessagesState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .add_edge("call_model", END)
    .compile()
)

# Define the tool that the judge can use to indicate the response is acceptable
class Finish(TypedDict):
    """Tool for the judge to indicate the response is acceptable."""
    finish: bool

# ────────────────────────────────────────────────
# CHUNKING AGENT (Reflection)
# ────────────────────────────────────────────────

def chunking_assistant_node(state: MessagesState):
    """Generate chunking strategy for document indexing."""
    return {"messages": llm_model.invoke(state["messages"])}

def truncate_long_values(obj, char_limit=3000):
    """Recursively truncate string values in a dictionary or list."""
    if isinstance(obj, dict):
        return {
            k: truncate_long_values(v, char_limit)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [truncate_long_values(item, char_limit) for item in obj]
    elif isinstance(obj, str):
        return (obj[:char_limit] + "...[truncated]") if len(obj) > char_limit else obj
    else:
        return obj

def get_chunking_prompt(envelope, char_limit=3000):
    keys_to_remove = ['pdf', 'screenshots']
    # Filter out specific keys
    filtered_envelope = {k: v for k, v in envelope.items() if k not in keys_to_remove}
    # Recursively truncate long values
    truncated_envelope = truncate_long_values(filtered_envelope, char_limit)

    document_text = "\n".join(chunk.get("text", "") for chunk in truncated_envelope.get('text'))

    return f"""You are a Chunking Strategy Agent.

    Analyze the document structure below and propose a LlamaIndex ingestion strategy for this particular type of document.
    Part of your strategy you have to define the indices you are going to use and to give a name for the query so then when
    the index is going to be used by an LLM, it knows what it does. The following indices are available including their parameters:
    1. SimpleSentenceIndex -> Text is chunked on multiple smaller chunks, one or more texts are retrieved but context around is lost.
    This doesn't split context by sentences but by the specified chunk amount.
        - chunk_size: int
    2. SentenceWindowIndex -> Just like SimpleSentenceIndex but it adds a window around each chunk. Ideal for answering questions that depend on 
    relationships between adjacent concepts. The window provides an opportunity to look around the found chunk.
        - window_size: int
    2. AutoMergingIndex -> Automatically merges smaller semantic units into meaningful chunks depending on context. Useful for high-density documents like 
        specifications and APIs where the response might include adjacent concepts part of other chunks.
        - chunk_sizes: list[int], for instance [2048, 512, 128]
    3. SemanticIndex -> Divides the text into higher level semantics and splits according to ideas rather than fixed sizes.

    Here is a minimal example:
    Return a JSON like:
    [{{
    "strategy": "SentenceWindowIndex",
    "query_node_description": "This will split the document on sentences with overlapp specified by the window size",
    "params": {{
        "window_size": 3
    }},
    "reasoning": "Propose text with concepts that span accross but not too much and its not hierarchical, therefore sentence window index is a perfect candidate"
    }}] where multiple strategies returned in a list

    Document Structure (this include the actual text + other metadata):
    {document_text}

    Guidelines:
    - Reason through which concepts make sense to the given document information.
    """

chunking_judge_prompt = """You are a strategy reviewer. Here's the proposed chunking/indexing plan:

<response>
{outputs}
</response>

Evaluate:
- Is it appropriate for the document structure?
- Does it leverage LlamaIndex properly?
- Can it be improved?

If all criteria are met, set pass to True.
Otherwise, set pass to False and include constructive feedback.
"""

max_chunking_judge_reflections = 5
chunking_iter = 0

def chunking_judge_node(state: MessagesState, config=None):
    global chunking_iter
    """Evaluate and provide feedback on chunking strategy proposals."""
    evaluator = create_llm_as_judge(
        prompt=chunking_judge_prompt,
        model="gpt-3.5-turbo",
        feedback_key="pass"
    )
    eval_result = evaluator(outputs=state["messages"][-1].content, inputs=None)
    print('eval score is', eval_result['score'])
    chunking_iter += 1

    if eval_result["score"] or chunking_iter >= max_chunking_judge_reflections:
        if chunking_iter >= max_chunking_judge_reflections:
            print("Stopped because of max iterations! Still not judge approval!")
        else:
            print("✅ Response approved by judge")
        return
    else:
        print("⚠️ Judge requested improvements")
        return {"messages": [{"role": "user", "content": eval_result["comment"]}]}
    
chunking_judge_graph = (
    StateGraph(MessagesState)
    .add_node(chunking_judge_node)
    .add_edge(START, "chunking_judge_node")
    .add_edge("chunking_judge_node", END)
    .compile()
)

reflection_chunking = create_reflection_graph(assistant_graph, chunking_judge_graph)
reflection_chunking = reflection_chunking.compile()

import json

def chunking_node(state):
    prompt = get_chunking_prompt(state.envelope.get('final_structured_output'))
    result = reflection_chunking.invoke({"messages": prompt})
    print('result is', result)
    ai_message = next(
        (msg for msg in result["messages"] if isinstance(msg, AIMessage)),
        None
    )

    if not ai_message:
        ai_message = {
            'content': 'No plan'
        }

    # Safely parse the content as JSON
    try:
        plan_data = json.loads(ai_message.content.strip())
        strategies = plan_data.get("strategies", [])
    except json.JSONDecodeError:
        strategies = []
        plan_data = {"strategies": []}

    print("✅ Parsed chunking plan:", strategies)
    new_planner_steps = (state.planner_steps or []) + ['chunking']

    # update the state for chunking
    return LGCommand(
        goto="planner_orchestrator",
        update={"chunking_plan": json.dumps(strategies), "planner_steps": new_planner_steps}
    )

# ────────────────────────────────────────────────
# KNOWLEDGE GRAPH AGENT (Reflection)
# ────────────────────────────────────────────────

def kg_assistant_node(state: MessagesState):
    """Generate knowledge graph indexing strategy based on entities and relationships."""
    return {"messages": llm_model.invoke(state["messages"])}

kg_prompt = """You are planning a knowledge graph-based indexing strategy based on document entities/relationships.

Propose a JSON like:
{
  "index_type": "KnowledgeGraphIndex",
  "kg_nodes": [...],
  "kg_edges": [...],
  "reasoning": "Useful for retrieving relational information from structured entities."
}

Entities:
{entities}

Relationships:
{relationships}
"""

kg_judge_prompt = """You're reviewing a Knowledge Graph strategy:

<response>
{outputs}
</response>

Is this the best representation given the entity and relationship data?
If yes, pass = True.
If not, pass = False and suggest improvements.
"""

def kg_judge_node(state: MessagesState, config=None):
    """Evaluate and provide feedback on knowledge graph strategy proposals."""
    evaluator = create_llm_as_judge(
        prompt=kg_judge_prompt,
        model="gpt-3.5-turbo",
        feedback_key="pass"
    )
    eval_result = evaluator(outputs=state["messages"][-1].content, inputs=None)
    if eval_result["score"]:
        return
    else:
        return {"messages": [{"role": "user", "content": eval_result["comment"]}]}

kg_graph = create_reflection_graph(
    StateGraph(MessagesState).add_node(kg_assistant_node).add_edge(START, "kg_assistant_node").add_edge("kg_assistant_node", END).compile(),
    StateGraph(MessagesState).add_node(kg_judge_node).add_edge(START, "kg_judge_node").add_edge("kg_judge_node", END).compile()
).compile()

# ────────────────────────────────────────────────
# VISUAL CHUNKING AGENT (Reflection)
# ────────────────────────────────────────────────

def visual_assistant_node(state: MessagesState):
    """Generate visual chunking strategy for documents with visual elements."""
    return {"messages": llm_model.invoke(state["messages"])}

visual_prompt = """The document includes visual elements. Propose a visual chunking/indexing strategy using screenshots, OCR, or CLIP embeddings.

Return:
{
  "visual_strategy": "screenshot + embed",
  "embedding_type": "CLIP",
  "index_type": "MultimodalVectorStoreIndex",
  "pages_affected": [1, 3],
  "reasoning": "Pages 1 and 3 contain figures that need retrieval."
}
"""

visual_judge_prompt = """Review the visual indexing strategy:

<response>
{outputs}
</response>

Improve if visual components are not well addressed. Set pass = True only if complete.
"""

def visual_judge_node(state: MessagesState, config=None):
    """Evaluate and provide feedback on visual chunking strategy proposals."""
    evaluator = create_llm_as_judge(
        prompt=visual_judge_prompt,
        model="gpt-3.5-turbo",
        feedback_key="pass"
    )
    eval_result = evaluator(outputs=state["messages"][-1].content, inputs=None)
    if eval_result["score"]:
        return
    else:
        return {"messages": [{"role": "user", "content": eval_result["comment"]}]}

visual_graph = create_reflection_graph(
    StateGraph(MessagesState).add_node(visual_assistant_node).add_edge(START, "visual_assistant_node").add_edge("visual_assistant_node", END).compile(),
    StateGraph(MessagesState).add_node(visual_judge_node).add_edge(START, "visual_judge_node").add_edge("visual_judge_node", END).compile()
).compile()

# ────────────────────────────────────────────────
# ORCHESTRATOR + FINAL AGGREGATOR
# ────────────────────────────────────────────────

def planner_orchestrator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Orchestrate the planning process by determining which strategy to generate next."""
    steps = state.planner_steps or []
    print('steps are', steps)

    envelope = (state.envelope or {})
    if "chunking" not in steps:
        next_node = "chunking_node"
    # elif "kg" not in steps and "entities" in envelope:
    #     return {"next": "kg_graph", "planner_steps": steps + ["kg"]}
    # elif "visual" not in steps and "images" in envelope:
    #     return {"next": "visual_graph", "planner_steps": steps + ["visual"]}
    else:
        next_node = "planner_aggregator"

    print('next node', next_node)
    
    return LGCommand(goto=next_node, update={})

def planner_aggregator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate all planning results into a final ingestion plan."""
    print("✅ Final ingestion plan ready.")
    return LGCommand(goto=END, update={"plan": state.chunking_plan})

# ────────────────────────────────────────────────
# BUILD REFLECTIVE PLANNER GRAPH
# ────────────────────────────────────────────────

def build_ingestion_planner_agents_graph() -> StateGraph:
    graph = StateGraph(PlannerState)

    graph.add_node("planner_orchestrator", planner_orchestrator_node)
    graph.add_node("planner_aggregator", planner_aggregator_node)
    graph.add_node("chunking_node", chunking_node)
    # graph.add_node("kg_graph", kg_graph)
    # graph.add_node("visual_graph", visual_graph)

    graph.set_entry_point("planner_orchestrator")
    graph.set_finish_point("planner_aggregator")

    return graph.compile()

def run_ingestion_planner_graph(envelope: dict, config: MCPConfig) -> dict:
    """Run the ingestion planner graph with proper initial state."""
    graph = build_ingestion_planner_agents_graph()
    
    initial_state = PlannerState(envelope=envelope, config=config)
    
    result = graph.invoke(initial_state)
    print('result is', result)
    return result['plan']
