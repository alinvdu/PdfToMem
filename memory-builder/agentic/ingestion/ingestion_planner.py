from langgraph_reflection import create_reflection_graph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command as LGCommand
from langchain.chat_models import ChatOpenAI
from openevals.llm import create_llm_as_judge
from typing import Dict, Any, Optional, TypedDict
from configs import MCPConfig
from pydantic import BaseModel
from typing import List
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

### Reflection utils, assistant
# Define the main assistant model that will generate responses
llm_model = ChatOpenAI(model="gpt-3.5-turbo")

def call_model(state):
    """Process the user query with a large language model."""
    return {"messages": llm_model.invoke(state["messages"])}

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
# LLM SETUP
# ────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-3.5-turbo")

# ────────────────────────────────────────────────
# CHUNKING AGENT (Reflection)
# ────────────────────────────────────────────────

def chunking_assistant_node(state: MessagesState):
    """Generate chunking strategy for document indexing."""
    return {"messages": llm.invoke(state["messages"])}

def get_chunking_prompt(envelope):
    return f"""You are a Chunking Strategy Agent.

    Analyze the document structure below and propose a LlamaIndex indexing strategy.

    Examples:
    - Use HierarchicalNodeParser + AutoMerging if the document has deeply nested headers.
    - Use SentenceWindowNodeParser for flat, continuous prose.
    - Use SemanticSplitterNodeParser for concept-level segmentation.

    Return a JSON like:
    [{{
    "strategy": "hierarchical",
    "node_parser": "HierarchicalNodeParser",
    "index_type": "VectorStoreIndex",
    "query_node_description": "This hierarchical query tool can be used to extract hierarchical information from the data",
    "params": {{
        "leaf_parser": "SentenceWindowNodeParser",
        "window_size": 5,
        "storage_context": true
    }},
    "reasoning": "Clear section headers indicate a hierarchical layout."
    }}] where multiple strategies returned in a list

    Document Structure:
    {envelope}
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

def chunking_node(state):
    prompt = get_chunking_prompt(state.envelope)
    result = reflection_chunking.invoke({"messages": prompt})

    ai_message = next(
        (msg for msg in result["messages"] if isinstance(msg, AIMessage)),
        None
    )

    if not ai_message:
        ai_message = {
            'content': 'No plan'
        }
    plan_str = ai_message.content.strip()

    print("✅ Parsed chunking plan:", plan_str)
    new_planner_steps = (state.planner_steps or []) + ['chunking']

    # update the state for chunking
    return LGCommand(
        goto="planner_orchestrator",
        update={"chunking_plan": plan_str, "planner_steps": new_planner_steps}
    )

# ────────────────────────────────────────────────
# KNOWLEDGE GRAPH AGENT (Reflection)
# ────────────────────────────────────────────────

def kg_assistant_node(state: MessagesState):
    """Generate knowledge graph indexing strategy based on entities and relationships."""
    return {"messages": llm.invoke(state["messages"])}

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
    return {"messages": llm.invoke(state["messages"])}

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
