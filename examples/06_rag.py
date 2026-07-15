"""
Example 06: RAG (Retrieval-Augmented Generation)
Demonstrates vector search with embeddings and retrieval nodes.
"""
from graphforge import Graph, GraphState, configure_logging
from graphforge.rag import InMemoryVectorStore, RetrievalNode, chunk_text
from graphforge.rag._embeddings import DeterministicEmbeddings

configure_logging()


class RagState(GraphState):
    question: str = ""
    context: list = []
    answer: str = ""


# Step 1: Create a knowledge base
documents = [
    "GraphForge is a type-safe graph execution framework for LLM applications.",
    "It supports MCP, A2A, and custom tool integration out of the box.",
    "The framework uses Pydantic v2 for state management with explicit merge strategies.",
    "Nodes can be async functions, generators, subgraphs, or pipelines.",
    "Checkpointing supports InMemory, SQLite, Redis, and PostgreSQL backends.",
    "GraphForge includes built-in ReAct agents, multi-agent patterns, and guardrails.",
    "The AutoOptimizer detects independent execution paths and suggests parallelization.",
    "Store provides cross-session persistent memory independent of checkpoints.",
]

# Step 2: Create embeddings and vector store
embeddings = DeterministicEmbeddings(dimension=64)
store = InMemoryVectorStore(embeddings)

# Chunk and add documents
for doc in documents:
    chunks = chunk_text(doc, strategy="fixed", chunk_size=100, chunk_overlap=0)
    store.add_texts(chunks)

print(f"Added {len(documents)} documents to vector store")


# Step 3: Create retrieval node
def generate(state: RagState) -> dict:
    """Generate an answer based on retrieved context."""
    context = state.context
    if not context:
        return {"answer": "No relevant information found."}
    sources = [c["content"] for c in context]
    return {"answer": f"Based on retrieved information:\n" + "\n".join(f"- {s}" for s in sources)}


# Step 4: Build graph
graph = Graph[RagState]()
graph.add_node("retrieve", RetrievalNode(store, query_field="question", output_field="context"))
graph.add_node("generate", generate)
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "__end__")
graph.set_entry_point("retrieve")

compiled = graph.compile()

# Step 5: Test with questions
for question in ["What is GraphForge?", "What databases does checkpointing support?", "What is AutoOptimizer?"]:
    result = compiled.invoke(RagState(question=question))
    print(f"\nQ: {question}")
    print(f"  Retrieved {len(result.context)} documents")
    for c in result.context:
        print(f"    [{c['score']:.2f}] {c['content'][:60]}...")
    print(f"  Answer: {result.answer[:120]}...")
