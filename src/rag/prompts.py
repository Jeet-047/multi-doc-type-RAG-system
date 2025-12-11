"""
Prompt templates for RAG answering with strict grounding and citations.
"""

from langchain_core.prompts import PromptTemplate

# Core system safety + grounding rules
SYSTEM_PROMPT = """
You are a careful assistant that must answer ONLY with the provided context.
If the answer is not fully supported by the context, reply: "I don't know."
Always include citations in the form (source: {source}, page {page}).
Do not fabricate information or sources.
""".strip()


def build_stuff_prompt() -> PromptTemplate:
    """Prompt for the simple Stuff strategy (small context)."""
    template = """
{system_prompt}

Context:
{context}

Question: {question}

Answer using only the context above. Include citations after every claim.
If the context is insufficient, say "I don't know."
""".strip()
    return PromptTemplate(
        input_variables=["system_prompt", "context", "question"], template=template
    )


def build_map_prompt() -> PromptTemplate:
    """Prompt for the map step of Map-Reduce."""
    template = """
{system_prompt}

Context chunk:
{context}

Question: {question}

Extract only the information that answers the question from this chunk.
Return concise bullet points with citations.
If nothing is relevant, return "No relevant information.".
""".strip()
    return PromptTemplate(
        input_variables=["system_prompt", "context", "question"], template=template
    )


def build_reduce_prompt() -> PromptTemplate:
    """Prompt for the reduce step of Map-Reduce."""
    template = """
{system_prompt}

You received the following partial answers (each may include citations):
{map_summaries}

Question: {question}

Combine the partial answers into a single, concise answer.
Keep only supported facts and preserve citations.
If the partial answers do not cover the question, say "I don't know."
""".strip()
    return PromptTemplate(
        input_variables=["system_prompt", "map_summaries", "question"],
        template=template,
    )

