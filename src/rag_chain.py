from typing import Dict, List, Tuple
from langchain_core.documents import Document


def format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata
        blocks.append(
            f"[Source {i}] file={meta.get('source')} | page={meta.get('page_number')} | "
            f"description={meta.get('page_description')}\n{d.page_content}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, docs: List[Document]) -> str:
    context = format_context(docs)
    return f"""
Use ONLY the context below to answer the question.
If the answer is not present in the context, say: "I don't know based on the uploaded documents."

Context:
{context}

Question:
{question}

Return:
1. Direct answer
2. Evidence
3. Sources with file name and page number
"""


def answer_question(llm, retriever, question: str) -> Tuple[str, List[Document]]:
    docs = retriever.invoke(question)
    prompt = build_prompt(question, docs)
    answer = llm.invoke(prompt)
    return answer, docs
