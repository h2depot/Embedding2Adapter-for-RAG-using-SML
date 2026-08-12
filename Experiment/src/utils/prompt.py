FINAL_PROMPT_TEMPLATE = """Answer the question using only the provided context.

You may combine information from multiple parts of the context.
Do not use knowledge that is not in the context.

If the context is insufficient to determine the answer, output exactly:
no

Output only the shortest final answer.

<context>
{context}
</context>

<question>
{question}
</question>

Answer:

"""

def build_final_prompt(context: str, question: str) -> str:
    return FINAL_PROMPT_TEMPLATE.format(context=context, question=question)
