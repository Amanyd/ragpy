
from llama_index.core.prompts.base import PromptTemplate


QA_PROMPT = PromptTemplate(
    template=(
        "You are a helpful assistant.\n"
        "Answer the user's question using only the provided context.\n"
        "If the context is insufficient, say \"I don't know.\".\n"
        "Cite sources by file name from the context metadata when available.\n"
        "\n"
        "Context:\n"
        "{context_str}\n"
        "\n"
        "Question:\n"
        "{query_str}\n"
        "\n"
        "Answer:\n"
    )
)


QUIZ_GENERATION_PROMPT = PromptTemplate(
    template=(
        "You generate quizzes from the provided context only.\n"
        "Do not use outside knowledge. If the context is insufficient, output an empty questions list.\n"
        "\n"
        "Return valid JSON only (no markdown) that matches the expected schema exactly.\n"
        "\n"
        "Context:\n"
        "{context_str}\n"
        "\n"
        "Output JSON:\n"
    )
)

