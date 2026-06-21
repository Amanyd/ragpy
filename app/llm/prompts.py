
from llama_index.core.prompts.base import PromptTemplate


QA_PROMPT = PromptTemplate(
    template=(
        "You are a helpful, conversational AI teaching assistant.\n"
        "Use the provided context to answer the user's question accurately.\n"
        "If the user is just saying hello or making small talk, respond conversationally normally without mentioning the context.\n"
        "If the user asks a question and the context is insufficient to answer it, try your best to answer using your general intelligence.\n"
        "Cite sources by file name from the context metadata when using the context.\n"
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


CONDENSE_PROMPT = PromptTemplate(
    template=(
        "Given the following conversation history and a follow-up question, "
        "rewrite the follow-up question to be a standalone, self-contained question "
        "that can be understood without the conversation history.\n"
        "Do NOT answer the question. Only rewrite it.\n"
        "If the question is a greeting, irrelevant, or not a follow-up, return it exactly unchanged.\n"
        "NEVER add commentary like 'I cannot' or 'You don't have'. ONLY output the query string.\n"
        "\n"
        "Conversation History:\n"
        "{chat_history}\n"
        "\n"
        "Follow-up Question: {question}\n"
        "\n"
        "Standalone Question:"
    )
)



QUIZ_GENERATION_PROMPT = PromptTemplate(
    template=(
        "You generate quizzes from the provided context only.\n"
        "Do not use outside knowledge. If the context is insufficient, output an empty questions list.\n"
        "\n"
        "Difficulty level: {difficulty}\n"
        "- easy: Basic recall and definition questions. Straightforward single-concept answers.\n"
        "- medium: Application and understanding questions. May require connecting two concepts.\n"
        "- hard: Analysis and synthesis questions. Requires deep understanding and multi-step reasoning.\n"
        "\n"
        "Return valid JSON only (no markdown) that matches the expected schema exactly.\n"
        "\n"
        "Context:\n"
        "{context_str}\n"
        "\n"
        "Output JSON:\n"
    )
)

