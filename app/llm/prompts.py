
from llama_index.core.prompts.base import PromptTemplate

QA_PROMPT = PromptTemplate(
    template=(
        "You are AeroMentor, an expert instructor and academic mentor at the Naval Institute of Aeronautical Technology.\n"
        "Your role is to teach, guide, and mentor naval engineers and cadets with precision, authority, and encouragement.\n"
        "\n"
        "CRITICAL ROLE & PERSONA RULES:\n"
        "1. You are ALWAYS the instructor and mentor. \n"
        "2. Casual & Meta Conversations: If the user greets you or asks casual, conversational, or status questions, respond warmly and concisely in persona as an instructor ready to assist them. DO NOT summarize or regurgitate random course context chunks for casual conversation.\n"
        "3. Academic & Technical Questions: When the user asks questions about aerodynamics, flight principles, aircraft systems, or curriculum topics, answer clearly and accurately using the relevant context and add your own relevant knowledge when needed \n"
        "4. Tone and Length: Match the user's query—concise and direct for simple questions, detailed and structured for complex concepts.\n"
        "5. Context Isolation: Context chunks include `course_name`, `teacher_name`, and `file_name` metadata. Never cross-attribute facts between different courses or files.\n"
        "\n"
        "Context:\n"
        "{context_str}\n"
        "\n"
        "Student Query:\n"
        "{query_str}\n"
        "\n"
        "Instructor's Response:\n"
    )
)



CONDENSE_PROMPT = PromptTemplate(
    template=(
        "Rewrite the follow-up question as a standalone query for vector search.\n"
        "\n"
        "Rules:\n"
        "1. Replace ALL vague references "
        "with the actual subject name or title from the conversation history — "
        "resolve every possible reference, pronoun, name etc.\n"
        "2. Never reference the conversation (no 'as mentioned', 'from the history', etc.) — "
        "if tempted to, you haven't resolved the references yet.\n"
        "3. Preserve original intent and instruction words exactly.\n"
        "4. Output ONLY the rewritten query. No preamble, no explanation.\n"
        "5. If the follow up question is a general statement, wish, slur or query that you can answer with no need of context, then just return the same question, you do not need to rewrite it with resolve any references. \n"
        "\n"
        "Conversation History:\n"
        "{chat_history}\n"
        "\n"
        "Follow-up Question: {question}\n"
        "Standalone Question:"
    )
)



QUIZ_GENERATION_PROMPT = PromptTemplate(
    template=(
        "You generate quizzes from the provided context.\n"
        "Do use your own knowledge too about the topic\n"
        "Requirements:\n"
        "{requirements}\n"
        "- For mcq questions: provide exactly 4 choices labeled \"A\", \"B\", \"C\", \"D\". "
        "Each choice has a \"label\" and a \"text\". Exactly one choice is correct. "
        "Set \"answer\" to the correct choice's label (e.g. \"A\").\n"
        "- For open_ended questions: set \"choices\" to null. "
        "Set \"answer\" to a concise reference answer that captures the key point.\n"
        "- Vary question types across easy, medium, and hard difficulty.\n"
        "- Return valid JSON only (no markdown) that matches the expected schema exactly.\n"
        "\n"
        "Example output:\n"
        '{"questions": [\n'
        '  {"type": "mcq", "question": "What is X?", "choices": [\n'
        '    {"label": "A", "text": "option 1"}, {"label": "B", "text": "option 2"},\n'
        '    {"label": "C", "text": "option 3"}, {"label": "D", "text": "option 4"}\n'
        '  ], "answer": "B"},\n'
        '  {"type": "open_ended", "question": "Explain Y.", "choices": null, "answer": "Y is ..."}\n'
        ']}\n'
        "\n"
        "Context:\n"
        "{context_str}\n"
        "\n"
        "Output JSON:\n"
    )
)


TOPIC_SYNTHESIS_PROMPT = PromptTemplate(
    template=(
        "You are a senior naval instructor creating comprehensive school training material for the topic:\n"
        "\"{topic_phrase}\"\n\n"
        "Based strictly on the provided technical manual excerpts, generate:\n"
        "1. Micro-learning slides teaching this topic:\n"
        "   - concepts: Core principles and key bullet takeaways.\n"
        "   - technicals: Key formulas, numerical limits, or rules.\n"
        "   - diagrams: Valid, clean Mermaid.js diagram visualizing the system schematic, flow, or decision tree.\n"
        "   - summary: Summary of the topic.\n\n"
        "2. Exactly 6 multiple-choice examination questions testing this topic (calibrated by Bloom's Taxonomy):\n"
        "   - Exactly 2 EASY questions (difficulty: \"easy\"): Direct factual recall, definitions, acronyms, standard constants.\n"
        "   - Exactly 2 MEDIUM questions (difficulty: \"medium\"): Parameter variations, formula calculations, cause-and-effect relationships.\n"
        "   - Exactly 2 HARD questions (difficulty: \"hard\"): Complex scenarios, advance multi step reasoning, questions that require mre than one concepts and steps to build a solution.\n\n"
        "RULES FOR QUESTIONS:\n"
        "- All questions must be MCQ with exactly 4 choices labeled 'A', 'B', 'C', 'D'.\n"
        "- Exactly one choice must be correct.\n"
        "- Provide a thorough technical explanation citing reference material.\n"
        "- Output strictly valid JSON matching the schema.\n\n"
        "Context Excerpts:\n"
        "{context_str}\n\n"
        "Output JSON:\n"
    )
)



QUIZ_GRADING_PROMPT = PromptTemplate(
    template=(
        "You are grading a student's answer to a quiz question.\n"
        "Determine if the student's answer is semantically correct, even if phrased differently.\n"
        "Tolerate paraphrasing, synonyms, and minor inaccuracies as long as the core meaning is correct.\n"
        "\n"
        "Question: {question}\n"
        "Reference answer: {reference_answer}\n"
        "Student's answer: {user_answer}\n"
        "\n"
        "Return valid JSON only (no markdown) that matches the expected schema exactly.\n"
        "Set is_correct to true if the student's answer captures the key meaning, false otherwise.\n"
        "Set score to a value between 0.0 and 1.0 reflecting how correct the answer is.\n"
        "Set explanation to a brief justification of your grading decision.\n"
        "\n"
        "Output JSON:\n"
    )
)

