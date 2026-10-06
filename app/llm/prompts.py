
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
        "You are an expert instructor creating educational briefing material and examination questions for the topic:\n"
        "\"{topic_phrase}\"\n\n"
        "Based on the provided technical excerpts, teach this topic clearly and progressively using structured slides, and create multiple-choice questions for the question bank.\n\n"
        "1. SLIDES:\n"
        "- Generate as many slides as needed to thoroughly and clearly teach this topic .\n"
        "- Do NOT use rigid or pre-set slide names. Choose natural, informative titles for each slide based on what it teaches.\n"
        "- For each slide, provide:\n"
        "  - 'title': A descriptive heading for what this slide teaches.\n"
        "  - 'bullets': 2 to 4 clear, instructive bullet points explaining the core concepts.\n"
        "  - 'formula_or_rule': (Optional) A governing formula or calculation rule if applicable, else null.\n"
        "  - 'diagram_mermaid': (Optional) A clean Mermaid.js diagram ('graph LR' or 'graph TD') if a diagram helps visualize the concept, else null.\n"
        "  - 'warning': (Optional) An important note, operating limit, or key takeaway, else null.\n\n"
        "2. EXAMINATION QUESTIONS (Question Bank):\n"
        "- Generate exactly 6 multiple-choice questions testing this topic across Bloom's Taxonomy:\n"
        "  - 2 EASY questions (difficulty: \"easy\"): Direct recall, definitions, standard terminology.\n"
        "  - 2 MEDIUM questions (difficulty: \"medium\"): Parameter variations, formula calculations, cause-and-effect relationships.\n"
        "  - 2 HARD questions (difficulty: \"hard\"): Complex scenarios, multi-step reasoning, analytical problem-solving.\n"
        "- Each question must be MCQ with exactly 4 choices labeled 'A', 'B', 'C', 'D' and one correct answer.\n\n"
        "CRITICAL: Output strictly valid JSON matching this structure with top-level 'slides' and 'questions':\n"
        "{{\n"
        '  "slides": [\n'
        '    {{\n'
        '      "title": "Descriptive Slide Title",\n'
        '      "bullets": ["Clear explanation point 1", "Clear explanation point 2"],\n'
        '      "formula_or_rule": null,\n'
        '      "diagram_mermaid": null,\n'
        '      "warning": null\n'
        '    }}\n'
        '  ],\n'
        '  "questions": [\n'
        '    {{\n'
        '      "type": "mcq",\n'
        '      "question": "Question text testing the topic?",\n'
        '      "choices": [\n'
        '        {{"label": "A", "text": "Option A"}},\n'
        '        {{"label": "B", "text": "Option B"}},\n'
        '        {{"label": "C", "text": "Option C"}},\n'
        '        {{"label": "D", "text": "Option D"}}\n'
        '      ],\n'
        '      "answer": "A",\n'
        '      "explanation": "Technical explanation citing aerodynamic/physical principles.",\n'
        '      "difficulty": "easy"\n'
        '    }}\n'
        '  ]\n'
        "}}\n\n"
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

