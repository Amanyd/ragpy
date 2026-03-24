RAG_INGEST_STREAM = "RAG_INGEST"
RAG_QUIZ_STREAM = "RAG_QUIZ"

RAG_INGEST_SUBJECTS = ["rag.ingest.request"]
RAG_INGEST_DONE_STREAM = "RAG_INGEST_DONE"
RAG_INGEST_DONE_SUBJECTS = ["rag.ingest.done"]
RAG_QUIZ_SUBJECTS = ["quiz.generate.>"]

RAG_INGEST_PUBLISH_SUBJECT = "rag.ingest.request"
RAG_INGEST_DONE_SUBJECT = "rag.ingest.done"
RAG_QUIZ_PUBLISH_SUBJECT = "quiz.generate.request"

DURABLE_INGEST_WORKER = "rag-ingest"
DURABLE_QUIZ_WORKER = "rag-quiz"

QUEUE_GROUP_INGEST = "rag-ingest"
QUEUE_GROUP_QUIZ = "rag-quiz"

