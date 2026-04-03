RAG_INGEST_STREAM = "RAG_INGEST"
RAG_QUIZ_STREAM = "RAG_QUIZ"
RAG_QUIZ_DONE_STREAM = "RAG_QUIZ_DONE"

RAG_INGEST_SUBJECTS = ["rag.ingest.request"]
RAG_INGEST_DONE_STREAM = "RAG_INGEST_DONE"
RAG_INGEST_DONE_SUBJECTS = ["rag.ingest.done"]
RAG_QUIZ_SUBJECTS = ["quiz.generate.>"]
RAG_QUIZ_DONE_SUBJECTS = ["quiz.generate.done"]

RAG_INGEST_PUBLISH_SUBJECT = "rag.ingest.request"
RAG_INGEST_DONE_SUBJECT = "rag.ingest.done"
RAG_QUIZ_PUBLISH_SUBJECT = "quiz.generate.request"
RAG_QUIZ_DONE_SUBJECT = "quiz.generate.done"

DURABLE_INGEST_WORKER = "rag-ingest"
DURABLE_QUIZ_WORKER = "rag-quiz"

QUEUE_GROUP_INGEST = "rag-ingest"
QUEUE_GROUP_QUIZ = "rag-quiz"
