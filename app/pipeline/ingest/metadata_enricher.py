
from llama_index.core import Document


def enrich_metadata(
    documents: list[Document],
    course_id: str,
    file_id: str,
    file_name: str,
    teacher_id: str,
) -> list[Document]:
    for doc in documents:
        doc.metadata["course_id"] = course_id
        doc.metadata["file_id"] = file_id
        doc.metadata["teacher_id"] = teacher_id
        doc.metadata["file_name"] = file_name

        excluded = set(getattr(doc, "excluded_embed_metadata_keys", []) or [])
        excluded.update({"course_id", "file_id", "teacher_id"})
        doc.excluded_embed_metadata_keys = list(excluded)

    return documents

