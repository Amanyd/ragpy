
import tempfile
from pathlib import Path

from app.store.minio import download_file


async def load_file(bucket: str, key: str) -> Path:
    data = await download_file(bucket=bucket, key=key)
    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_ingest_"))
    file_path = tmp_dir / Path(key).name
    file_path.write_bytes(data)
    return file_path

