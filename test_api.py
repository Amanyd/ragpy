import time
import uuid
import boto3
import requests

API_URL = "http://localhost:8000/api/v1"
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS = "minioadmin"
MINIO_SECRET = "minioadmin"
BUCKET_NAME = "course-documents"
COURSE_ID = "course_test_api"
HEADERS = {"x-internal-token": "development_secret_token"}

FILES = [
    {
        "path": "zstdChunkedProposal.pdf",
        "query": "What is the primary topic of the Zstandard Chunked Proposal document?"
    },
    {
        "path": "ad3.docx",
        "query": "Can you summarize the main topics or keywords found exclusively in the ad3 document?"
    },
    {
        "path": "adp2.pptx",
        "query": "What is the presentation adp2 generally about? List any major slide headers you can see."
    }
]

print("1. Uploading files to MinIO...")
s3 = boto3.client(
    "s3", 
    endpoint_url=MINIO_ENDPOINT, 
    aws_access_key_id=MINIO_ACCESS, 
    aws_secret_access_key=MINIO_SECRET
)

try:
    s3.head_bucket(Bucket=BUCKET_NAME)
except:
    s3.create_bucket(Bucket=BUCKET_NAME)

print("\n2. Triggering Ingestion for all files...")
for file_info in FILES:
    file_path = file_info["path"]
    file_id = str(uuid.uuid4())
    key = f"{COURSE_ID}/{file_id}_{file_path}"
    
    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, BUCKET_NAME, key)
    
    print(f"Uploaded {file_path} to MinIO: {key}")
    
    res = requests.post(f"{API_URL}/ingest/", headers=HEADERS, json={
        "bucket": BUCKET_NAME, 
        "key": key, 
        "course_id": COURSE_ID,
        "file_id": file_id, 
        "file_name": file_path, 
        "teacher_id": "test_teacher"
    })
    print(f"Ingest API Status for {file_path}:", res.status_code)
    print(f"Response: {res.text}\n")

print("Waiting 60 seconds for background ingestion to complete for all files...")
time.sleep(60)

print("\n3. Testing Chatbot for each file...")
for file_info in FILES:
    query = file_info["query"]
    print("-" * 50)
    print(f"User Query: {query}")
    chat_res = requests.post(f"{API_URL}/chat/", headers=HEADERS, json={
        "course_id": COURSE_ID,
        "query": query,
        "stream": False
    })
    print("Chat API Status:", chat_res.status_code)
    if chat_res.ok:
        print("\n[Chatbot Reply]\n", chat_res.json()["answer"])
    else:
        print("Error:", chat_res.text)

print("-" * 50)
print("\nQuiz generation skipped for now.")
