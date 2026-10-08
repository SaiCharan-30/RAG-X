from pathlib import Path

from backend.app.database.connection import SessionLocal
from backend.app.services.indexing import index_document


document_id = "f9438aea-9977-4e3b-a54a-2b0d8ae8f63c"

file_path = Path(
    "data/uploads/f9438aea-9977-4e3b-a54a-2b0d8ae8f63c.txt"
)

db = SessionLocal()

try:
    count = index_document(
        file_path=file_path,
        document_id=document_id,
        db=db,
    )
    print("Indexed chunks:", count)
finally:
    db.close()