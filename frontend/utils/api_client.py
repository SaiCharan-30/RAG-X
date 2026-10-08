import requests


BACKEND_URL = "http://127.0.0.1:8000"


def health_check() -> dict:
    response = requests.get(
        f"{BACKEND_URL}/health",
        timeout=5,
    )

    response.raise_for_status()

    return response.json()


def summarize_document(
    document_id: str,
    summary_type: str = "quick",
) -> dict:
    response = requests.post(
        f"{BACKEND_URL}/documents/{document_id}/summary",
        params={"summary_type": summary_type},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()

def list_documents() -> list[dict]:
    response = requests.get(
        f"{BACKEND_URL}/documents",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()