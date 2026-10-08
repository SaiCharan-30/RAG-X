from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG-X API"
    app_version: str = "0.1.0"
    debug: bool = True
    

    database_url: str = (
        "postgresql+psycopg://"
        "rag_user:rag_password"
        "@localhost:5432/rag_x"
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_x_documents"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"

    mock_llm: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()