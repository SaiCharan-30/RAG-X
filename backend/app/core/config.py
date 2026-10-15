from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG-X API"
    app_version: str = "0.1.0"

    # True prints every SQL statement (very noisy). Turn on only when debugging.
    debug: bool = False

    # Defaults match docker-compose.yml in this project (host ports 5433 / 6335),
    # so a missing .env can never connect to another project's database.
    database_url: str = (
        "postgresql+pg8000://"
        "rag_user:rag_password"
        "@localhost:5433/rag_x"
    )

    qdrant_url: str = "http://localhost:6335"
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