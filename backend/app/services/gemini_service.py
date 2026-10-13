from google import genai

from backend.app.core.config import settings


class GeminiService:
    def __init__(self) -> None:
        self.model = settings.gemini_model
        self.mock_mode = settings.mock_llm

        if self.mock_mode:
            self.client = None
            print("GeminiService: MOCK_LLM mode enabled.")
            return

        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(
            api_key=settings.gemini_api_key,
        )

    def generate(
        self,
        prompt: str,
    ) -> str:
        if self.mock_mode:
            return self._mock_generate(prompt)

        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
        )

        if not interaction.output_text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        return interaction.output_text.strip()

    def _mock_generate(
        self,
        prompt: str,
    ) -> str:
        prompt_lower = prompt.lower()

        # Query rewriting
        if "standalone search query" in prompt_lower:
            # Mock: return the latest user question unchanged.
            start = prompt.rfind("Latest user question:")
            end = prompt.rfind("Standalone search query:")
            return prompt[start + len("Latest user question:"):end].strip()

        # Citation validation
        if "supported source numbers" in prompt_lower:
            return "3"

        # Grounding evaluation
        if "grounding evaluator" in prompt_lower:
            return "1.0"

        # Context evaluation
        if (
            "retrieval quality" in prompt_lower
            and "return only a decimal number between 0 and 1" in prompt_lower
        ):
            return "0.75"

        # Answer generation
        if "answer:" in prompt_lower:
            return "The architecture is designed to be modular, testable, and production-oriented [1]."

        # Document summaries
        if "document summarization assistant" in prompt_lower:

            # Quick summary
            if "3 to 5 sentences" in prompt_lower:
                return (
                    "RAG-X is a multi-document AI knowledge assistant "
                    "for uploading documents and asking questions across "
                    "a knowledge base."
                )

            # Reduce stage: combine multiple partial summaries
            if "partial summaries" in prompt_lower:
                return (
                    "RAG-X is a multi-document AI knowledge assistant "
                    "that processes documents into chunks, uses hybrid "
                    "retrieval and reranking to find relevant evidence, "
                    "and generates grounded answers with citations."
                )

            # Map stage: summarize one chunk group
            if "document section" in prompt_lower:
                return (
                    "Partial summary: The section describes document "
                    "processing, chunking, retrieval, and grounded "
                    "generation in the RAG-X system."
                )

        if "detailed summary" in prompt_lower:
            return (
                "RAG-X processes uploaded documents, extracts and "
                "chunks their text, generates dense and sparse embeddings, "
                "and stores vectors in Qdrant. PostgreSQL stores application "
                "metadata, while hybrid retrieval and reranking improve "
                "evidence selection. LangGraph organizes the workflow, "
                "and grounding and citations help produce supported answers."
            )

        if "bullet-point list" in prompt_lower:
            return (
                "- Multi-document upload and processing\n"
                "- Dense and sparse hybrid retrieval\n"
                "- Reranking for relevant evidence\n"
                "- Grounded answers with citations\n"
                "- LangGraph-based RAG workflow"
            )

        if "methodology" in prompt_lower:
            return (
                "The system processes documents into chunks, generates "
                "dense and sparse embeddings, retrieves relevant evidence "
                "using hybrid search, reranks the candidates, and uses an "
                "LLM to generate grounded responses."
            )

        if "main findings" in prompt_lower:
            return (
                "The document describes a modular RAG architecture that "
                "combines document processing, hybrid retrieval, reranking, "
                "grounded generation, citations, and evaluation."
            )

        if "limitations" in prompt_lower:
            return (
                "The document does not describe specific limitations."
            )

        if "comma-separated list" in prompt_lower:
            return (
                "RAG, multi-document retrieval, embeddings, Qdrant, "
                "PostgreSQL, hybrid search, reranking, LangGraph, "
                "grounding, citations"
            )

        return "Mock Gemini response."


gemini_service = GeminiService()