class AIServiceError(Exception):
    """
    The question could not be answered because a backend service
    (Gemini, Qdrant, ...) failed. The message is safe to show to users;
    the technical details are written to the server log instead.
    """