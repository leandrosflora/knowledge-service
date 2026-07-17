class KnowledgeBackendUnavailableError(Exception):
    """Raised when OpenSearch or the OpenAI embeddings API cannot be reached.

    Mapped to a 503 response by app.main's exception handler. A genuine "no relevant
    FAQ content" is a normal 200 with an empty result list (see app/api/search.py) -
    this exception is only for actual infrastructure failures, so the caller's own
    tenacity retry (in agent-runtime-renegotiation's search_knowledge_base tool) does
    its job instead of a transient failure being silently mistaken for "no answer."
    """
