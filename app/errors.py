class UpstreamServiceError(Exception):
    """Base error for failures returned by upstream model providers."""


class UpstreamTimeoutError(UpstreamServiceError):
    """Raised when an upstream request times out."""


class UpstreamAuthError(UpstreamServiceError):
    """Raised when upstream authentication fails."""


class UpstreamNotFoundError(UpstreamServiceError):
    """Raised when the upstream endpoint or model is not found."""

