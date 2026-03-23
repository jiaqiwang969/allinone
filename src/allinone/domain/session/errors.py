"""Session domain errors."""


class SessionStateError(RuntimeError):
    """Raised when a work session enters an invalid state transition."""
