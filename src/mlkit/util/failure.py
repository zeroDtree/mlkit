from contextvars import ContextVar

_deferred: ContextVar[BaseException | None] = ContextVar("deferred_error", default=None)


def defer_error(exc: BaseException) -> None:
    """Remember the first rank-local failure.

    Later failures stay behind the first one. :func:`raise_deferred_error`
    raises it after every rank has left the training loop.

    Args:
        exc: failure that must not abort this rank while others are still
            inside a collective.
    """
    if _deferred.get() is None:
        _deferred.set(exc)


def has_deferred_error() -> bool:
    """Return whether this process has remembered a rank-local failure."""
    return _deferred.get() is not None


def clear_deferred_error() -> None:
    """Drop a remembered rank-local failure without raising it."""
    _deferred.set(None)


def raise_deferred_error() -> None:
    """Raise the remembered rank-local failure, if this process stored one."""
    exc = _deferred.get()
    _deferred.set(None)
    if exc is not None:
        raise exc
