"""Shared logging setup for pipeline scripts."""

import inspect
import logging

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: int = logging.INFO,
    *,
    root_level: int | None = None,
) -> tuple[Console, logging.Logger]:
    """Configure Rich logging and silence noisy third-party loggers.

    Args:
        level: Log level for the calling module's logger.
        root_level: Root logger level. Defaults to ``level``. Set to
            ``logging.CRITICAL`` to suppress all third-party INFO noise
            while keeping the module logger at ``level``.

    Returns:
        A ``(console, log)`` pair ready to use.
    """
    caller = inspect.stack()[1]
    module = caller.frame.f_globals.get("__name__", __name__)

    console = Console(stderr=True)
    logging.basicConfig(
        level=root_level if root_level is not None else level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_time=True, show_path=False)],
        force=True,
    )
    log = logging.getLogger(module)
    log.setLevel(level)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)

    return console, log
