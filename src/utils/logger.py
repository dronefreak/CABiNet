#!/usr/bin/python
# -*- encoding: utf-8 -*-

from functools import lru_cache
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
import torch.distributed as dist


def setup_logger(logpth):
    logpth = Path(logpth)
    Path.mkdir(logpth, parents=True, exist_ok=True)
    logfile = "CABiNet-{}.log".format(time.strftime("%Y-%m-%d-%H-%M-%S"))
    logfile = logpth / logfile
    FORMAT = "%(message)s"
    log_level = logging.INFO
    if dist.is_initialized() and not dist.get_rank() == 0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, filename=str(logfile))
    logging.root.addHandler(logging.StreamHandler())


class RichConsoleManager:
    """Provides a memoized and configurable Rich Console instance, with optional file
    logging and logging module integration."""

    DEFAULT_THEME = {
        "info": "bold bright_green",
        "warning": "bold bright_yellow",
        "danger": "bold bright_red",
        "summary": "italic green",
    }

    @classmethod
    def get_console(
        cls,
        theme_overrides: Optional[Dict[str, str]] = None,
        record: bool = False,
        log_path: Optional[Union[str, Path]] = None,
    ) -> Console:
        """Get a Rich Console instance.

        Args:
            theme_overrides: Custom styles to override the defaults.
            record: Enable console recording (for testing/logging).
            log_path: If set, console output will be written to the given file.

        Returns:
            Console: Configured Rich Console instance.
        """
        theme = cls._build_theme(theme_overrides)
        log_path_str = str(log_path) if log_path else None
        return cls._get_cached_console(theme, record, log_path_str)

    @classmethod
    def setup_logging(
        cls,
        level: int = logging.INFO,
        console: Optional[Console] = None,
        log_format: str = "%(message)s",
        show_path: bool = False,
    ) -> None:
        """Setup logging with RichHandler.

        Args:
            level: Logging level (e.g., logging.DEBUG).
            console: Optional Rich Console to attach (default will be used otherwise).
            log_format: Log format string.
            show_path: Whether to display file path in log messages.
        """
        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    console=console,
                    show_path=show_path,
                    markup=True,
                )
            ],
        )

    @staticmethod
    def _build_theme(overrides: Optional[Dict[str, str]]) -> Dict[str, str]:
        theme = RichConsoleManager.DEFAULT_THEME.copy()
        if overrides:
            theme.update(overrides)
        return theme

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_console_factory(
        theme_dict_frozen: frozenset,
        record: bool,
        log_path: Optional[str],
    ) -> Console:
        theme = Theme(dict(theme_dict_frozen))
        file = open(log_path, "a") if log_path else sys.stdout
        return Console(theme=theme, record=record, file=file)

    @classmethod
    def _get_cached_console(
        cls,
        theme_dict: Dict[str, str],
        record: bool,
        log_path: Optional[str],
    ) -> Console:
        frozen = frozenset(theme_dict.items())
        return cls._cached_console_factory(
            theme_dict_frozen=frozen, record=record, log_path=log_path
        )
