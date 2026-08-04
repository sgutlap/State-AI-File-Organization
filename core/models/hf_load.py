"""
Quiet Hugging Face / transformers loads for CLI paths.

Prefers local cache (local_files_only) and suppresses Hub auth warnings,
LOAD REPORT noise (e.g. DistilBERT vocab_* unexpected keys), and progress bars.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

T = TypeVar("T")

_HUB_LOGGERS = (
    "transformers",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
    "transformers.tokenization_utils_base",
    "transformers.utils.logging",
    "transformers.utils.loading_report",
    "huggingface_hub",
    "huggingface_hub.utils",
    "huggingface_hub.file_download",
    "huggingface_hub._snapshot_download",
)


@contextmanager
def quiet_hf_logging() -> Iterator[None]:
    """Raise transformers/huggingface_hub log levels and disable progress bars."""
    env_overrides = {
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    saved_env = {k: os.environ.get(k) for k in env_overrides}
    for k, v in env_overrides.items():
        os.environ[k] = v

    saved_levels: dict[str, int] = {}
    for name in _HUB_LOGGERS:
        lg = logging.getLogger(name)
        saved_levels[name] = lg.level
        lg.setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as transformers_logging

        prev_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        disable_bar = getattr(transformers_logging, "disable_progress_bar", None)
        enable_bar = getattr(transformers_logging, "enable_progress_bar", None)
        if callable(disable_bar):
            disable_bar()
    except Exception:
        prev_verbosity = None
        enable_bar = None

    try:
        yield
    finally:
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)
        for k, old in saved_env.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        try:
            from transformers.utils import logging as transformers_logging

            if prev_verbosity is not None:
                transformers_logging.set_verbosity(prev_verbosity)
            if callable(enable_bar):
                enable_bar()
        except Exception:
            pass


def from_pretrained_cached(loader: Callable[..., T], name_or_path: str, **kwargs: Any) -> T:
    """
    Load with local_files_only=True when the Hub cache already has the artifact.
    Falls back to network only if the local load fails.
    """
    with quiet_hf_logging():
        try:
            return loader(name_or_path, local_files_only=True, **kwargs)
        except Exception:
            return loader(name_or_path, local_files_only=False, **kwargs)
