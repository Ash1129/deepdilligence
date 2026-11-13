"""Disk-based caching decorator.

Caching is OFF by default (see `USE_CACHE` in src/utils/config.py), so every
call hits the live source. Set DD_USE_CACHE=1 to turn it back on.
"""

import functools
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

from src.utils import config
from src.utils.config import CACHE_DIR

logger = logging.getLogger(__name__)


def _make_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a deterministic cache key from function name and arguments."""
    key_data = json.dumps({"func": func_name, "args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in sorted(kwargs.items())}}, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()


def disk_cache(subfolder: str = "general") -> Callable:
    """Decorator that caches function return values to disk as JSON.

    Args:
        subfolder: Subdirectory within the cache folder to store results.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Caching disabled → always fetch live, never read or write disk.
            if not config.cache_enabled():
                return func(*args, **kwargs)

            cache_dir = CACHE_DIR / subfolder
            cache_dir.mkdir(parents=True, exist_ok=True)

            cache_key = _make_cache_key(func.__name__, args, kwargs)
            cache_file = cache_dir / f"{cache_key}.json"

            if cache_file.exists():
                logger.debug("Cache hit for %s (key=%s)", func.__name__, cache_key[:12])
                with open(cache_file, "r") as f:
                    return json.load(f)

            logger.debug("Cache miss for %s (key=%s)", func.__name__, cache_key[:12])
            result = func(*args, **kwargs)

            try:
                with open(cache_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)
            except (TypeError, ValueError) as e:
                logger.warning("Could not cache result for %s: %s", func.__name__, e)

            return result
        return wrapper
    return decorator


def clear_cache(subfolder: str | None = None) -> int:
    """Remove cached files. Returns count of files deleted.

    Args:
        subfolder: If provided, only clear this subfolder. Otherwise clear all.
    """
    target = CACHE_DIR / subfolder if subfolder else CACHE_DIR
    if not target.exists():
        return 0
    count = 0
    for f in target.rglob("*.json"):
        f.unlink()
        count += 1
    return count
