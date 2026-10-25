"""Serialize training writers sharing a campaign output directory."""
from contextlib import contextmanager
import fcntl

from .campaign import run_dir


@contextmanager
def training_lock(out, job):
    directory = run_dir(out, job)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'train.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
