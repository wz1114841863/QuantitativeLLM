import sys
import os
import time
import contextlib
from pathlib import Path


class DualLogger:
    """A logger that writes to both stdout and a file."""

    def __init__(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = file_path.open("w", buffering=1)
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


@contextlib.contextmanager
def dual_log(log_file: Path):
    dl = DualLogger(log_file)
    sys.stdout = dl
    try:
        yield dl
    finally:
        dl.close()
