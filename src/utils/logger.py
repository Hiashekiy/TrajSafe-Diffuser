"""Simple logger that mirrors output to stdout and an optional log file."""

import os
import sys
import time


class Logger:
    def __init__(self, path=None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.start = time.time()

    def info(self, msg):
        line = f"[{time.time() - self.start:7.1f}s] {msg}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def scalar(self, name, value, step=None):
        tag = f"{name}" + (f"@{step}" if step is not None else "")
        self.info(f"{tag} = {value}")
