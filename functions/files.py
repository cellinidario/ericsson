"""Atomic file writes (a crash never leaves a half-written file) and content hashes."""
import hashlib
import json
from pathlib import Path
import time

import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _atomic(path, write):
    """Write to a temporary file, then rename. Windows can briefly lock a file being read: retry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    for attempt in range(20):
        try:
            write(temporary)
            temporary.replace(path)
            return path
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1*(attempt+1))


def save_json(path, value):
    return _atomic(path, lambda p: p.write_text(json.dumps(value, indent=2), encoding='utf-8'))


def save_torch(path, value):
    return _atomic(path, lambda p: torch.save(value, p))
