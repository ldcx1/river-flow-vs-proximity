"""Make ``experiments`` importable when running scripts from a source checkout
without ``pip install``. Import this first in each script.
"""

import glob
import gzip
import io
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Shipped data bundle. Scripts default their path arguments here so the whole
# suite runs against the ingested CSV/JSON with no arguments (flags still override).
DATA_DE = os.path.join(_ROOT, "data", "de")
DATA_RO = os.path.join(_ROOT, "data", "ro")


def _resolve(path):
    """Return ``path`` if it exists, else its ``.gz`` sibling, else ``path``.

    One shipped panel (``de/turb_series_national.csv``) exceeds GitHub's 100 MiB
    per-file limit uncompressed, so the bundle carries it gzipped. Resolving here
    keeps every script's default argument written as the plain ``.csv`` name.
    """
    if os.path.exists(path):
        return path
    gz = path + ".gz"
    return gz if os.path.exists(gz) else path


def data_de(name):
    """Absolute path to a file in the shipped German data bundle (data/de/)."""
    return _resolve(os.path.join(DATA_DE, name))


def data_ro(name):
    """Absolute path to a file in the shipped Romanian data bundle (data/ro/)."""
    return _resolve(os.path.join(DATA_RO, name))


class _Concat(io.RawIOBase):
    """Read several files back to back as one byte stream."""

    def __init__(self, paths):
        self._files = [open(p, "rb") for p in paths]

    def readable(self):
        return True

    def readinto(self, buf):
        while self._files:
            n = self._files[0].readinto(buf)
            if n:
                return n
            self._files.pop(0).close()
        return 0


def open_text(path, encoding="utf-8"):
    """Open a shipped data file for reading, transparently handling gzip.

    Accepts the plain or the ``.gz`` name and opens whichever is present. If only
    the split ``.gz.part*`` files exist (the German per-pass archive), they are
    read in order as one gzip stream, so no reassembly is needed.
    """
    path = _resolve(path)
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding=encoding)
    if not os.path.exists(path):
        gz = path if path.endswith(".gz") else path + ".gz"
        parts = sorted(glob.glob(gz + ".part*"))
        if parts:
            raw = io.BufferedReader(_Concat(parts))
            return io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding=encoding)
    return open(path, encoding=encoding)
