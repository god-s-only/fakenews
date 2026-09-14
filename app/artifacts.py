"""Fingerprinting for the frozen model/vectorizer artifacts.

The promoted detector artifacts are immutable and pinned by SHA-256 in
``reports/release_manifest.json``.  Computing the fingerprint is part of the
model load lifecycle so a loaded ``ModelService`` can always answer "which
exact bytes are serving predictions" without the caller re-reading files.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactFingerprint:
    """SHA-256 and size of a single artifact file."""

    path: Path
    sha256: str
    size_bytes: int


def fingerprint_file(path: Path, chunk_size: int = 1 << 20) -> ArtifactFingerprint:
    """Return the SHA-256 and size of ``path`` (streaming, low memory).

    The whole file is hashed so the digest provably binds the exact bytes on
    disk to the running detector, matching the release-manifest pins.
    """
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
            size += len(chunk)
    return ArtifactFingerprint(path=path, sha256=digest.hexdigest(), size_bytes=size)