from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path


def _open_content_readonly(path: Path):
    """Open a content file without following a final-component symlink."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    return os.fdopen(descriptor, "rb")


def read_verified_content_file(path: Path, *, checksum_sha256: str, size_bytes: int) -> bytes | None:
    """Read and freshly verify one regular content file in the same operation."""

    if path.is_symlink() or size_bytes < 0:
        return None
    digest = hashlib.sha256()
    content = bytearray()
    try:
        with _open_content_readonly(path) as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                content.extend(chunk)
                digest.update(chunk)
    except OSError:
        return None
    if len(content) != size_bytes or digest.hexdigest() != checksum_sha256:
        return None
    return bytes(content)


def verify_content_file(path: Path, *, checksum_sha256: str, size_bytes: int) -> bool:
    """Freshly verify one content-addressed file without metadata caching."""

    if path.is_symlink() or size_bytes < 0:
        return False
    digest = hashlib.sha256()
    actual_size = 0
    try:
        with _open_content_readonly(path) as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return False
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                actual_size += len(chunk)
                digest.update(chunk)
    except OSError:
        return False
    return actual_size == size_bytes and digest.hexdigest() == checksum_sha256


def store_content_atomically(path: Path, data: bytes, *, checksum_sha256: str) -> str:
    """Durably store verified bytes at a content-addressed path.

    The final path is never written in place. A correct existing file is reused;
    a corrupt or interrupted file is repaired through an atomic replacement.
    """

    actual_checksum = hashlib.sha256(data).hexdigest()
    if actual_checksum != checksum_sha256:
        raise ValueError("Content checksum does not match the requested storage identity.")

    path.parent.mkdir(parents=True, exist_ok=True)
    existed_before = path.exists() or path.is_symlink()
    if verify_content_file(path, checksum_sha256=checksum_sha256, size_bytes=len(data)):
        return "existing"

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if not verify_content_file(
            temporary_path,
            checksum_sha256=checksum_sha256,
            size_bytes=len(data),
        ):
            raise OSError("Temporary content-addressed write failed verification.")
        os.replace(temporary_path, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass

    if not verify_content_file(path, checksum_sha256=checksum_sha256, size_bytes=len(data)):
        raise OSError("Final content-addressed write failed verification.")
    return "repaired" if existed_before else "created"
