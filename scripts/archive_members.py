#!/usr/bin/env python3
"""Read the members of the archive directory's tars, plain or gzip-compressed, without unpacking them.

The display dumps are archived as plain `.tar`, which `tarfile` can seek in: a member's bytes are
read by opening the tar and jumping to it. The raw board traces are archived as `.tar.gz`
(docs/Artifact/external_data.md, "Raw traces"), where a jump is a decompression from the start, so
hashing 1579 members one by one decompresses the archive 1579 times. For a compressed tar this reads
the stream once and records every member's sha256 on the way; for a plain tar it keeps the seek.

    import archive_members as AM
    for tar in AM.archives(dir): ...
    for m in AM.members(tar): m.name, m.isfile(), m.issym(), m.islnk(), m.linkname
    AM.member_sha256(tar, name)
"""
from __future__ import annotations

import glob
import hashlib
import os
import tarfile

PATTERNS = ("*.tar", "*.tar.gz", "*.tgz")
_SCAN: dict[str, tuple[list, dict]] = {}


def archives(archdir):
    """Every tar in the directory this module can read, sorted by path."""
    out = set()
    for p in PATTERNS:
        out |= set(glob.glob(os.path.join(archdir, p)))
    return sorted(out)


def compressed(tar):
    return not tar.endswith(".tar")


def _scan(tar):
    if tar not in _SCAN:
        mems, shas = [], {}
        if compressed(tar):
            with tarfile.open(tar, "r|*") as tf:
                for m in tf:
                    mems.append(m)
                    if m.isfile():
                        h = hashlib.sha256()
                        fo = tf.extractfile(m)
                        for b in iter(lambda: fo.read(1 << 20), b""):
                            h.update(b)
                        shas[m.name] = h.hexdigest()
        else:
            with tarfile.open(tar) as tf:
                mems = tf.getmembers()
        _SCAN[tar] = (mems, shas)
    return _SCAN[tar]


def members(tar):
    """The TarInfo of every member, in archive order. Raises what tarfile raises on a bad archive."""
    return _scan(tar)[0]


def member_sha256(tar, name):
    """sha256 of one regular member's bytes."""
    mems, shas = _scan(tar)
    if name in shas:
        return shas[name]
    h = hashlib.sha256()
    with tarfile.open(tar) as tf:
        fo = tf.extractfile(name)
        for b in iter(lambda: fo.read(1 << 20), b""):
            h.update(b)
    shas[name] = h.hexdigest()
    return shas[name]
