"""Resolve Hetionet v1.0 source: verified raw files, user ZIP, or official HTTPS.

Never unpack arbitrary archive paths. Select only the expected two dataset members,
stream them into temporary files, validate content and SHA-256 before replace.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from .fetch_hetionet import EDGES, NODES, ensure_hetionet_inputs, validate


ARCHIVE_SUFFIXES = {
    NODES.filename: '/hetnet/tsv/' + NODES.filename,
    EDGES.filename: '/hetnet/tsv/' + EDGES.filename,
}


def _extract_from_zip(archive: Path, raw_dir: Path) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as pack:
        members = {}
        for artifact in (NODES, EDGES):
            matches = [entry for entry in pack.infolist()
                       if (entry.filename == artifact.filename or
                           entry.filename.endswith(ARCHIVE_SUFFIXES[artifact.filename]))
                       and not entry.is_dir()]
            if len(matches) != 1:
                raise ValueError(f'{archive}: expected exactly one {artifact.filename}, found {len(matches)}')
            item = matches[0]
            if item.file_size > 50_000_000:
                raise ValueError(f'Unexpectedly large member {item.filename}: {item.file_size}')
            members[artifact.filename] = item
        # Validate all extracted files before modifying an existing good cache.
        staged = {}
        try:
            for artifact in (NODES, EDGES):
                with tempfile.NamedTemporaryFile(prefix='.hetionet-zip-', suffix='.part',
                                                 dir=raw_dir, delete=False) as tmp:
                    staged_path = Path(tmp.name)
                    staged[artifact.filename] = staged_path
                    with pack.open(members[artifact.filename]) as source:
                        shutil.copyfileobj(source, tmp, length=1_048_576)
                validate(staged_path, artifact)
            for artifact in (NODES, EDGES):
                os.replace(staged[artifact.filename], raw_dir / artifact.filename)
            return raw_dir / NODES.filename, raw_dir / EDGES.filename
        finally:
            for path in staged.values():
                path.unlink(missing_ok=True)


def resolve_hetionet_source(data_root: Path, *, download: bool = True) -> tuple[Path, Path, str]:
    """Reuses verified raw files; extracts a supplied ZIP; otherwise downloads.

    Cache invalidity does not silently fall through with download disabled.
    """
    data_root = Path(data_root)
    raw_dir = data_root / 'raw'
    nodes, edges = raw_dir / NODES.filename, raw_dir / EDGES.filename
    try:
        validate(nodes, NODES)
        validate(edges, EDGES)
        print('[source] reused verified data/raw TSV+SIF.gz', flush=True)
        return nodes, edges, 'verified_cached_raw'
    except (ValueError, FileNotFoundError) as err:
        print(f'[source] raw cache missing or invalid: {err}', flush=True)

    archive_candidates = [data_root / 'archives' / 'hetionet-v1.0.0.zip',
                          raw_dir / 'hetionet-v1.0.0.zip',
                          data_root / 'hetionet-v1.0.0.zip']
    archive_dir = data_root / 'archives'
    if archive_dir.exists():
        archive_candidates.extend(sorted(archive_dir.glob('*.zip')))
    seen = set()
    for archive in archive_candidates:
        if archive in seen or not archive.is_file():
            continue
        seen.add(archive)
        try:
            nodes, edges = _extract_from_zip(archive, raw_dir)
            print(f'[source] extracted and verified {archive}', flush=True)
            return nodes, edges, f'local_zip:{archive.name}'
        except (BadZipFile, ValueError, OSError, EOFError) as err:
            # A bad supplied ZIP should not accidentally trigger a massive download
            raise RuntimeError(f'Invalid Hetionet archive {archive}: {err}') from err

    if not download:
        raise FileNotFoundError('Hetionet data missing. Place hetionet-v1.0.0.zip under '
                                'data/archives/ or the official raw TSV/SIF.gz under data/raw/, '
                                'or enable automatic download.')
    print('[source] local dataset absent; downloading verified official TSV/SIF.gz', flush=True)
    nodes, edges = ensure_hetionet_inputs(raw_dir, download=True)
    return nodes, edges, 'official_download'
