"""Explicit, verified download of Hetionet v1.0 TSV/SIF for the optional demo.

Hetionet's SIF is stored with Git LFS. raw.githubusercontent.com serves a
small pointer, not the gzipped graph. Use GitHub's media endpoint for LFS data.
Do not bundle the dataset: see the upstream license and provenance policies.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Artifact:
    filename: str
    urls: tuple[str, ...]
    min_bytes: int
    magic: bytes
    sha256: str | None = None
    expected_bytes: int | None = None


BASE = "https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/"
MEDIA = "https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/tsv/"
GITHUB = "https://github.com/hetio/hetionet/raw/refs/heads/main/hetnet/tsv/"
NODES = Artifact(
    filename="hetionet-v1.0-nodes.tsv",
    urls=(BASE + "hetionet-v1.0-nodes.tsv", GITHUB + "hetionet-v1.0-nodes.tsv"),
    min_bytes=100_000,
    magic=b"id\tname\tkind",
)
EDGES = Artifact(
    filename="hetionet-v1.0-edges.sif.gz",
    urls=(MEDIA + "hetionet-v1.0-edges.sif.gz", GITHUB + "hetionet-v1.0-edges.sif.gz"),
    min_bytes=10_000_000,
    magic=b"\x1f\x8b",
    expected_bytes=12_401_407,
    sha256="611f0411ac666be4e0270e54bbca67407e14720bfc3099311f0f5eec25c5a947",
)


def validate(path: Path, artifact: Artifact) -> None:
    """Check actual tabular content, not a platform-specific newline sequence.

    The Hetionet nodes TSV can use LF or CRLF; comparing the raw first
    13 bytes against ``b"id\tname\tkind\n"`` incorrectly rejects valid
    data with CRLF. Read CSV columns instead, just like the converter does.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing required Hetionet file: {path}")
    size = path.stat().st_size
    with path.open("rb") as handle:
        preview = handle.read(192)
    if b"git-lfs.github.com/spec/v1" in preview:
        raise ValueError(f"{path} is a Git LFS pointer (not the dataset). Delete it and retry.")
    if preview.lstrip().lower().startswith((b"<html", b"<!doctype html")):
        raise ValueError(f"{path} contains HTML instead of Hetionet data (HTTP proxy or download error)")
    if size < artifact.min_bytes:
        raise ValueError(
            f"{path} is unexpectedly small ({size} bytes; expected at least "
            f"{artifact.min_bytes}). First bytes: {preview[:64]!r}"
        )
    if artifact.expected_bytes is not None and size != artifact.expected_bytes:
        raise ValueError(
            f"{path} size mismatch ({size} vs {artifact.expected_bytes} bytes)"
        )

    if artifact.filename.endswith(".tsv"):
        # The official nodes file may be CRLF-terminated. Decode as UTF-8
        # (optional BOM), and let csv handle LF and CRLF transparently.
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader, [])
                if header != ["id", "name", "kind"]:
                    raise ValueError(
                        f"{path} unexpected nodes header {str(header)[:160]!r}; "
                        "expected ['id', 'name', 'kind']"
                    )
                first = next(reader, [])
                if len(first) != 3 or not all(first):
                    raise ValueError(f"{path} has no valid three-column node row")
        except (OSError, UnicodeError, csv.Error) as err:
            raise ValueError(f"{path} is not readable Hetionet TSV: {err}") from err
    elif artifact.filename.endswith(".gz"):
        if not preview.startswith(b"\x1f\x8b"):
            raise ValueError(
                f"{path} is not gzip data (first bytes: {preview[:64]!r})"
            )
        try:
            with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader, [])
                if header != ["source", "metaedge", "target"]:
                    raise ValueError(f"{path} has unexpected SIF header {header!r}")
                first = next(reader, [])
                if len(first) != 3 or not all(first):
                    raise ValueError(f"{path} has no valid three-column edge row")
        except (OSError, UnicodeError, csv.Error, EOFError) as err:
            raise ValueError(f"{path} is not a readable gzip file: {err}") from err
    else:
        if not preview.startswith(artifact.magic):
            raise ValueError(f"{path} does not match {artifact.filename} signature")

    if artifact.sha256 is not None:
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        if sha.hexdigest() != artifact.sha256:
            raise ValueError(
                f"{path} SHA-256 mismatch; got {sha.hexdigest()}, "
                f"expected {artifact.sha256}"
            )


def download_one(artifact: Artifact, directory: Path, *, timeout: int = 180) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / artifact.filename
    cache_error: str | None = None
    if destination.exists():
        try:
            validate(destination, artifact)
        except ValueError as err:
            cache_error = str(err)
            print(f"[hetionet] cached file is invalid; attempting fresh download: {err}", flush=True)
        else:
            print(f"[hetionet] using verified existing file: {destination}", flush=True)
            return destination

    failures: list[str] = []
    for url in artifact.urls:
        temporary = None
        try:
            print(f"[hetionet] downloading {artifact.filename} from {url}", flush=True)
            with tempfile.NamedTemporaryFile(prefix=".hetionet-", suffix=".part", dir=directory,
                                             delete=False) as target:
                temporary = Path(target.name)
                req = Request(url, headers={"User-Agent": "gnn-rag-interview/0.2",
                                            "Accept": "application/octet-stream"})
                with urlopen(req, timeout=timeout) as response:
                    total = 0
                    while True:
                        piece = response.read(1024 * 1024)
                        if not piece:
                            break
                        total += len(piece)
                        if total > 50_000_000:
                            raise ValueError("Download larger than expected 50 MB limit")
                        target.write(piece)
            validate(temporary, artifact)
            os.replace(temporary, destination)
            print(f"[hetionet] verified: {destination}", flush=True)
            return destination
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            failures.append(f"{url}: {exc}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    raise RuntimeError(
        f"Could not obtain {artifact.filename}.\n" +
        (f"Cached file invalid: {cache_error}\n" if cache_error else "") +
        "\n".join(failures) +
        "\nAlternatively, download it manually from https://github.com/hetio/hetionet "
        "into data/raw/ and rerun. Git LFS pointer files are not valid inputs."
    )


def ensure_hetionet_inputs(directory: Path, *, download: bool = True) -> tuple[Path, Path]:
    """Fetch once, use cached validated raw files on later runs."""
    results = []
    for artifact in (NODES, EDGES):
        path = directory / artifact.filename
        if not path.exists() and not download:
            raise FileNotFoundError(
                f"{path} missing. Set HETIONET_AUTO_DOWNLOAD=1 or place official Hetionet "
                "v1.0 TSV/SIF files under data/raw/"
            )
        results.append(download_one(artifact, directory))
    return results[0], results[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate official Hetionet TSV/SIF")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    nodes, edges = ensure_hetionet_inputs(args.output)
    print(f"Hetionet inputs ready:\n  {nodes}\n  {edges}")


if __name__ == "__main__":
    main()
