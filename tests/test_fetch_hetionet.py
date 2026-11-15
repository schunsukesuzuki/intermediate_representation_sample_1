"""Offline downloader tests; no biomedical data or network used."""
from __future__ import annotations

import gzip
import hashlib
import io

import pytest

from gnn_rag.fetch_hetionet import Artifact, download_one, validate, ensure_hetionet_inputs, EDGES


def test_offline_missing_inputs_have_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="HETIONET_AUTO_DOWNLOAD"):
        ensure_hetionet_inputs(tmp_path, download=False)


def test_reject_lfs_pointer(tmp_path):
    path = tmp_path / "hetionet-v1.0-edges.sif.gz"
    path.write_bytes(b"version https://git-lfs.github.com/spec/v1\n" + b"a" * 120)
    with pytest.raises(ValueError, match="Git LFS pointer"):
        validate(path, EDGES)


def test_atomic_download_and_cached_file(monkeypatch, tmp_path):
    content = b"id\tname\tkind\n" + b"x\tlabel\tDisease\n" * 8
    spec = Artifact("tiny-nodes.tsv", ("https://example.org/nodes.tsv",), min_bytes=64,
                    magic=b"id\tname\tkind\n", sha256=hashlib.sha256(content).hexdigest())
    calls = []

    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): self.close()

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return Response(content)

    monkeypatch.setattr("gnn_rag.fetch_hetionet.urlopen", fake_urlopen)
    path = download_one(spec, tmp_path)
    assert path.read_bytes() == content
    assert download_one(spec, tmp_path) == path
    assert len(calls) == 1
    assert not list(tmp_path.glob("*.part"))


def test_bad_download_has_no_final_file(monkeypatch, tmp_path):
    spec = Artifact("hetionet-v1.0-edges.sif.gz", ("https://example.org/edges",),
                    min_bytes=50, magic=b"\x1f\x8b")

    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): self.close()

    monkeypatch.setattr("gnn_rag.fetch_hetionet.urlopen", lambda request, timeout: Response(
        b"version https://git-lfs.github.com/spec/v1\n" + b"0" * 80))
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        download_one(spec, tmp_path)
    assert not (tmp_path / spec.filename).exists()
    assert not list(tmp_path.glob("*.part"))


def test_gzip_header_validation(tmp_path):
    blob = gzip.compress(b"source\tmetaedge\ttarget\nA\tGpBP\tB\n")
    spec = Artifact("mini.sif.gz", ("https://example.org/x",),
                    min_bytes=10, magic=b"\x1f\x8b",
                    expected_bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
    path = tmp_path / spec.filename
    path.write_bytes(blob)
    validate(path, spec)


def test_nodes_crlf_is_not_rejected_as_wrong_header(tmp_path):
    # Old implementation compared a hardcoded LF-terminated byte prefix.
    from gnn_rag.fetch_hetionet import NODES
    file = tmp_path / NODES.filename
    file.write_bytes(b"id\tname\tkind\r\n" +
                     b"Disease::DOID:10652\tAlzheimer's disease\tDisease\r\n" * 4000)
    validate(file, NODES)


def test_nodes_utf8_bom_supported(tmp_path):
    from gnn_rag.fetch_hetionet import NODES
    file = tmp_path / NODES.filename
    file.write_bytes(b"\xef\xbb\xbf" + b"id\tname\tkind\n" +
                     b"Disease::DOID:10652\tAlzheimer's disease\tDisease\n" * 4000)
    validate(file, NODES)


def test_nodes_html_invalid_has_clear_error(tmp_path):
    from gnn_rag.fetch_hetionet import NODES
    file = tmp_path / NODES.filename
    file.write_text("<html>access denied</html>" * 10000, encoding="utf-8")
    with pytest.raises(ValueError, match="contains HTML"):
        validate(file, NODES)


def test_bad_cached_file_is_replaced_only_after_valid_download(monkeypatch, tmp_path):
    spec = Artifact("nodes.tsv", ("https://example.org/one",), min_bytes=32,
                    magic=b"id\tname\tkind")
    old = tmp_path / "nodes.tsv"
    old.write_bytes(b"bad cached content" * 10)
    body = b"id\tname\tkind\r\n" + b"a\tA\tDisease\r\n" * 5

    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): self.close()

    monkeypatch.setattr("gnn_rag.fetch_hetionet.urlopen",
                        lambda request, timeout: Response(body))
    assert download_one(spec, tmp_path).read_bytes() == body
    assert not list(tmp_path.glob("*.part"))


def test_invalid_cache_survives_failed_download(monkeypatch, tmp_path):
    spec = Artifact("nodes.tsv", ("https://example.org/offline",), min_bytes=32,
                    magic=b"id\tname\tkind")
    destination = tmp_path / "nodes.tsv"
    old = b"stale invalid file" * 10
    destination.write_bytes(old)

    def failed_urlopen(request, timeout):
        raise URLError("offline")

    from urllib.error import URLError
    monkeypatch.setattr("gnn_rag.fetch_hetionet.urlopen", failed_urlopen)
    with pytest.raises(RuntimeError, match="Cached file invalid"):
        download_one(spec, tmp_path)
    assert destination.read_bytes() == old
    assert not list(tmp_path.glob("*.part"))
