from pathlib import Path

from pipeline.common.hashing import (
    hash_sampled_frames,
    sample_frame_indices,
    sha256_bytes,
    sha256_file,
    sha256_json,
)


def test_sha256_bytes_known_value():
    assert sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sha256_file_matches_bytes(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert sha256_file(p) == sha256_bytes(b"hello")


def test_sha256_json_is_key_order_invariant():
    a = sha256_json({"a": 1, "b": 2})
    b = sha256_json({"b": 2, "a": 1})
    assert a == b


def test_sample_frame_indices_includes_endpoints():
    idxs = sample_frame_indices(1000, 5)
    assert idxs[0] == 0 and idxs[-1] == 999 and len(idxs) == 5


def test_sample_frame_indices_small_total():
    assert sample_frame_indices(3, 5) == [0, 1, 2]
    assert sample_frame_indices(0, 5) == []


def test_hash_sampled_frames_round_trip(tmp_path: Path):
    paths = []
    for i in range(10):
        p = tmp_path / f"{i:03d}.png"
        p.write_bytes(f"frame_{i}".encode())
        paths.append(p)
    h = hash_sampled_frames(paths, n=5)
    assert len(h) == 5
    # Rerun should match.
    h2 = hash_sampled_frames(paths, n=5)
    assert h == h2
