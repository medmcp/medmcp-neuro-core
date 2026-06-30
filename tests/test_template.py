"""Tests for MNI152 template resolution."""

import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

import medmcp_neuro_core.tools._template as _tmpl
from medmcp_neuro_core.tools._template import get_mni152_1mm

_MNI152_FILENAME: str = _tmpl._MNI152_FILENAME  # type: ignore[reportPrivateUsage]


# ── cache hit ─────────────────────────────────────────────────────────────────


def test_cache_hit_returns_cached_path(tmp_path: Path) -> None:
    """Returns the cached file immediately when it exists."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = cache_dir / _MNI152_FILENAME
    cached.touch()
    with patch.object(_tmpl, "_CACHE_DIR", cache_dir):
        result = get_mni152_1mm()
    assert result == cached


# ── download ──────────────────────────────────────────────────────────────────


def test_download_creates_cached_file(tmp_path: Path) -> None:
    """On first use with no cached copy, downloads and caches the template."""
    cache_dir = tmp_path / "cache"
    cached = cache_dir / _MNI152_FILENAME
    with (
        patch.object(_tmpl, "_CACHE_DIR", cache_dir),
        patch.object(urllib.request, "urlretrieve"),
    ):
        result = get_mni152_1mm()
    assert result == cached
    assert cached.exists()


def test_fsldir_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A populated $FSLDIR is never used — only the cache and templateflow download are."""
    fsl_std = tmp_path / "fsl" / "data" / "standard"
    fsl_std.mkdir(parents=True)
    (fsl_std / "MNI152_T1_1mm.nii.gz").touch()
    monkeypatch.setenv("FSLDIR", str(tmp_path / "fsl"))
    cache_dir = tmp_path / "cache"
    with (
        patch.object(_tmpl, "_CACHE_DIR", cache_dir),
        patch.object(urllib.request, "urlretrieve"),
    ):
        result = get_mni152_1mm()
    assert result == cache_dir / _MNI152_FILENAME


def test_download_failure_cleans_up_tmp_file(tmp_path: Path) -> None:
    """On download failure the partial temp file is removed and the error propagates."""
    cache_dir = tmp_path / "cache"
    with (
        patch.object(_tmpl, "_CACHE_DIR", cache_dir),
        patch.object(urllib.request, "urlretrieve", side_effect=OSError("network error")),
        pytest.raises(OSError, match="network error"),
    ):
        get_mni152_1mm()
    assert not any(cache_dir.glob("*.tmp"))


# ── skull_stripped variant ────────────────────────────────────────────────────


def test_skull_stripped_cache_hit_returns_brain_path(tmp_path: Path) -> None:
    """skull_stripped=True returns the desc-brain cached file when it exists."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    brain_cached = cache_dir / _tmpl._MNI152_BRAIN_FILENAME  # type: ignore[reportPrivateUsage]
    brain_cached.touch()
    with patch.object(_tmpl, "_CACHE_DIR", cache_dir):
        result = get_mni152_1mm(skull_stripped=True)
    assert result == brain_cached


def test_skull_stripped_download_uses_brain_url(tmp_path: Path) -> None:
    """skull_stripped=True downloads and caches the desc-brain template."""
    cache_dir = tmp_path / "cache"
    brain_cached = cache_dir / _tmpl._MNI152_BRAIN_FILENAME  # type: ignore[reportPrivateUsage]
    with (
        patch.object(_tmpl, "_CACHE_DIR", cache_dir),
        patch.object(urllib.request, "urlretrieve") as mock_dl,
    ):
        result = get_mni152_1mm(skull_stripped=True)
    assert result == brain_cached
    url_used = mock_dl.call_args[0][0]
    assert "desc-brain" in url_used


def test_default_download_uses_full_template_url(tmp_path: Path) -> None:
    """skull_stripped=False (default) downloads the full T1w template URL."""
    cache_dir = tmp_path / "cache"
    with (
        patch.object(_tmpl, "_CACHE_DIR", cache_dir),
        patch.object(urllib.request, "urlretrieve") as mock_dl,
    ):
        get_mni152_1mm(skull_stripped=False)
    url_used = mock_dl.call_args[0][0]
    assert "desc-brain" not in url_used
