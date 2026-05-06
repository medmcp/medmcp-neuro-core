"""Tests for MNI152 template resolution."""

import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

import medmcp_neuro.tools._template as _tmpl
from medmcp_neuro.tools._template import get_mni152_1mm

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


# ── FSL fallback ──────────────────────────────────────────────────────────────


def test_fsl_primary_template_returned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns the FSL full T1w when $FSLDIR is set and MNI152_T1_1mm.nii.gz exists."""
    fsl_std = tmp_path / "fsl" / "data" / "standard"
    fsl_std.mkdir(parents=True)
    primary = fsl_std / "MNI152_T1_1mm.nii.gz"
    primary.touch()
    monkeypatch.setenv("FSLDIR", str(tmp_path / "fsl"))
    with patch.object(_tmpl, "_CACHE_DIR", tmp_path / "cache"):
        result = get_mni152_1mm()
    assert result == primary


def test_fsl_brain_only_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns the brain-extracted FSL template when the full T1w is absent."""
    fsl_std = tmp_path / "fsl" / "data" / "standard"
    fsl_std.mkdir(parents=True)
    brain = fsl_std / "MNI152_T1_1mm_brain.nii.gz"
    brain.touch()
    monkeypatch.setenv("FSLDIR", str(tmp_path / "fsl"))
    with patch.object(_tmpl, "_CACHE_DIR", tmp_path / "cache"):
        result = get_mni152_1mm()
    assert result == brain


def test_fsl_set_but_no_templates_falls_through_to_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falls through to download when $FSLDIR has no usable templates."""
    monkeypatch.setenv("FSLDIR", str(tmp_path / "fsl"))
    cache_dir = tmp_path / "cache"
    with (
        patch.object(_tmpl, "_CACHE_DIR", cache_dir),
        patch.object(urllib.request, "urlretrieve"),
    ):
        get_mni152_1mm()
    assert (cache_dir / _MNI152_FILENAME).exists()


# ── download ──────────────────────────────────────────────────────────────────


def test_download_creates_cached_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On first use with no FSL, downloads and caches the template."""
    monkeypatch.delenv("FSLDIR", raising=False)
    cache_dir = tmp_path / "cache"
    cached = cache_dir / _MNI152_FILENAME
    with (
        patch.object(_tmpl, "_CACHE_DIR", cache_dir),
        patch.object(urllib.request, "urlretrieve"),
    ):
        result = get_mni152_1mm()
    assert result == cached
    assert cached.exists()


def test_download_failure_cleans_up_tmp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On download failure the partial temp file is removed and the error propagates."""
    monkeypatch.delenv("FSLDIR", raising=False)
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


def test_skull_stripped_fsl_returns_brain_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """skull_stripped=True uses FSL brain template when available."""
    fsl_std = tmp_path / "fsl" / "data" / "standard"
    fsl_std.mkdir(parents=True)
    full = fsl_std / "MNI152_T1_1mm.nii.gz"
    brain = fsl_std / "MNI152_T1_1mm_brain.nii.gz"
    full.touch()
    brain.touch()
    monkeypatch.setenv("FSLDIR", str(tmp_path / "fsl"))
    with patch.object(_tmpl, "_CACHE_DIR", tmp_path / "cache"):
        result = get_mni152_1mm(skull_stripped=True)
    assert result == brain


def test_skull_stripped_fsl_missing_brain_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """skull_stripped=True downloads the brain template when FSL has only the full T1w."""
    fsl_std = tmp_path / "fsl" / "data" / "standard"
    fsl_std.mkdir(parents=True)
    (fsl_std / "MNI152_T1_1mm.nii.gz").touch()  # full present, brain absent
    monkeypatch.setenv("FSLDIR", str(tmp_path / "fsl"))
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


def test_default_download_uses_full_template_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """skull_stripped=False (default) downloads the full T1w template URL."""
    monkeypatch.delenv("FSLDIR", raising=False)
    cache_dir = tmp_path / "cache"
    with (
        patch.object(_tmpl, "_CACHE_DIR", cache_dir),
        patch.object(urllib.request, "urlretrieve") as mock_dl,
    ):
        get_mni152_1mm(skull_stripped=False)
    url_used = mock_dl.call_args[0][0]
    assert "desc-brain" not in url_used
