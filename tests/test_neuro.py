"""Tests for shared device-resolution helpers in _neuro."""

from unittest.mock import patch

import pytest

from medmcp_neuro.tools._neuro import Device, resolve_device

_DETECT = "medmcp_neuro.tools._neuro.detect_devices"


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_explicit_device_passes_through(device: Device) -> None:
    """An explicit device is returned unchanged (no torch probe needed)."""
    # No patching: an explicit device must not even call detect_devices.
    with patch(_DETECT, side_effect=AssertionError("should not probe for explicit device")):
        assert resolve_device(device) == device


def test_auto_prefers_cuda() -> None:
    """Resolve 'auto' to cuda when it is available."""
    with patch(_DETECT, return_value=["cpu", "cuda", "mps"]):
        assert resolve_device("auto") == "cuda"


def test_auto_picks_mps_when_no_cuda() -> None:
    """Resolve 'auto' to mps when cuda is absent but mps is available."""
    with patch(_DETECT, return_value=["cpu", "mps"]):
        assert resolve_device("auto") == "mps"


def test_auto_falls_back_to_cpu() -> None:
    """Resolve 'auto' to cpu when no accelerator is available."""
    with patch(_DETECT, return_value=["cpu"]):
        assert resolve_device("auto") == "cpu"
