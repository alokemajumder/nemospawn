"""Tests for NGC registry operations."""

from unittest.mock import patch, MagicMock

from nemospawn.ngc.registry import NGCModel, check_ngc_auth, list_models


def test_ngc_model_full_name():
    m = NGCModel(org="nvidia", name="llama-3-70b", version="v1.0")
    assert m.full_name == "nvidia/llama-3-70b:v1.0"


def test_ngc_model_default_version():
    m = NGCModel(org="nvidia", name="nemotron")
    assert m.full_name == "nvidia/nemotron:latest"


def test_check_ngc_auth_not_installed():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert check_ngc_auth() is False


def test_check_ngc_auth_success():
    result = MagicMock()
    result.returncode = 0
    with patch("subprocess.run", return_value=result):
        assert check_ngc_auth() is True


def test_list_models_not_installed():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert list_models("nvidia") == []


def test_list_models_success():
    result = MagicMock()
    result.returncode = 0
    result.stdout = "HEADER\nllama-3-70b  v1.0\nnemotron  v2.0\n"
    with patch("subprocess.run", return_value=result):
        models = list_models("nvidia")
        assert len(models) == 2
        assert models[0]["name"] == "llama-3-70b"
