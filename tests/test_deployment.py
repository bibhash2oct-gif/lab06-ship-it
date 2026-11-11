"""The files a Space reads must agree with each other and with what is installed."""

import json
import re
from importlib.metadata import version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read_settings() -> dict:
    """The key: value lines between the two --- lines at the top of README.md."""
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "---", "README.md must start with the Space settings block"
    end = lines.index("---", 1)
    settings = {}
    for line in lines[1:end]:
        key, _, value = line.partition(":")
        settings[key.strip()] = value.strip()
    return settings


def read_pins(file_name: str) -> dict:
    """package -> pinned version, for every line except comments and -r lines."""
    pins = {}
    for line in (ROOT / file_name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-r")):
            package, _, pinned = line.partition("==")
            pins[package.strip().lower()] = pinned.strip()
    return pins


def test_scikit_learn_matches_the_trained_model():
    info = json.loads((ROOT / "models" / "model_info.json").read_text(encoding="utf-8"))
    trained = info["scikit_learn"]
    assert version("scikit-learn") == trained
    assert read_pins("requirements.txt")["scikit-learn"] == trained


@pytest.mark.parametrize("package", ["scikit-learn", "pandas", "joblib"])
def test_installed_version_matches_the_pin(package):
    assert version(package) == read_pins("requirements.txt")[package]


def test_requirements_are_short_and_pinned():
    pins = read_pins("requirements.txt")
    assert set(pins) == {"scikit-learn", "pandas", "joblib"}
    assert all(pins.values()), "every line in requirements.txt needs ==version"


def test_gradio_is_not_in_requirements():
    assert "gradio" not in read_pins("requirements.txt")


def test_sdk_version_matches_installed_gradio():
    assert read_settings()["sdk_version"] == version("gradio")


def test_app_file_named_in_readme_exists():
    assert (ROOT / read_settings()["app_file"]).is_file()


def test_dev_requirements_add_pinned_tools():
    first_line = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").split()[:2]
    assert first_line == ["-r", "requirements.txt"]
    pins = read_pins("requirements-dev.txt")
    for package in ("gradio", "pytest", "ruff"):
        assert pins.get(package), f"{package} must be pinned in requirements-dev.txt"


def test_models_folder_is_not_ignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert not any(line.strip().strip("/") == "models" for line in ignored)


# Added by Joe: the README settings block must name Gradio and Python 3.12, and its
# Gradio version must match the pin in requirements-dev.txt.
def test_readme_deployment_metadata():
    """Verify README.md specifies Gradio and Python 3.12.

    Also check that its Gradio version matches requirements-dev.txt.
    """
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")

    # Check SDK and Python version in YAML block
    assert re.search(r"^sdk:\s*gradio", readme_text, re.MULTILINE), (
        "README missing 'sdk: gradio'"
    )
    assert re.search(r"^python_version:\s*3\.12", readme_text, re.MULTILINE), (
        "README missing 'python_version: 3.12'"
    )

    # Extract sdk_version from README
    sdk_match = re.search(r"^sdk_version:\s*([\d\.]+)", readme_text, re.MULTILINE)
    assert sdk_match, "README missing 'sdk_version'"
    readme_gradio_ver = sdk_match.group(1)

    # Check against requirements-dev.txt
    req_text = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    gradio_req_match = re.search(r"gradio==([\d\.]+)", req_text)
    assert gradio_req_match, "Gradio version not pinned in requirements-dev.txt"
    req_gradio_ver = gradio_req_match.group(1)

    assert readme_gradio_ver == req_gradio_ver, (
        f"Gradio version mismatch: README has {readme_gradio_ver}, "
        f"requirements has {req_gradio_ver}"
    )
