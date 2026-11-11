import re
import pytest

def test_readme_deployment_metadata():
    """Verify that README.md specifies Gradio, Python 3.12, and matches requirements-dev.txt."""
    with open("README.md", "r", encoding="utf-8") as f:
        readme_text = f.read()

    # Check SDK and Python version in YAML block
    assert re.search(r"^sdk:\s*gradio", readme_text, re.MULTILINE), "README missing 'sdk: gradio'"
    assert re.search(r"^python_version:\s*3\.12", readme_text, re.MULTILINE), "README missing 'python_version: 3.12'"

    # Extract sdk_version from README
    sdk_match = re.search(r"^sdk_version:\s*([\d\.]+)", readme_text, re.MULTILINE)
    assert sdk_match, "README missing 'sdk_version'"
    readme_gradio_ver = sdk_match.group(1)

    # Check against requirements-dev.txt
    with open("requirements-dev.txt", "r", encoding="utf-8") as f:
        req_text = f.read()

    gradio_req_match = re.search(r"gradio==([\d\.]+)", req_text)
    assert gradio_req_match, "Gradio version not pinned in requirements-dev.txt"
    req_gradio_ver = gradio_req_match.group(1)

    assert readme_gradio_ver == req_gradio_ver, (
        f"Gradio version mismatch: README has {readme_gradio_ver}, requirements has {req_gradio_ver}"
    )
