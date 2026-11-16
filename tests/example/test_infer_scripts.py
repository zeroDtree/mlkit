"""Every example training script has an inference script that can find its checkpoints.

Run with:
    uv run pytest tests/example/test_infer_scripts.py
"""

from __future__ import annotations

import re
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "example"


def _config_name(script: Path) -> str:
    match = re.search(r'config_name="([^"]+)"', script.read_text(encoding="utf-8"))
    assert match is not None, f"{script.relative_to(REPO_ROOT)} has no config_name"
    return match.group(1)


def test_every_training_script_has_an_inference_script_with_the_same_config():
    training_scripts = sorted(EXAMPLE_ROOT.rglob("main*.py"))
    assert training_scripts, "expected example training scripts"

    for training_script in training_scripts:
        inference_script = training_script.with_name(training_script.name.replace("main", "infer", 1))
        assert inference_script.is_file(), f"missing {inference_script.relative_to(REPO_ROOT)}"
        assert _config_name(inference_script) == _config_name(training_script)


def test_example_configs_define_the_infer_section():
    config_paths = sorted(path for path in EXAMPLE_ROOT.rglob("*.yaml") if "artifacts" not in path.parts)
    assert config_paths, "expected example configs"

    for config_path in config_paths:
        config = OmegaConf.to_container(OmegaConf.load(config_path))
        assert isinstance(config, dict)
        infer = config.get("infer")
        assert isinstance(infer, dict), f"{config_path.relative_to(REPO_ROOT)} has no infer section"
        assert {"seed", "mixed_precision", "checkpoint_dir"} <= set(infer)
