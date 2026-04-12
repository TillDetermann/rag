import logging
from pathlib import Path
from typing import Any

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.settings.yaml import load_yaml_with_envvars

logger = logging.getLogger(__name__)


def load_active_settings() -> dict[str, Any]:
    """Load settings from settings.yaml."""
    path = Path(PROJECT_ROOT_PATH) / "settings.yaml"
    with path.open("r") as f:
        config = load_yaml_with_envvars(f)
    if not isinstance(config, dict):
        raise TypeError(f"Config file has no top-level mapping: {path}")
    logger.info("Loaded settings from %s", path)
    return config