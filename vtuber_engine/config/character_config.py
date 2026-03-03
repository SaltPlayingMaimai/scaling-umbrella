"""
角色配置加载模块。

从 YAML 文件加载角色定义，映射为 CharacterConfig。
"""

from __future__ import annotations

import os
from typing import List

from vtuber_engine.models.data_models import CharacterConfig

try:
    import yaml
except ImportError:
    yaml = None


def load_character_config(config_path: str) -> CharacterConfig:
    """
    从 YAML 加载角色配置。

    Args:
        config_path: YAML 配置文件路径。

    Returns:
        CharacterConfig 实例。
    """
    if yaml is None:
        raise ImportError("pyyaml is required. Install with: pip install pyyaml")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    char_data = data.get("character", data)

    return CharacterConfig(
        name=char_data.get("name", "unnamed"),
        resolution=tuple(char_data.get("resolution", [1080, 1920])),
        emotions=char_data.get("emotions", ["calm", "excited", "panic"]),
        mouth_threshold=char_data.get("mouth_threshold", 0.5),
        blink_interval=char_data.get("blink_interval", 3.0),
        blink_duration=char_data.get("blink_duration", 0.15),
    )


def create_default_config() -> CharacterConfig:
    """创建默认角色配置（不需要 YAML 文件）。"""
    return CharacterConfig(
        name="my_oc",
        resolution=(1080, 1920),
        emotions=["calm", "excited", "panic"],
        mouth_threshold=0.5,
        blink_interval=3.0,
        blink_duration=0.15,
    )
