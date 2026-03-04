"""
Character Store — 角色持久化存储模块。

将已注册的角色（CharacterConfig + 差分图）保存到用户的 Documents 目录，
重启软件后可一键恢复，无需重新上传图片和重跑 AI 识别。

存储路径（Windows）：
  ~/Documents/VTuber Engine/characters/{character_name}/
      config.json          — CharacterConfig 序列化（JSON）
      images/
          calm_eo_mo.png   — 差分图（PNG，带透明通道）
          calm_eo_mc.png
          ...

跨平台路径策略：
  Windows / macOS : ~/Documents/VTuber Engine/
  Linux           : ~/.local/share/VTuber Engine/
"""

from __future__ import annotations

import io
import json
import platform
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

from vtuber_engine.models.data_models import CharacterConfig, UploadedAssets


# ──────────────────── 路径工具 ────────────────────


def get_app_data_dir() -> Path:
    """
    返回 VTuber Engine 的用户数据根目录。

    - Windows / macOS : ~/Documents/VTuber Engine
    - Linux           : ~/.local/share/VTuber Engine
    """
    system = platform.system()
    if system in ("Windows", "Darwin"):
        base = Path.home() / "Documents"
    else:
        base = Path.home() / ".local" / "share"
    # 使用项目名作为用户数据目录，便于识别与管理
    return base / "scaling-umbrella"


def get_characters_dir() -> Path:
    """返回角色存储根目录，自动创建。"""
    d = get_app_data_dir() / "characters"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_dirname(name: str) -> str:
    """将角色名转为安全的目录名（去除非法字符）。"""
    # 保留字母、数字、空格、横线、下划线
    safe = re.sub(r"[^\w\s\-]", "_", name, flags=re.UNICODE).strip()
    return safe or "unnamed"


def get_character_dir(name: str) -> Path:
    """返回指定角色的存储目录（不一定存在）。"""
    return get_characters_dir() / _safe_dirname(name)


# ──────────────────── 枚举已保存角色 ────────────────────


def list_saved_characters() -> List[str]:
    """
    返回所有已保存的角色名列表（按修改时间倒序，最新在前）。

    读取每个子目录下 config.json 中的 name 字段。
    """
    root = get_characters_dir()
    results: List[Tuple[float, str]] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            char_name = data.get("name", d.name)
        except Exception:
            char_name = d.name
        mtime = cfg_path.stat().st_mtime
        results.append((mtime, char_name))

    results.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in results]


# ──────────────────── 保存 ────────────────────


def save_character(
    config: CharacterConfig,
    assets: UploadedAssets,
    overwrite: bool = True,
) -> Path:
    """
    将角色配置 + 差分图持久化到磁盘。

    Args:
        config  : CharacterConfig（含 emotions, emotion_vectors 等）
        assets  : UploadedAssets（含 PIL Image 字典）
        overwrite: 若目录已存在是否覆盖（默认 True）

    Returns:
        保存目录的 Path。

    Raises:
        FileExistsError: overwrite=False 且目录已存在时。
        RuntimeError   : Pillow 未安装。
    """
    if Image is None:
        raise RuntimeError("Pillow 未安装，无法保存图片。pip install Pillow")

    char_dir = get_character_dir(config.name)

    if char_dir.exists() and not overwrite:
        raise FileExistsError(f"角色目录已存在：{char_dir}")

    # 确保目录存在
    img_dir = char_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # ── 序列化 CharacterConfig ──
    cfg_dict = {
        "name": config.name,
        "resolution": list(config.resolution),  # tuple → list（JSON 兼容）
        "emotions": config.emotions,
        "emotion_vectors": config.emotion_vectors,
        "mouth_threshold": config.mouth_threshold,
        "blink_interval": config.blink_interval,
        "blink_duration": config.blink_duration,
        "bounce_enabled": config.bounce_enabled,
        "bounce_frequency": config.bounce_frequency,
        "bounce_amplitude": config.bounce_amplitude,
    }
    cfg_path = char_dir / "config.json"
    cfg_path.write_text(
        json.dumps(cfg_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── 保存每张差分图 ──
    saved_keys: List[str] = []
    missing_keys: List[str] = []
    for key in config.all_image_keys():
        img = assets.get(key)
        if img is None:
            missing_keys.append(key)
            continue
        # 确保 RGBA
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        out_path = img_dir / f"{key}.png"
        img.save(str(out_path), format="PNG")
        saved_keys.append(key)

    print(
        f"[CharacterStore] save_character: '{config.name}' → {char_dir}\n"
        f"  saved {len(saved_keys)} images, missing {len(missing_keys)}: {missing_keys}"
    )
    return char_dir


# ──────────────────── 载入 ────────────────────


def load_character(name: str) -> Tuple[CharacterConfig, UploadedAssets]:
    """
    从磁盘载入角色配置 + 差分图。

    Args:
        name: 角色名（与保存时一致）。

    Returns:
        (CharacterConfig, UploadedAssets)

    Raises:
        FileNotFoundError: 角色目录或 config.json 不存在。
    """
    if Image is None:
        raise RuntimeError("Pillow 未安装，无法加载图片。pip install Pillow")

    char_dir = get_character_dir(name)
    cfg_path = char_dir / "config.json"

    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到角色配置：{cfg_path}")

    # ── 反序列化 CharacterConfig ──
    data = json.loads(cfg_path.read_text(encoding="utf-8"))

    config = CharacterConfig(
        name=data.get("name", name),
        resolution=tuple(data.get("resolution", [1080, 1920])),
        emotions=data.get("emotions", []),
        emotion_vectors=data.get("emotion_vectors", {}),
        mouth_threshold=data.get("mouth_threshold", 0.5),
        blink_interval=data.get("blink_interval", 3.0),
        blink_duration=data.get("blink_duration", 0.15),
        bounce_enabled=data.get("bounce_enabled", True),
        bounce_frequency=data.get("bounce_frequency", 1.0),
        bounce_amplitude=data.get("bounce_amplitude", 8.0),
    )

    # ── 载入差分图 ──
    assets = UploadedAssets()
    img_dir = char_dir / "images"
    loaded: List[str] = []
    missing: List[str] = []

    for key in config.all_image_keys():
        img_path = img_dir / f"{key}.png"
        if img_path.exists():
            img = Image.open(str(img_path)).convert("RGBA")
            assets.put(key, img)
            loaded.append(key)
        else:
            missing.append(key)

    print(
        f"[CharacterStore] load_character: '{name}' ← {char_dir}\n"
        f"  loaded {len(loaded)} images, missing {len(missing)}: {missing}"
    )
    return config, assets


# ──────────────────── 删除 ────────────────────


def delete_character(name: str) -> bool:
    """
    删除已保存的角色目录。

    Returns:
        True=成功删除，False=目录不存在。
    """
    char_dir = get_character_dir(name)
    if char_dir.exists():
        shutil.rmtree(char_dir)
        print(f"[CharacterStore] delete_character: '{name}' 已删除")
        return True
    return False
