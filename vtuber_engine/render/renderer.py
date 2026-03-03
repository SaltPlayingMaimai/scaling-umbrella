"""
Renderer — 图像合成模块。

职责：
  - 按 AnimatedState 选取对应的角色图片（4 张组合之一）
  - 绿幕背景合成
  - 帧缓存（相同状态不重复合成）

素材体系（每个表情状态 4 张图）：
  {emotion}_eo_mo  —  眼开 + 嘴开
  {emotion}_eo_mc  —  眼开 + 嘴闭
  {emotion}_ec_mo  —  眼闭 + 嘴开
  {emotion}_ec_mc  —  眼闭 + 嘴闭

不做：
  - 状态决策（交给 State Engine）
  - 视频编码（交给 Video Exporter）
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

from vtuber_engine.models.data_models import (
    AnimatedState,
    CharacterConfig,
    RenderFrame,
    UploadedAssets,
)

try:
    from PIL import Image
except ImportError:
    Image = None


# 绿幕颜色
CHROMA_GREEN = (0, 255, 0, 255)


class Renderer:
    """根据动画状态 + 上传素材合成帧图像。"""

    def __init__(
        self,
        character_config: CharacterConfig,
        assets: UploadedAssets,
    ):
        """
        Args:
            character_config: 角色配置。
            assets: 用户上传的素材（PIL Image 字典）。
        """
        if Image is None:
            raise ImportError(
                "Pillow is required for rendering. "
                "Install it with: pip install Pillow"
            )
        self.config = character_config
        self.assets = assets
        self.resolution = character_config.resolution  # (width, height)

        # 帧缓存：state_hash → Image
        self._frame_cache: Dict[str, Image.Image] = {}

    # ──────────────────── 公共接口 ────────────────────

    def render_frame(self, state: AnimatedState) -> Image.Image:
        """
        根据动画状态合成单帧（绿幕背景）。

        Args:
            state: 插值后的角色状态。

        Returns:
            RGBA Image。
        """
        # 检查帧缓存
        state_hash = self._hash_state(state)
        if state_hash in self._frame_cache:
            return self._frame_cache[state_hash]

        # 确定该帧使用哪张图
        render_frame = self._resolve_image(state)

        # 合成
        canvas = self._compose(render_frame)

        # 缓存
        self._frame_cache[state_hash] = canvas
        return canvas

    def render_sequence(
        self, states: list[AnimatedState], progress_callback=None
    ) -> list[Image.Image]:
        """
        渲染完整帧序列。

        Args:
            states: AnimatedState 列表。
            progress_callback: 可选回调 fn(current, total)。

        Returns:
            Image 列表。
        """
        frames = []
        total = len(states)
        for i, state in enumerate(states):
            frame = self.render_frame(state)
            frames.append(frame)
            if progress_callback and i % 30 == 0:
                progress_callback(i, total)
        return frames

    # ──────────────────── 图片选择 ────────────────────

    def _resolve_image(self, state: AnimatedState) -> RenderFrame:
        """
        将 AnimatedState 映射为具体的图片键名。

        逻辑：
          - mouth_open > threshold → 嘴开(mo)，否则嘴闭(mc)
          - blink_phase > 0.5 → 眼闭(ec)，否则眼开(eo)
        """
        eye_open = state.blink_phase <= 0.5
        mouth_open = state.mouth_open > self.config.mouth_threshold

        key = CharacterConfig.image_key(state.emotion, eye_open, mouth_open)

        # 如果该情绪没有对应素材，回退到 calm
        if not self.assets.has(key):
            key = CharacterConfig.image_key("calm", eye_open, mouth_open)

        return RenderFrame(image_key=key)

    # ──────────────────── 合成 ────────────────────

    def _compose(self, render_frame: RenderFrame) -> Image.Image:
        """在绿幕上放置角色图片。"""
        w, h = self.resolution

        # 绿幕背景
        canvas = Image.new("RGBA", (w, h), CHROMA_GREEN)

        # 获取角色图片
        char_img = self.assets.get(render_frame.image_key)
        if char_img is None:
            return canvas  # 没有素材时返回纯绿幕

        # 确保 RGBA
        if char_img.mode != "RGBA":
            char_img = char_img.convert("RGBA")

        # 居中放置
        x = (w - char_img.width) // 2
        y = (h - char_img.height) // 2
        canvas.paste(char_img, (x, y), mask=char_img)

        return canvas

    # ──────────────────── 缓存 ────────────────────

    def _hash_state(self, state: AnimatedState) -> str:
        """生成状态的哈希，用于帧缓存。"""
        eye_open = state.blink_phase <= 0.5
        mouth_open = state.mouth_open > self.config.mouth_threshold
        key = (state.emotion, eye_open, mouth_open)
        return hashlib.md5(str(key).encode()).hexdigest()

    def clear_cache(self) -> None:
        """清空帧缓存。"""
        self._frame_cache.clear()
