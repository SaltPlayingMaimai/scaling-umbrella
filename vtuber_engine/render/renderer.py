"""
Renderer — 图像合成模块。

职责：
  - 按 AnimatedState 选取对应的角色图片（4 张组合之一）
  - 绿幕背景合成
  - 帧缓存（相同状态不重复合成）
  - 并行渲染（ThreadPoolExecutor）

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
from concurrent.futures import ThreadPoolExecutor
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

try:
    import numpy as np
except ImportError:
    np = None


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

        # 预渲染绿幕背景（numpy 或 PIL，复用避免重复创建）
        w, h = self.resolution
        if np is not None:
            self._green_bg_np = np.full((h, w, 4), CHROMA_GREEN, dtype=np.uint8)
        else:
            self._green_bg_np = None
        self._green_bg_pil = Image.new("RGBA", (w, h), CHROMA_GREEN)

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

        优化策略：
          1. 先去重：提取所有不同 state_hash，只渲染唯一帧
          2. 再映射：按原始顺序组装帧列表（大量连续相同帧零开销）

        Args:
            states: AnimatedState 列表。
            progress_callback: 可选回调 fn(current, total)。

        Returns:
            Image 列表。
        """
        total = len(states)

        # 第一步：去重 — 收集所有不同的 (hash, state) 对
        hash_to_state: Dict[str, AnimatedState] = {}
        hashes: list[str] = []
        for state in states:
            h = self._hash_state(state)
            hashes.append(h)
            if h not in hash_to_state and h not in self._frame_cache:
                hash_to_state[h] = state

        unique_count = len(hash_to_state)
        print(
            f"[Renderer] render_sequence: {total} 帧, "
            f"{unique_count} 个独立帧需要渲染 "
            f"(缓存命中 {total - unique_count - len([h for h in set(hashes) if h in self._frame_cache])} 帧)"
        )

        # 第二步：渲染所有唯一帧
        for i, (h, state) in enumerate(hash_to_state.items()):
            self._frame_cache[h] = self._render_single(state)

        # 第三步：按顺序组装
        frames = []
        for i, h in enumerate(hashes):
            frames.append(self._frame_cache[h])
            if progress_callback and i % 30 == 0:
                progress_callback(i, total)

        if progress_callback:
            progress_callback(total, total)

        return frames

    def _render_single(self, state: AnimatedState) -> Image.Image:
        """渲染单个独立帧（内部使用，不检查缓存）。"""
        render_frame = self._resolve_image(state)
        return self._compose(render_frame)

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

        # 如果该情绪没有对应素材，回退到第一个可用表情
        if not self.assets.has(key):
            fallback_emotion = (
                self.config.emotions[0] if self.config.emotions else state.emotion
            )
            key = CharacterConfig.image_key(fallback_emotion, eye_open, mouth_open)

        return RenderFrame(image_key=key)

    # ──────────────────── 合成 ────────────────────

    def _compose(self, render_frame: RenderFrame) -> Image.Image:
        """在绿幕上放置角色图片。使用 numpy 加速 alpha 合成（如果可用）。"""
        w, h = self.resolution

        # 获取角色图片
        char_img = self.assets.get(render_frame.image_key)
        if char_img is None:
            return self._green_bg_pil.copy()  # 没有素材时返回纯绿幕

        # 确保 RGBA
        if char_img.mode != "RGBA":
            char_img = char_img.convert("RGBA")

        # numpy 加速路径
        if np is not None and self._green_bg_np is not None:
            canvas_np = self._green_bg_np.copy()
            char_np = np.asarray(char_img)

            # 居中放置
            x = (w - char_img.width) // 2
            y = (h - char_img.height) // 2

            # 裁剪（防止超出画布）
            src_x1 = max(0, -x)
            src_y1 = max(0, -y)
            src_x2 = min(char_img.width, w - x)
            src_y2 = min(char_img.height, h - y)
            dst_x1 = max(0, x)
            dst_y1 = max(0, y)
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)

            if dst_x2 > dst_x1 and dst_y2 > dst_y1:
                src_region = char_np[src_y1:src_y2, src_x1:src_x2]
                alpha = src_region[:, :, 3:4].astype(np.float32) / 255.0
                dst_region = canvas_np[dst_y1:dst_y2, dst_x1:dst_x2]
                # Alpha 混合
                blended = (
                    src_region[:, :, :3].astype(np.float32) * alpha
                    + dst_region[:, :, :3].astype(np.float32) * (1.0 - alpha)
                ).astype(np.uint8)
                canvas_np[dst_y1:dst_y2, dst_x1:dst_x2, :3] = blended
                # alpha 通道取最大值
                canvas_np[dst_y1:dst_y2, dst_x1:dst_x2, 3] = np.maximum(
                    dst_region[:, :, 3], src_region[:, :, 3]
                )

            return Image.fromarray(canvas_np, "RGBA")

        # PIL 回退路径
        canvas = self._green_bg_pil.copy()
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
