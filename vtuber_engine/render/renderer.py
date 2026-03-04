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
from typing import Dict, Generator, Optional

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
        # 检查帧缓存（基础帧，不含跳动偏移）
        state_hash = self._hash_state(state)
        if state_hash in self._frame_cache:
            base = self._frame_cache[state_hash]
        else:
            # 确定该帧使用哪张图
            render_frame = self._resolve_image(state)
            # 合成
            base = self._compose(render_frame)
            # 缓存
            self._frame_cache[state_hash] = base

        # 应用跳动 + 形变
        bounce_px = int(round(getattr(state, "bounce_offset", 0.0)))
        squash_stretch = float(getattr(state, "squash_stretch", 0.0))
        if bounce_px != 0 or abs(squash_stretch) > 0.01:
            return self._apply_jelly_bounce(state, bounce_px, squash_stretch)
        return base

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

        # 第三步：按顺序组装（并应用果冻跳动）
        frames = []
        for i, h in enumerate(hashes):
            base = self._frame_cache[h]
            bounce_px = int(round(getattr(states[i], "bounce_offset", 0.0)))
            squash_stretch = float(getattr(states[i], "squash_stretch", 0.0))
            if bounce_px != 0 or abs(squash_stretch) > 0.01:
                frames.append(
                    self._apply_jelly_bounce(states[i], bounce_px, squash_stretch)
                )
            else:
                frames.append(base)
            if progress_callback and i % 30 == 0:
                progress_callback(i, total)

        if progress_callback:
            progress_callback(total, total)

        return frames

    def render_sequence_streaming(
        self, states: list[AnimatedState], progress_callback=None
    ) -> Generator:
        """
        流式渲染帧序列（生成器）。

        与 render_sequence 不同，不会将所有帧存入内存列表，
        老每次只 yield 一帧，由调用方（导出器）立即消费并释放。
        内存占用降低为 O(唯一帧数) 而非 O(总帧数)。

        Yields:
            PIL Image 帧。
        """
        total = len(states)

        # 预渲染唯一基础帧（缓存）
        for state in states:
            h = self._hash_state(state)
            if h not in self._frame_cache:
                self._frame_cache[h] = self._render_single(state)

        # 逐帧 yield，调用方消费后即可 GC
        for i, state in enumerate(states):
            h = self._hash_state(state)
            base = self._frame_cache[h]
            bounce_px = int(round(getattr(state, "bounce_offset", 0.0)))
            squash_stretch = float(getattr(state, "squash_stretch", 0.0))
            if bounce_px != 0 or abs(squash_stretch) > 0.01:
                yield self._apply_jelly_bounce(state, bounce_px, squash_stretch)
            else:
                yield base
            if progress_callback and i % 30 == 0:
                progress_callback(i, total)

        if progress_callback:
            progress_callback(total, total)

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

    # ────────────────── 内部工具 ──────────────────

    def _get_headroom(self, char_h: int) -> int:
        """
        计算预留的顶部余量（角色向下偏移量）。
        余量 = 最大跳动偏移 + 最大拉伸高度，保证画面内不截角色顶部。
        """
        if not getattr(self.config, "bounce_enabled", False):
            return 0
        max_bounce = int(self.config.bounce_amplitude)
        max_stretch = int(char_h * 0.18)  # 18% 纵向拉伸最大値
        return max_bounce + max_stretch

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

            # 居中放置；向下偏移 headroom，为跳动+拉伸预留顶部空间
            bounce_headroom = self._get_headroom(char_img.height)
            x = (w - char_img.width) // 2
            y = (h - char_img.height) // 2 + bounce_headroom

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
        bounce_headroom = self._get_headroom(char_img.height)
        x = (w - char_img.width) // 2
        y = (h - char_img.height) // 2 + bounce_headroom
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

    # ────────────────── Q弹跳动应用 ──────────────────

    def _apply_jelly_bounce(
        self, state: AnimatedState, bounce_px: int, squash_stretch: float
    ) -> Image.Image:
        """
        只对角色贴图本身做 squash-stretch 缩放，再合成到绳幕。

        设计原则：
          - 不缩放整块画布（避免截图）—只缩放角色贴图自身
          - 脚底霨点固定（与 _compose 基准位置一致）
          - bounce_px 负责億直轻跳，squash_stretch 负责形变
          - 贴图如超出画布上边为自然渗出（不切剪贴图本身）
        """
        w, h = self.resolution

        # 获取角色贴图
        rf = self._resolve_image(state)
        char_img = self.assets.get(rf.image_key)
        if char_img is None:
            return self._green_bg_pil.copy()
        if char_img.mode != "RGBA":
            char_img = char_img.convert("RGBA")

        char_w, char_h = char_img.width, char_img.height

        # ── 形变比例（上下 & 左右互补配合，细微果冻感） ──
        # ss > 0 → 高瘦（纵向拉伸 + 横向收窄）
        # ss < 0 → 矮胖（纵向压扁 + 横向拉宽） → “向上压扁”
        ss = max(-0.3, min(0.3, squash_stretch))
        _fy = float(getattr(self.config, 'squash_stretch_factor', 0.1))
        _fx = float(getattr(self.config, 'squash_stretch_factor_x', 0.15))
        sy = max(0.7, min(1.3, 1.0 + _fy * ss))  # 纵向（可配置）
        sx = max(0.7, min(1.3, 1.0 - _fx * ss))  # 横向（可配置）

        if abs(sy - 1.0) > 0.005 or abs(sx - 1.0) > 0.005:
            new_w = max(1, int(round(char_w * sx)))
            new_h = max(1, int(round(char_h * sy)))
            scaled = char_img.resize((new_w, new_h), Image.BILINEAR)
        else:
            scaled = char_img
            new_w, new_h = char_w, char_h

        # ── 堑算脚底霨点 Y（与 _compose 居中+headroom 公式一致） ──
        headroom = self._get_headroom(char_h)
        # _compose 中： top_y = (h - char_h)//2 + headroom
        # 脚底 = top_y + char_h = (h + char_h)//2 + headroom
        foot_y = (h + char_h) // 2 + headroom

        # 脚底固定，向上跳动
        paste_x = (w - new_w) // 2
        paste_y = foot_y - new_h - bounce_px

        # 在绳幕上合成（PIL paste 自然处理画布边缘，贴图本身不内截剪）
        canvas = Image.new("RGBA", (w, h), CHROMA_GREEN)
        canvas.paste(scaled, (paste_x, paste_y), mask=scaled)
        return canvas
