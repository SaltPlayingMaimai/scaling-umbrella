"""
Animation Engine — 动画参数插值平滑层。

职责：
  - 接收 CharacterState 序列
  - 对所有数值参数做 lerp 插值
  - 输出 AnimatedState 序列（平滑后，供渲染器使用）

不做：
  - 状态决策（交给 State Engine）
  - 渲染（交给 Renderer）

核心原则：
  所有数值变化必须平滑过渡，禁止跳变。
"""

from __future__ import annotations

import math
from typing import List

from vtuber_engine.models.data_models import AnimatedState, CharacterState


class AnimationEngine:
    """对角色状态参数做插值平滑。"""

    def __init__(
        self,
        smoothing: float = 0.2,
        fps: int = 30,
        bounce_enabled: bool = True,
        bounce_frequency: float = 1.0,
        bounce_amplitude: float = 8.0,
    ):
        """
        Args:
            smoothing: 插值速度 (0~1)。
                       0.1 = 非常平滑（慢响应）
                       0.5 = 较快响应
                       1.0 = 无平滑（直接跳变）
            fps: 帧率，用于计算跳动时间。
            bounce_enabled: 是否启用讲话跳动效果。
            bounce_frequency: 跳动频率 (Hz)。
            bounce_amplitude: 跳动幅度（像素）。
        """
        self.smoothing = smoothing
        self._current = AnimatedState()
        self._fps = fps
        self._dt = 1.0 / max(1, fps)

        # 跳动参数
        self._bounce_enabled = bounce_enabled
        self._bounce_frequency = bounce_frequency
        self._bounce_amplitude = bounce_amplitude
        self._bounce_timer: float = 0.0
        self._current_bounce: float = 0.0
        self._current_squash_stretch: float = 0.0  # 果冻形变系数

    # ──────────────────── 公共接口 ────────────────────

    def process(self, states: list[CharacterState]) -> list[AnimatedState]:
        """
        对整个状态序列做逐帧插值。

        Args:
            states: State Engine 输出的原始状态列表。

        Returns:
            平滑后的 AnimatedState 列表。
        """
        total = len(states)
        print(
            f"[AnimationEngine] process() start: total_frames={total}, smoothing={self.smoothing}"
        )

        results: list[AnimatedState] = []

        for i, state in enumerate(states):
            animated = self._interpolate_frame(state)
            results.append(animated)

            # 首/末帧 及每 30 帧打印一次
            if i == 0 or i == total - 1 or (i > 0 and i % 30 == 0):
                print(
                    f"[AnimationEngine] frame[{i:04d}] "
                    f"emotion='{animated.emotion}' "
                    f"mouth_open={animated.mouth_open:.4f} "
                    f"blink_phase={animated.blink_phase:.4f} "
                    f"energy={animated.energy:.4f} "
                    f"gesture={animated.gesture}"
                )

        print(f"[AnimationEngine] process() done: {len(results)} animated frames")
        return results

    # ──────────────────── 内部逻辑 ────────────────────

    def _interpolate_frame(self, target: CharacterState) -> AnimatedState:
        """对单帧做插值。"""

        # 数值参数 lerp
        self._current.energy = self._lerp(
            self._current.energy, target.energy, self.smoothing
        )
        self._current.mouth_open = self._lerp(
            self._current.mouth_open, target.mouth_open, self._mouth_smoothing(target)
        )

        # 眨眼不做平滑（本身已经是平滑曲线）
        self._current.blink_phase = target.blink_phase

        # 离散参数：直接赋值
        self._current.emotion = target.emotion
        self._current.gesture = target.gesture
        self._current.expression_weights = dict(target.expression_weights)

        # 讲话跳动 + 果冻形变
        self._current.bounce_offset = self._compute_bounce(target.mouth_open)

        # 返回快照
        return AnimatedState(
            emotion=self._current.emotion,
            energy=round(self._current.energy, 4),
            mouth_open=round(self._current.mouth_open, 4),
            blink_phase=round(self._current.blink_phase, 4),
            gesture=self._current.gesture,
            expression_weights=dict(self._current.expression_weights),
            bounce_offset=round(self._current.bounce_offset, 2),
            squash_stretch=round(self._current_squash_stretch, 4),
        )

    def _mouth_smoothing(self, target: CharacterState) -> float:
        """
        嘴型使用动态平滑：
        - 张嘴（说话开始）响应适中
        - 闭嘴（说话结束）较慢，避免抖动
        """
        if target.mouth_open > self._current.mouth_open:
            return min(0.6, self.smoothing * 1.5)  # 张嘴：适度快
        else:
            return self.smoothing * 0.5  # 闭嘴：更慢，避免抖动

    @staticmethod
    def _lerp(current: float, target: float, speed: float) -> float:
        """线性插值。"""
        return current + (target - current) * speed

    # ──────────────────── 讲话跳动 ────────────────────

    def _compute_bounce(self, mouth_open: float) -> float:
        """
        计算讲话时的弹跳偏移 + squash-stretch 系数。

        非对称弹跳节奏（每个周期内）：
          ┌─ 上升阶段 (占 5/7 ≈ 71% 周期) ──────────────────┐
          │  前段（12/28）: 起跳+拉高，稍快                   │
          │  后段（8/28） : 接近峰值+最大拉伸，自然减速       │
          └──────────────────────────────────────────────────┘
          ┌─ 下降阶段 (占 2/7 ≈ 29% 周期) ──────────────────┐
          │  压扁+快速降落（8/28）                            │
          └──────────────────────────────────────────────────┘

        实现：非对称余弦缓动（两段半余弦拼接）。
        上升/峰值/下降/着地四个关键点速度均为 0 → 全程 C¹ 平滑。
        形变 squash-stretch 直接跟随 height，上下左右互补配合。
        """
        if not self._bounce_enabled:
            self._current_squash_stretch = 0.0
            return 0.0

        if mouth_open > 0.01:  # 正在讲话
            self._bounce_timer += self._dt
            t = self._bounce_timer * self._bounce_frequency

            # 周期内位置 [0, 1)
            t_cycle = t % 1.0

            # ── 非对称弹跳曲线 ──
            # 上升占 5/7 周期（慢起跳，缓到顶），下降占 2/7 周期（快落地）
            # 两段各用半余弦缓动 → 地面/峰值处导数均为 0 → C¹ 无缝衔接
            RISE = 5.0 / 7.0  # 上升占比

            if t_cycle < RISE:
                # 上升阶段：余弦缓入缓出
                # phase 0→1 映射到 height 0→1
                # 前 60% 上升较快（起跳+拉高），后 40% 自然减速（接近峰值）
                phase = t_cycle / RISE
                height = 0.5 * (1.0 - math.cos(math.pi * phase))
            else:
                # 下降阶段：余弦快降
                # phase 0→1 映射到 height 1→0
                phase = (t_cycle - RISE) / (1.0 - RISE)
                height = 0.5 * (1.0 + math.cos(math.pi * phase))

            bounce = height * self._bounce_amplitude
            self._current_bounce = bounce

            # ── 形变：直接跟随高度 ──
            # 着地 height=0 → ss=-0.5 → 矮胖（横向拉宽 + 纵向压扁）
            # 最高 height=1 → ss=+0.5 → 高瘦（纵向拉长 + 横向收窄）
            # 上下左右互补配合，与弹跳完全同步
            self._current_squash_stretch = (2.0 * height - 1.0) * 0.5

        else:
            # 停止讲话：平滑衰减回静止
            self._current_bounce *= 0.85
            self._current_squash_stretch *= 0.85
            if abs(self._current_bounce) < 0.5:
                self._current_bounce = 0.0
                self._bounce_timer = 0.0
                self._current_squash_stretch = 0.0

        return self._current_bounce
