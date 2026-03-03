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
        bounce_frequency: float = 3.0,
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

        # 讲话跳动
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
        计算讲话时的弹性跳动偏移。

        口型打开时（speaking）驱动跳动动画；
        口型关闭时平滑衰减回 0。
        """
        if not self._bounce_enabled:
            return 0.0

        if mouth_open > 0.01:  # 正在讲话
            self._bounce_timer += self._dt
            t = self._bounce_timer * self._bounce_frequency
            # 主跳动: abs(sin) 产生类似弹球的上下运动
            base = abs(math.sin(math.pi * t))
            # 弹性副振荡: 叠加高频微振让跳动感更有弹性
            elastic = 1.0 + 0.15 * math.sin(2 * math.pi * t * 3)
            bounce = max(0.0, base * elastic) * self._bounce_amplitude
            self._current_bounce = bounce
        else:
            # 不讲话时平滑衰减
            self._current_bounce *= 0.85
            if abs(self._current_bounce) < 0.5:
                self._current_bounce = 0.0
                self._bounce_timer = 0.0
            bounce = self._current_bounce

        return bounce
