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

from typing import List

from vtuber_engine.models.data_models import AnimatedState, CharacterState


class AnimationEngine:
    """对角色状态参数做插值平滑。"""

    def __init__(self, smoothing: float = 0.2):
        """
        Args:
            smoothing: 插值速度 (0~1)。
                       0.1 = 非常平滑（慢响应）
                       0.5 = 较快响应
                       1.0 = 无平滑（直接跳变）
        """
        self.smoothing = smoothing
        self._current = AnimatedState()

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

        # 返回快照
        return AnimatedState(
            emotion=self._current.emotion,
            energy=round(self._current.energy, 4),
            mouth_open=round(self._current.mouth_open, 4),
            blink_phase=round(self._current.blink_phase, 4),
            gesture=self._current.gesture,
            expression_weights=dict(self._current.expression_weights),
        )

    def _mouth_smoothing(self, target: CharacterState) -> float:
        """
        嘴型使用动态平滑：
        - 张嘴（说话开始）响应快
        - 闭嘴（说话结束）稍慢，避免抖动
        """
        if target.mouth_open > self._current.mouth_open:
            return min(1.0, self.smoothing * 2.0)  # 张嘴快
        else:
            return self.smoothing * 0.8  # 闭嘴慢

    @staticmethod
    def _lerp(current: float, target: float, speed: float) -> float:
        """线性插值。"""
        return current + (target - current) * speed
