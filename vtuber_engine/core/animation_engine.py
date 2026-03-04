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

        非对称弹跳节奏（基于 60fps 12+8+8 帧设计，比例自适应 fps）：
          ┌─ 上升阶段 (占 5/7 ≈ 71% 周期) ──────────────────┐
          │  前段(12/28): 起跳+拉高，稍快                    │
          │  后段(8/28) : 拉高+起跳，自然减速                │
          └──────────────────────────────────────────────────┘
          ┌─ 下降阶段 (占 2/7 ≈ 29% 周期) ──────────────────┐
          │  压扁+降落，稍快但不闪现                         │
          └──────────────────────────────────────────────────┘

        Squash-stretch 基于"速度"而非"高度"：
          • 上升中（vel > 0）→ ss < 0 → 压扁（矮宽）→ "向上压扁"感
          • 最高点 (vel = 0) → ss = 0 → 中性
          • 下降中（vel < 0）→ ss > 0 → 拉伸（高瘦）
          • 着地点 (vel = 0) → ss = 0 → 中性（初始状态不被压扁）
        周期边界处速度均为 0，无闪现跳变。
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
            # 上升 5/7 周期（对应 60fps 下 20 帧），下降 2/7 周期（8 帧）
            # 两段各用半余弦缓动 → 地面/峰值处导数均为 0 → C¹ 平滑
            RISE = 5.0 / 7.0  # 上升占比 71.4%

            if t_cycle < RISE:
                # 上升阶段：余弦缓入缓出
                phase = t_cycle / RISE  # 0 → 1
                height = 0.5 * (1.0 - math.cos(math.pi * phase))
                # 归一化速度：sin(π·phase)，在 phase=0 和 1 处为 0
                vel = math.sin(math.pi * phase)  # 正值 = 上升
            else:
                # 下降阶段：余弦缓入缓出
                phase = (t_cycle - RISE) / (1.0 - RISE)  # 0 → 1
                height = 0.5 * (1.0 + math.cos(math.pi * phase))
                # 归一化速度：-sin(π·phase)，负值 = 下降
                vel = -math.sin(math.pi * phase)

            bounce = height * self._bounce_amplitude
            self._current_bounce = bounce

            # ── 形变：跟随速度（细微果冻感） ──
            # vel > 0 (上升) → ss < 0 → 压扁(矮宽) → "向上压扁"
            # vel = 0 (峰值/地面) → ss = 0 → 中性
            # vel < 0 (下降) → ss > 0 → 拉伸(高瘦)
            # 幅度 0.20 = 细微差别，只是有果冻感而不夸张
            self._current_squash_stretch = -vel * 0.20

        else:
            # 停止讲话：平滑衰减回静止
            self._current_bounce *= 0.85
            self._current_squash_stretch *= 0.85
            if abs(self._current_bounce) < 0.5:
                self._current_bounce = 0.0
                self._bounce_timer = 0.0
                self._current_squash_stretch = 0.0

        return self._current_bounce
