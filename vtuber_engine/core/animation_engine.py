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
  所有 lerp/衰减操作必须帧率无关（通过 _adjust_speed / _adjust_decay 转换）。
"""

from __future__ import annotations

import math
import random
from typing import List

from vtuber_engine.models.data_models import AnimatedState, CharacterState

# 参考帧率：所有插值/衰减参数均以 30fps 为基准设计
_REFERENCE_FPS = 30
_REFERENCE_DT = 1.0 / _REFERENCE_FPS

# ── 灵动模式状态 ──
_LIVELY_IDLE = 0       # 空闲（未说话或衰减中）
_LIVELY_BURST = 1      # 连跳阶段
_LIVELY_COOLDOWN = 2   # 冷静期


class AnimationEngine:
    """对角色状态参数做插值平滑。"""

    def __init__(
        self,
        smoothing: float = 0.2,
        fps: int = 30,
        bounce_enabled: bool = True,
        bounce_frequency: float = 1.0,
        bounce_amplitude: float = 30.0,
        squash_stretch_factor: float = 0.1,
        bounce_lively_mode: bool = False,
        lively_burst_min: int = 2,
        lively_burst_max: int = 5,
        lively_cooldown_min: float = 1.5,
        lively_cooldown_max: float = 3.0,
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
            squash_stretch_factor: 果冻形变强度（0=无形变，0.3=夸张）。
            bounce_lively_mode: 是否启用灵动模式（连跳+冷静交替）。
            lively_burst_min: 灵动模式中每次连跳的最少次数。
            lively_burst_max: 灵动模式中每次连跳的最多次数。
            lively_cooldown_min: 灵动模式中冷静期最短秒数。
            lively_cooldown_max: 灵动模式中冷静期最长秒数。
        """
        self.smoothing = smoothing
        self._current = AnimatedState()
        self._fps = fps
        self._dt = 1.0 / max(1, fps)

        # 跳动参数
        self._bounce_enabled = bounce_enabled
        self._bounce_frequency = bounce_frequency
        self._bounce_amplitude = bounce_amplitude
        self._squash_stretch_factor = squash_stretch_factor
        self._bounce_timer: float = 0.0
        self._current_bounce: float = 0.0
        self._current_squash_stretch: float = 0.0  # 果冻形变系数

        # ── 灵动模式参数 ──
        self._lively_mode = bounce_lively_mode
        self._lively_burst_min = lively_burst_min
        self._lively_burst_max = lively_burst_max
        self._lively_cooldown_min = lively_cooldown_min
        self._lively_cooldown_max = lively_cooldown_max
        # 灵动模式状态
        self._lively_state: int = _LIVELY_IDLE
        self._lively_burst_total: int = 0       # 本次连跳总数
        self._lively_cooldown_timer: float = 0.0
        self._lively_cooldown_duration: float = 0.0
        self._lively_was_speaking: bool = False  # 上一帧是否在说话

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

        # 数值参数 lerp（帧率适配后的速度）
        energy_speed = self._adjust_speed(self.smoothing)
        self._current.energy = self._lerp(
            self._current.energy, target.energy, energy_speed
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
        # 使用平滑后的 mouth_open 驱动跳动，而非原始值。
        # 原始值在音节模型的开/闭相之间快速交替（~0.6 → 0 → 0.6），
        # 导致 bounce 每隔几帧就进入衰减分支 → 量化后产生重复帧 = 视觉卡顿。
        # 平滑值在持续说话期间始终 >0.05，保证跳动连续不中断。
        self._current.bounce_offset = self._compute_bounce(self._current.mouth_open)

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

    # ──────────────────── 帧率无关工具 ────────────────────

    def _adjust_speed(self, speed_at_ref: float) -> float:
        """
        将以 30fps 为基准设计的 lerp 速度适配到当前帧率。

        原理：lerp 每帧后残差 = (1-speed)，经过时间 T 后残差 = (1-speed)^(T/dt)。
        保持相同时间 T 内的收敛行为，对不同 dt 做指数校正。
        """
        if self._fps == _REFERENCE_FPS:
            return speed_at_ref
        ratio = self._dt / _REFERENCE_DT
        return 1.0 - (1.0 - speed_at_ref) ** ratio

    def _adjust_decay(self, decay_at_ref: float) -> float:
        """
        将以 30fps 为基准设计的每帧衰减因子适配到当前帧率。

        例：0.85 per frame @30fps → 0.85^0.5 ≈ 0.922 per frame @60fps。
        相同时间内总衰减量一致。
        """
        if self._fps == _REFERENCE_FPS:
            return decay_at_ref
        ratio = self._dt / _REFERENCE_DT
        return decay_at_ref ** ratio

    def _mouth_smoothing(self, target: CharacterState) -> float:
        """
        嘴型使用动态平滑：
        - 张嘴（说话开始）响应适中
        - 闭嘴（说话结束）较慢，避免抖动

        返回的速度自动适配当前帧率。
        """
        if target.mouth_open > self._current.mouth_open:
            base = min(0.6, self.smoothing * 1.5)  # 张嘴：适度快
        else:
            base = self.smoothing * 0.5  # 闭嘴：更慢，避免抖动
        return self._adjust_speed(base)

    @staticmethod
    def _lerp(current: float, target: float, speed: float) -> float:
        """线性插值。"""
        return current + (target - current) * speed

    # ──────────────────── 讲话跳动 ────────────────────

    def _compute_bounce(self, mouth_open: float) -> float:
        """
        计算讲话时的弹跳偏移 + squash-stretch 系数。

        支持两种模式：
          • 连续模式（默认）：说话时持续跳动。
          • 灵动模式（lively_mode）：说话开始时连跳 2~5 下，冷静一段时间后再跳。

        所有 lerp/衰减操作均经过帧率校正（_adjust_speed / _adjust_decay），
        确保 30fps / 60fps / 任意帧率下行为一致。
        """
        if not self._bounce_enabled:
            self._current_squash_stretch = 0.0
            return 0.0

        if self._lively_mode:
            return self._compute_bounce_lively(mouth_open)
        else:
            return self._compute_bounce_continuous(mouth_open)

    # ──────────────────── 连续跳动模式（原有逻辑 + 帧率修正） ────────────────────

    def _compute_bounce_continuous(self, mouth_open: float) -> float:
        """
        连续弹跳模式：说话时持续正弦跳动。

        非对称弹跳节奏：
          上升 5/7 周期（缓起+缓停），下降 2/7 周期。
          周期边界处导数为 0 → C¹ 平滑，无跳变。

        Squash-stretch 基于速度而非高度：
          • 上升 → 压扁（矮宽）  • 最高/最低点 → 中性  • 下降 → 拉伸（高瘦）
        """
        if mouth_open > 0.05:  # 正在讲话（基于平滑后 mouth_open）
            self._bounce_timer += self._dt
            t = self._bounce_timer * self._bounce_frequency

            height, vel = self._bounce_curve(t % 1.0)

            amplitude_scale = max(0.0, min(1.0, (mouth_open - 0.05) / 0.25))
            bounce_target = height * self._bounce_amplitude * amplitude_scale

            # 帧率无关 lerp：speed=0.35 @30fps
            chase_speed = self._adjust_speed(0.35)
            self._current_bounce += (bounce_target - self._current_bounce) * chase_speed

            # 形变（帧率无关：squash_stretch_factor 是比例系数，不含时间分量）
            self._current_squash_stretch = -vel * self._squash_stretch_factor * amplitude_scale

        else:
            self._decay_bounce()

        return self._current_bounce

    # ──────────────────── 灵动跳动模式（新增） ────────────────────

    def _compute_bounce_lively(self, mouth_open: float) -> float:
        """
        灵动弹跳模式：
          说话开始 → 连跳 2~5 下（burst）
          连跳结束 → 冷静 1.5~3.0 秒（cooldown）
          冷静结束 → 若仍在说话，再次连跳
          停止说话 → 平滑衰减到地面

        状态机：
          IDLE ──(开始说话)──→ BURST ──(跳完)──→ COOLDOWN ──(超时+仍在说话)──→ BURST
            ↑                    │                   │
            └───(停止说话)───────┴───(停止说话)──────┘
        """
        is_speaking = mouth_open > 0.05
        speech_just_started = is_speaking and not self._lively_was_speaking
        self._lively_was_speaking = is_speaking

        if self._lively_state == _LIVELY_IDLE:
            if speech_just_started or (is_speaking and abs(self._current_bounce) < 0.5):
                # 开始说话 → 启动一轮连跳
                self._start_lively_burst()
            else:
                self._decay_bounce()
            if self._lively_state == _LIVELY_IDLE:
                return self._current_bounce

        if self._lively_state == _LIVELY_BURST:
            if not is_speaking:
                # 说话停止 → 中断连跳，回到 IDLE 衰减
                self._lively_state = _LIVELY_IDLE
                self._decay_bounce()
                return self._current_bounce

            # 推进跳动计时
            self._bounce_timer += self._dt
            t = self._bounce_timer * self._bounce_frequency
            cycle_index = int(t)  # 当前处于第几跳（0-based）

            if cycle_index >= self._lively_burst_total:
                # 本轮连跳完成 → 进入冷静期
                self._lively_state = _LIVELY_COOLDOWN
                self._lively_cooldown_timer = 0.0
                self._lively_cooldown_duration = random.uniform(
                    self._lively_cooldown_min, self._lively_cooldown_max
                )
                # 平滑衰减到地面
                self._decay_bounce()
                return self._current_bounce

            # 计算当前帧弹跳
            t_cycle = t - cycle_index  # 当前跳内的相位 [0, 1)
            height, vel = self._bounce_curve(t_cycle)

            amplitude_scale = max(0.0, min(1.0, (mouth_open - 0.05) / 0.25))
            bounce_target = height * self._bounce_amplitude * amplitude_scale

            chase_speed = self._adjust_speed(0.35)
            self._current_bounce += (bounce_target - self._current_bounce) * chase_speed
            self._current_squash_stretch = -vel * self._squash_stretch_factor * amplitude_scale
            return self._current_bounce

        if self._lively_state == _LIVELY_COOLDOWN:
            if not is_speaking:
                # 说话停止 → 回到 IDLE
                self._lively_state = _LIVELY_IDLE
                self._decay_bounce()
                return self._current_bounce

            self._lively_cooldown_timer += self._dt
            if self._lively_cooldown_timer >= self._lively_cooldown_duration:
                # 冷静期结束 → 开始新一轮连跳
                self._start_lively_burst()
                return self._current_bounce

            # 冷静期间衰减到地面
            self._decay_bounce()
            return self._current_bounce

        return self._current_bounce

    def _start_lively_burst(self) -> None:
        """启动一轮灵动连跳。"""
        self._lively_state = _LIVELY_BURST
        self._lively_burst_total = random.randint(
            self._lively_burst_min, self._lively_burst_max
        )
        self._bounce_timer = 0.0
        print(
            f"[AnimationEngine] lively burst start: {self._lively_burst_total} bounces"
        )

    # ──────────────────── 共享工具 ────────────────────

    @staticmethod
    def _bounce_curve(t_cycle: float) -> tuple[float, float]:
        """
        非对称弹跳曲线：上升 5/7 + 下降 2/7 周期。

        Args:
            t_cycle: 周期内位置 [0, 1)。

        Returns:
            (height, velocity) —— height ∈ [0,1]，velocity ∈ [-1,1]。
        """
        RISE = 5.0 / 7.0  # 上升占比 71.4%

        if t_cycle < RISE:
            phase = t_cycle / RISE
            height = 0.5 * (1.0 - math.cos(math.pi * phase))
            vel = math.sin(math.pi * phase)
        else:
            phase = (t_cycle - RISE) / (1.0 - RISE)
            height = 0.5 * (1.0 + math.cos(math.pi * phase))
            vel = -math.sin(math.pi * phase)

        return height, vel

    def _decay_bounce(self) -> None:
        """停止讲话时平滑衰减弹跳到地面（帧率无关）。"""
        decay = self._adjust_decay(0.85)  # 0.85 per frame @30fps
        self._current_bounce *= decay
        self._current_squash_stretch *= decay
        if abs(self._current_bounce) < 0.5:  # 像素阈值
            self._current_bounce = 0.0
            self._bounce_timer = 0.0
            self._current_squash_stretch = 0.0
