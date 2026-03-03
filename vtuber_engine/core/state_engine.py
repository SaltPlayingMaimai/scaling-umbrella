"""
State Engine — 角色状态决策核心。

职责：
  - 接收 EmotionVector（AI 输出）+ AudioFeatures
  - 输出 CharacterState（角色状态参数）
  - 管理眨眼周期
  - 历史平滑（防止情绪跳变）

不做：
  - 情绪推理（交给 Emotion Engine）
  - 动画插值（交给 Animation Engine）
  - 渲染（交给 Renderer）

核心理念：
  AI → 状态
  状态 → 动画参数
  AI ≠ 直接控制动画
"""

from __future__ import annotations

import math
from typing import List, Optional

from vtuber_engine.models.data_models import (
    AudioFeatures,
    CharacterConfig,
    CharacterState,
    EmotionVector,
)


class StateEngine:
    """将情绪向量 + 音频特征映射为角色状态。"""

    def __init__(self, character_config: CharacterConfig, fps: int = 30):
        """
        Args:
            character_config: 角色配置。
            fps: 帧率。
        """
        self.config = character_config
        self.fps = fps

        # 历史状态（用于平滑）
        self._history: List[CharacterState] = []
        self._history_window = 5  # 平滑窗口大小（帧数）

        # 眨眼状态
        self._blink_timer = 0.0
        self._blink_active = False
        self._blink_progress = 0.0

    # ──────────────────── 公共接口 ────────────────────

    def process(
        self,
        audio_features: AudioFeatures,
        emotion_vectors: list[EmotionVector],
    ) -> list[CharacterState]:
        """
        逐帧生成角色状态。

        Args:
            audio_features: 音频特征。
            emotion_vectors: 与帧对齐的情绪向量列表。

        Returns:
            与帧数等长的 CharacterState 列表。
        """
        frame_count = audio_features.frame_count
        dt = 1.0 / self.fps
        states: list[CharacterState] = []

        for i in range(frame_count):
            emotion = (
                emotion_vectors[i] if i < len(emotion_vectors) else EmotionVector()
            )
            is_speaking = (
                audio_features.is_speaking[i]
                if i < len(audio_features.is_speaking)
                else False
            )
            volume = audio_features.volume[i] if i < len(audio_features.volume) else 0.0

            state = self._compute_frame_state(emotion, is_speaking, volume, dt)
            states.append(state)

        return states

    # ──────────────────── 内部逻辑 ────────────────────

    def _compute_frame_state(
        self,
        emotion: EmotionVector,
        is_speaking: bool,
        volume: float,
        dt: float,
    ) -> CharacterState:
        """计算单帧角色状态。"""

        state = CharacterState()

        # 1. 主情绪（限制为角色配置中定义的情绪列表）
        dominant = emotion.dominant_emotion()
        if dominant in self.config.emotions:
            state.emotion = dominant
        else:
            # 回退到配置中第一个情绪
            state.emotion = self.config.emotions[0] if self.config.emotions else "calm"

        # 2. 能量
        state.energy = emotion.energy

        # 3. 嘴型（说话时按音量开合）
        if is_speaking:
            state.mouth_open = min(1.0, volume * 1.5)
        else:
            state.mouth_open = 0.0

        # 4. 眨眼
        state.blink_phase = self._update_blink(dt, state.energy)

        # 5. 动作
        state.gesture = self._decide_gesture(state.energy)

        # 6. 表情权重
        state.expression_weights = emotion.as_dict()

        # 7. 历史平滑
        state = self._smooth_state(state)

        return state

    def _update_blink(self, dt: float, energy: float) -> float:
        """
        更新眨眼状态，返回 blink_phase (0~1)。
        0 = 眼睛全开, 1 = 眼睛全闭。
        """
        blink_interval = self.config.blink_interval
        blink_duration = self.config.blink_duration

        # 能量越高，眨眼越频繁
        adjusted_interval = blink_interval * (1.0 - energy * 0.5)
        adjusted_interval = max(0.5, adjusted_interval)

        self._blink_timer += dt

        if not self._blink_active:
            if self._blink_timer >= adjusted_interval:
                self._blink_active = True
                self._blink_timer = 0.0
                self._blink_progress = 0.0
            return 0.0
        else:
            self._blink_progress += dt / blink_duration
            if self._blink_progress >= 1.0:
                self._blink_active = False
                self._blink_timer = 0.0
                return 0.0
            # 正弦曲线：0 → 1 → 0
            return math.sin(self._blink_progress * math.pi)

    def _decide_gesture(self, energy: float) -> int:
        """根据能量决定动作。"""
        if energy > 0.8:
            return 2  # 大幅动作
        elif energy > 0.5:
            return 1  # 小幅动作
        return 0  # 无动作

    def _smooth_state(self, state: CharacterState) -> CharacterState:
        """
        历史窗口平滑，防止情绪标签频繁跳变。
        数值参数由 Animation Engine 做插值，这里只做情绪标签稳定化。
        """
        self._history.append(state.clone())

        # 保持窗口大小
        if len(self._history) > self._history_window:
            self._history = self._history[-self._history_window :]

        # 情绪投票：窗口内出现最多的情绪标签胜出
        emotion_counts: dict[str, int] = {}
        for s in self._history:
            emotion_counts[s.emotion] = emotion_counts.get(s.emotion, 0) + 1

        dominant = max(emotion_counts, key=emotion_counts.get)
        state.emotion = dominant

        return state
