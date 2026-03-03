"""
State Engine — 角色状态决策核心。

职责：
  - 接收 EmotionVector（AI 输出）+ AudioFeatures
  - 输出 CharacterState（角色状态参数）
  - 管理眨眼周期
  - 基于余弦相似度匹配最佳表情
  - 表情切换控制：
    · 最短保持时间（emotion_min_hold_seconds）—— 防止表情频繁跳变
    · 句子边界切换 —— 只在「静音→说话」的转变点切换表情
    · 强制切换（force_switch_seconds）—— 超时后在句首强制切换不同表情

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
from typing import Optional

from vtuber_engine.models.data_models import (
    AudioFeatures,
    CharacterConfig,
    CharacterState,
    EmotionVector,
    cosine_similarity,
)


class StateEngine:
    """将情绪向量 + 音频特征映射为角色状态。"""

    def __init__(
        self,
        character_config: CharacterConfig,
        fps: int = 30,
        force_switch_seconds: float = 0.0,
        gesture_min_hold_seconds: float = 5.0,
        emotion_min_hold_seconds: float = 5.0,
        mouth_frequency: float = 2.5,
    ):
        """
        Args:
            character_config: 角色配置。
            fps: 帧率。
            force_switch_seconds: 强制切换表情的最长秒数（0 = 不启用）。
            gesture_min_hold_seconds: 动作切换后最小停留秒数（防止跳变）。
            emotion_min_hold_seconds: 每个表情最少保持秒数（防止频繁切换）。
            mouth_frequency: 嘴型开合频率 (Hz)，越低越自然。
        """
        self.config = character_config
        self.fps = fps

        # 眨眼状态
        self._blink_timer = 0.0
        self._blink_active = False
        self._blink_progress = 0.0

        # ── 表情切换控制 ──
        self._force_switch_seconds = force_switch_seconds
        self._emotion_min_hold_seconds = emotion_min_hold_seconds
        self._current_emotion = ""
        self._current_emotion_frames = 0  # 当前表情持续的帧数
        self._prev_is_speaking = False  # 上一帧的 is_speaking 状态

        # 动作（gesture）防抖状态
        self._current_gesture: int = 0
        self._gesture_hold_frames: int = 0
        self._gesture_min_hold_frames: int = max(1, int(gesture_min_hold_seconds * fps))

        # 嘴巴开合振荡定时器（讲话时张嘴闭嘴切换，而不是一直张嘴）
        self._mouth_timer: float = 0.0
        self._mouth_frequency: float = mouth_frequency

        # 表情使用历史（用于降权，避免始终停留在同一表情）
        self._emotion_usage: dict[str, float] = {}
        self._emotion_decay: float = 0.995  # 每帧衰减系数
        self._emotion_penalty_factor: float = 0.003  # 每单位使用量的惩罚系数
        self._emotion_penalty_cap: float = 0.3  # 最大惩罚比例 (30%)

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

        print(
            f"[StateEngine] process() start: frame_count={frame_count}, "
            f"fps={self.fps}, dt={dt:.4f}, "
            f"available_emotions={self.config.emotions}, "
            f"emotion_vectors_count={len(emotion_vectors)}, "
            f"emotion_min_hold={self._emotion_min_hold_seconds}s, "
            f"force_switch_seconds={self._force_switch_seconds}, "
            f"mouth_frequency={self._mouth_frequency}Hz, "
            f"has_expression_vectors={bool(self.config.emotion_vectors)}"
        )

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

            # 每 30 帧 or 首/末帧打印一次调试信息
            if i == 0 or i == frame_count - 1 or (i > 0 and i % 30 == 0):
                print(
                    f"[StateEngine] frame[{i:04d}] "
                    f"emotion='{state.emotion}' "
                    f"mouth_open={state.mouth_open:.3f} "
                    f"blink_phase={state.blink_phase:.3f} "
                    f"energy={state.energy:.3f} "
                    f"gesture={state.gesture} "
                    f"is_speaking={is_speaking} volume={volume:.3f}"
                )

        print(
            f"[StateEngine] process() done: {len(states)} states generated. "
            f"Emotion distribution: "
            + str(
                {
                    e: sum(1 for s in states if s.emotion == e)
                    for e in set(s.emotion for s in states)
                }
            )
        )
        return states

    # ──────────────────── 表情匹配 ────────────────────

    def _match_expression_by_vector(
        self,
        audio_emotion: EmotionVector,
        exclude_emotion: str = "",
    ) -> str:
        """
        使用余弦相似度将音频情绪向量与已注册表情的情绪向量匹配。

        Args:
            audio_emotion: 当前帧的音频情绪向量。
            exclude_emotion: 要排除的表情（强制切换时使用）。

        Returns:
            最佳匹配的表情标签。
        """
        if not self.config.emotion_vectors:
            # 没有表情向量数据时回退到 dominant_emotion 逻辑
            return self._match_expression_by_dominant(audio_emotion, exclude_emotion)

        audio_dict = audio_emotion.emotion_only_dict()
        best_emotion = ""
        best_sim = -2.0

        for emo, emo_vec in self.config.emotion_vectors.items():
            if emo == exclude_emotion:
                continue
            if emo not in self.config.emotions:
                continue
            sim = cosine_similarity(audio_dict, emo_vec)
            # 历史使用降权：最近频繁使用的表情得分降低
            usage = self._emotion_usage.get(emo, 0.0)
            penalty = min(
                usage * self._emotion_penalty_factor,
                self._emotion_penalty_cap,
            )
            adjusted_sim = sim * (1.0 - penalty)
            if adjusted_sim > best_sim:
                best_sim = adjusted_sim
                best_emotion = emo

        if not best_emotion:
            # 如果排除后没有可选项，回退到不排除
            return self._match_expression_by_vector(audio_emotion, exclude_emotion="")

        print(
            f"[StateEngine] vector_match: best='{best_emotion}' sim={best_sim:.3f} "
            f"(excluded='{exclude_emotion}')"
        )
        return best_emotion

    def _match_expression_by_dominant(
        self,
        audio_emotion: EmotionVector,
        exclude_emotion: str = "",
    ) -> str:
        """回退方案：使用 dominant_emotion 匹配（无向量数据时）。"""
        dominant = audio_emotion.dominant_emotion()

        candidates = [e for e in self.config.emotions if e != exclude_emotion]
        if not candidates:
            candidates = self.config.emotions[:]

        if dominant in candidates:
            return dominant

        # 回退到第一个可用
        fallback = candidates[0] if candidates else "calm"
        print(
            f"[StateEngine] dominant_fallback: dominant='{dominant}' not in candidates, "
            f"using '{fallback}'"
        )
        return fallback

    # ──────────────────── 内部逻辑 ────────────────────

    # ──────────────────── 表情决策（统一） ────────────────────

    def _decide_emotion(
        self,
        emotion_vec: EmotionVector,
        is_speaking: bool,
    ) -> str:
        """
        统一的表情决策逻辑，保证表情平稳过渡：
          1. 最短保持时间内锁定当前表情
          2. 超过最短时间后，在句子边界（静音→说话）处允许切换
          3. 超过强制切换时间后，在句子边界处强制切换到不同表情

        "句子边界" 定义：从不说话过渡到说话的那一帧（silence → speech）。
        """
        # 获取 AI 匹配的理想表情
        matched = self._match_expression_by_vector(emotion_vec)

        # 首帧初始化
        if not self._current_emotion:
            self._current_emotion = matched
            self._current_emotion_frames = 0
            return matched

        self._current_emotion_frames += 1
        elapsed = self._current_emotion_frames / self.fps

        # 只有 1 个表情时无需切换
        if len(self.config.emotions) <= 1:
            return self._current_emotion

        # ── 最短保持：锁定当前表情 ──
        if elapsed < self._emotion_min_hold_seconds:
            return self._current_emotion

        # ── 判断是否需要切换 ──
        wants_switch = matched != self._current_emotion
        force_mode = False

        if self._force_switch_seconds > 0 and elapsed >= self._force_switch_seconds:
            wants_switch = True
            force_mode = True
            # 强制切换时排除当前表情
            if matched == self._current_emotion:
                matched = self._match_expression_by_vector(
                    emotion_vec, exclude_emotion=self._current_emotion
                )

        if not wants_switch:
            return self._current_emotion

        # ── 等待句子边界切换 ──
        is_sentence_start = is_speaking and not self._prev_is_speaking
        if is_sentence_start:
            print(
                f"[StateEngine] emotion_switch: '{self._current_emotion}' -> '{matched}' "
                f"at sentence boundary (held {elapsed:.1f}s, force={force_mode})"
            )
            self._current_emotion = matched
            self._current_emotion_frames = 0
            return matched

        # 还没到句子边界，保持当前表情
        return self._current_emotion

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

        # 1. 表情决策（统一逻辑：最短保持 + 句子边界切换）
        state.emotion = self._decide_emotion(emotion, is_speaking)

        # 2. 能量
        state.energy = emotion.energy

        # 3. 嘴型（讲话时张嘴闭嘴切换，而非一直张嘴）
        if is_speaking:
            self._mouth_timer += dt
            freq = self._mouth_frequency
            phase = math.sin(2 * math.pi * freq * self._mouth_timer)
            # 正半周期张嘴，负半周期闭嘴，创造明显的开合节奏
            mouth_envelope = max(0.0, phase)
            state.mouth_open = mouth_envelope * min(1.0, volume * 2.0)
        else:
            state.mouth_open = 0.0
            self._mouth_timer = 0.0  # 不讲话时重置计时器

        # 4. 眨眼
        state.blink_phase = self._update_blink(dt, state.energy)

        # 5. 动作
        state.gesture = self._decide_gesture(state.energy)

        # 6. 表情权重
        state.expression_weights = emotion.as_dict()

        # 7. 更新前帧说话状态（用于句子边界检测）
        self._prev_is_speaking = is_speaking

        # 8. 更新表情使用历史（所有表情衰减，当前表情累加）
        for k in self._emotion_usage:
            self._emotion_usage[k] *= self._emotion_decay
        self._emotion_usage[state.emotion] = (
            self._emotion_usage.get(state.emotion, 0.0) + 1.0
        )

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
        """根据能量决定动作，带最小停留时间防止跳变。"""
        # 目标 gesture
        if energy > 0.8:
            target = 2
        elif energy > 0.5:
            target = 1
        else:
            target = 0

        self._gesture_hold_frames += 1

        # 只有持续足够长时间才允许切换
        if (
            target != self._current_gesture
            and self._gesture_hold_frames >= self._gesture_min_hold_frames
        ):
            self._current_gesture = target
            self._gesture_hold_frames = 0

        return self._current_gesture

    # 注：原 _smooth_state 历史投票已移除。
    # 表情稳定化由 _decide_emotion() 的「最短保持 + 句子边界」机制统一处理。
