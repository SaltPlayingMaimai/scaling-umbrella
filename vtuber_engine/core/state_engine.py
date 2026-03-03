"""
State Engine — 角色状态决策核心。

职责：
  - 接收 EmotionVector（AI 输出）+ AudioFeatures
  - 输出 CharacterState（角色状态参数）
  - 管理眨眼周期
  - 历史平滑（防止情绪跳变）
  - 基于余弦相似度匹配最佳表情
  - 强制切换：超时后在句首切换不同表情

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
    ):
        """
        Args:
            character_config: 角色配置。
            fps: 帧率。
            force_switch_seconds: 强制切换表情的最长秒数（0 = 不启用）。
            gesture_min_hold_seconds: 动作切换后最小停留秒数（防止跳变）。
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

        # 强制切换状态
        self._force_switch_seconds = force_switch_seconds
        self._current_emotion = ""
        self._current_emotion_frames = 0  # 当前表情持续的帧数
        self._waiting_for_sentence_start = False  # 是否在等待句首切换点
        self._prev_is_speaking = False  # 上一帧的 is_speaking 状态

        # 动作（gesture）防抖状态
        self._current_gesture: int = 0
        self._gesture_hold_frames: int = 0
        self._gesture_min_hold_frames: int = max(1, int(gesture_min_hold_seconds * fps))

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
            f"history_window={self._history_window}, "
            f"force_switch_seconds={self._force_switch_seconds}, "
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
            if sim > best_sim:
                best_sim = sim
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

    def _compute_frame_state(
        self,
        emotion: EmotionVector,
        is_speaking: bool,
        volume: float,
        dt: float,
    ) -> CharacterState:
        """计算单帧角色状态。"""

        state = CharacterState()

        # 1. 表情匹配（通过余弦相似度或 dominant 回退）
        matched = self._match_expression_by_vector(emotion)

        # 2. 强制切换逻辑
        if (
            self._force_switch_seconds > 0
            and self.config.emotions
            and len(self.config.emotions) > 1
        ):
            if matched == self._current_emotion:
                self._current_emotion_frames += 1
            else:
                # 自然地切换了表情
                self._current_emotion = matched
                self._current_emotion_frames = 1
                self._waiting_for_sentence_start = False

            elapsed_seconds = self._current_emotion_frames / self.fps

            if (
                elapsed_seconds >= self._force_switch_seconds
                and not self._waiting_for_sentence_start
            ):
                # 超过时限，开始等待句首切换点
                self._waiting_for_sentence_start = True
                print(
                    f"[StateEngine] force_switch: emotion='{self._current_emotion}' "
                    f"held for {elapsed_seconds:.1f}s (>={self._force_switch_seconds}s), "
                    f"waiting for sentence boundary..."
                )

            if self._waiting_for_sentence_start:
                # 检测句首：从不说话 → 说话的转变（silence → speech transition）
                is_sentence_start = is_speaking and not self._prev_is_speaking
                if is_sentence_start:
                    # 强制切换到不同的表情
                    forced = self._match_expression_by_vector(
                        emotion, exclude_emotion=self._current_emotion
                    )
                    print(
                        f"[StateEngine] force_switch: TRIGGERED at sentence start! "
                        f"'{self._current_emotion}' -> '{forced}' "
                        f"(held {elapsed_seconds:.1f}s)"
                    )
                    matched = forced
                    self._current_emotion = forced
                    self._current_emotion_frames = 1
                    self._waiting_for_sentence_start = False
                else:
                    # 还没到句首，保持当前表情
                    matched = self._current_emotion

            self._prev_is_speaking = is_speaking
        else:
            # 不启用强制切换
            self._current_emotion = matched
            self._current_emotion_frames = 1

        state.emotion = matched

        # 3. 能量
        state.energy = emotion.energy

        # 4. 嘴型（说话时按音量开合）
        if is_speaking:
            state.mouth_open = min(1.0, volume * 1.5)
        else:
            state.mouth_open = 0.0

        # 5. 眨眼
        state.blink_phase = self._update_blink(dt, state.energy)

        # 6. 动作
        state.gesture = self._decide_gesture(state.energy)

        # 7. 表情权重
        state.expression_weights = emotion.as_dict()

        # 8. 历史平滑
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
