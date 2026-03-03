"""
Emotion Engine — AI 情绪推理模块。

职责：
  - 接收 AudioFeatures（+ 可选文本）
  - 输出 EmotionVector（抽象情绪权重）

不做：
  - 选择表情图片
  - 控制动画参数

支持后端：
  - "rule"    — 基于规则的简易情绪映射（无需 API）
  - "openai"  — 调用 OpenAI API 多模态推理
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from vtuber_engine.models.data_models import AudioFeatures, EmotionVector


class EmotionEngine:
    """AI 情绪推理引擎。"""

    def __init__(self, backend: str = "rule"):
        """
        Args:
            backend: 推理后端，"rule" 或 "openai"。
        """
        self.backend = backend

        if backend == "openai":
            try:
                import openai  # noqa: F401
            except ImportError:
                raise ImportError(
                    "openai package is required for the openai backend. "
                    "Install it with: pip install openai"
                )

    # ──────────────────── 公共接口 ────────────────────

    def analyze(
        self,
        audio_features: AudioFeatures,
        text: Optional[str] = None,
    ) -> list[EmotionVector]:
        """
        对音频逐段分析情绪，返回与帧对齐的 EmotionVector 列表。

        阶段 1 策略：整段音频输出一个情绪，广播到所有帧。
        阶段 2 策略：按句子 / 时间窗分段分析。

        Args:
            audio_features: 音频特征。
            text: 配音对应的文本（可选，辅助判断）。

        Returns:
            与帧数等长的 EmotionVector 列表。
        """
        if self.backend == "rule":
            return self._analyze_rule(audio_features)
        elif self.backend == "openai":
            return self._analyze_openai(audio_features, text)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ──────────────────── 规则后端 ────────────────────

    def _analyze_rule(self, features: AudioFeatures) -> list[EmotionVector]:
        """
        基于音频特征的简易规则推理。
        适用于阶段 1，无需 API 即可运行。
        """
        # 计算整体统计量
        avg_volume = float(np.mean(features.volume)) if features.volume else 0.0
        avg_energy = float(np.mean(features.energy)) if features.energy else 0.0
        avg_pitch = (
            float(np.mean([p for p in features.pitch if p > 0]))
            if features.pitch
            else 0.0
        )

        # 基于统计量推导情绪权重
        emotion = self._rule_based_emotion(avg_volume, avg_energy, avg_pitch)

        # 阶段 1：整段广播（每帧相同情绪）
        # 但 energy 按帧变化以驱动口型
        results = []
        for i in range(features.frame_count):
            frame_emotion = EmotionVector(
                calm=emotion.calm,
                excited=emotion.excited,
                panic=emotion.panic,
                sad=emotion.sad,
                angry=emotion.angry,
                happy=emotion.happy,
                energy=features.volume[i] if i < len(features.volume) else 0.0,
            )
            results.append(frame_emotion)

        return results

    def _rule_based_emotion(
        self, volume: float, energy: float, pitch: float
    ) -> EmotionVector:
        """从音频统计量推导情绪权重。"""
        # 简易规则：
        # - 低音量 + 低能量 → calm
        # - 高音量 + 高能量 → excited
        # - 高音量 + 高音调 → panic
        # - 低音量 + 低音调 → sad

        calm = max(0.0, 1.0 - volume - energy)
        excited = min(1.0, (volume + energy) / 2)
        panic = min(1.0, volume * 0.5 + (pitch / 500.0) * 0.5) if pitch > 300 else 0.0
        sad = max(0.0, (1.0 - volume) * 0.5 + (1.0 - energy) * 0.5 - 0.3)
        happy = min(1.0, excited * 0.7 + (pitch / 400.0) * 0.3) if pitch > 200 else 0.0

        # 归一化
        total = calm + excited + panic + sad + happy
        if total > 0:
            calm /= total
            excited /= total
            panic /= total
            sad /= total
            happy /= total

        return EmotionVector(
            calm=round(calm, 3),
            excited=round(excited, 3),
            panic=round(panic, 3),
            sad=round(sad, 3),
            angry=0.0,
            happy=round(happy, 3),
            energy=round((volume + energy) / 2, 3),
        )

    # ──────────────────── OpenAI 后端 ────────────────────

    def _analyze_openai(
        self,
        features: AudioFeatures,
        text: Optional[str] = None,
    ) -> list[EmotionVector]:
        """
        调用 OpenAI API 进行情绪推理。
        阶段 2+ 实现。
        """
        import openai

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required.")

        client = openai.OpenAI(api_key=api_key)

        # 构建提示
        prompt = self._build_emotion_prompt(features, text)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an emotion analysis engine. "
                        "Given audio features and optional text, output emotion weights as JSON. "
                        "Keys: calm, excited, panic, sad, angry, happy, energy. "
                        "All values are floats between 0 and 1. "
                        "Do not output anything else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        import json

        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)

        base_emotion = EmotionVector(
            calm=data.get("calm", 0.0),
            excited=data.get("excited", 0.0),
            panic=data.get("panic", 0.0),
            sad=data.get("sad", 0.0),
            angry=data.get("angry", 0.0),
            happy=data.get("happy", 0.0),
            energy=data.get("energy", 0.0),
        )

        # 广播到所有帧，energy 按帧变化
        results = []
        for i in range(features.frame_count):
            frame_emotion = EmotionVector(
                calm=base_emotion.calm,
                excited=base_emotion.excited,
                panic=base_emotion.panic,
                sad=base_emotion.sad,
                angry=base_emotion.angry,
                happy=base_emotion.happy,
                energy=features.volume[i] if i < len(features.volume) else 0.0,
            )
            results.append(frame_emotion)

        return results

    def _build_emotion_prompt(
        self, features: AudioFeatures, text: Optional[str]
    ) -> str:
        """构建发给 AI 的情绪分析提示。"""
        avg_vol = float(np.mean(features.volume)) if features.volume else 0.0
        avg_energy = float(np.mean(features.energy)) if features.energy else 0.0
        speaking_ratio = features.speech_rate

        parts = [
            f"Audio duration: {features.duration:.1f}s",
            f"Average volume: {avg_vol:.3f}",
            f"Average energy: {avg_energy:.3f}",
            f"Speaking ratio: {speaking_ratio:.3f}",
        ]
        if text:
            parts.append(f"Transcript: {text}")

        return "\n".join(parts)
