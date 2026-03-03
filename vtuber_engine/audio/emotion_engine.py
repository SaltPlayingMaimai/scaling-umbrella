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
  - "qwen"    — 调用通义千问 Qwen API（通过 DashScope）
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv

from vtuber_engine.models.data_models import AudioFeatures, EmotionVector, EMOTION_KEYS

# 加载 .env
load_dotenv()


class EmotionEngine:
    """AI 情绪推理引擎。"""

    def __init__(
        self,
        backend: str = "rule",
        available_emotions: Optional[List[str]] = None,
        model: Optional[str] = None,
    ):
        """
        Args:
            backend: 推理后端，"rule"、"openai" 或 "qwen"。
            available_emotions: 用户已上传的表情名称列表（用于约束 AI 输出）。
            model: 指定调用的模型名称，None 时使用各后端默认值。
        """
        self.backend = backend
        self.available_emotions = available_emotions or []
        # 模型名称：None 则各后端内部默认
        self.model = model

        if backend == "openai":
            try:
                import openai  # noqa: F401
            except ImportError:
                raise ImportError(
                    "openai package is required for the openai backend. "
                    "Install it with: pip install openai"
                )
        elif backend == "qwen":
            try:
                import dashscope  # noqa: F401
            except ImportError:
                raise ImportError(
                    "dashscope package is required for the qwen backend. "
                    "Install it with: pip install dashscope"
                )

    # ──────────────────── 公共接口 ────────────────────

    def analyze(
        self,
        audio_features: AudioFeatures,
        text: Optional[str] = None,
        segment_seconds: float = 1.0,
    ) -> list[EmotionVector]:
        """
        对音频按时间片段分析情绪，返回与帧对齐的 EmotionVector 列表。

        策略：将音频分割为 segment_seconds 秒的片段，每段调用 AI 一次，
        所得结果广播到该段的所有帧（energy 仍按帧变化）。

        Args:
            audio_features: 音频特征。
            text: 配音对应的文本（可选，辅助判断）。
            segment_seconds: 每个分析片段的时长（秒），最小 0.1，默认 1.0。

        Returns:
            与帧数等长的 EmotionVector 列表。
        """
        segment_seconds = max(0.1, segment_seconds)
        print(
            f"[EmotionEngine] analyze() start: backend={self.backend}, "
            f"total_frames={audio_features.frame_count}, "
            f"duration={audio_features.duration:.2f}s, fps={audio_features.fps}, "
            f"segment_seconds={segment_seconds}, "
            f"available_emotions={self.available_emotions}"
        )

        if self.backend == "rule":
            return self._analyze_rule(audio_features, segment_seconds)
        elif self.backend == "openai":
            return self._analyze_openai(audio_features, text, segment_seconds)
        elif self.backend == "qwen":
            return self._analyze_qwen(audio_features, text, segment_seconds)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ──────────────────── 分段辅助 ────────────────────

    def _slice_features(
        self, features: AudioFeatures, start_frame: int, end_frame: int
    ) -> AudioFeatures:
        """将 AudioFeatures 按帧范围切片，返回新的 AudioFeatures。"""
        seg_duration = (end_frame - start_frame) / features.fps
        return AudioFeatures(
            duration=seg_duration,
            sample_rate=features.sample_rate,
            fps=features.fps,
            volume=features.volume[start_frame:end_frame],
            pitch=features.pitch[start_frame:end_frame],
            energy=features.energy[start_frame:end_frame],
            speech_rate=features.speech_rate,
            is_speaking=features.is_speaking[start_frame:end_frame],
        )

    def _iter_segments(self, features: AudioFeatures, segment_seconds: float):
        """
        生成 (start_frame, end_frame, segment_features) 三元组。
        每个片段时长约 segment_seconds 秒。
        """
        fps = features.fps
        total = features.frame_count
        frames_per_seg = max(1, int(fps * segment_seconds))
        seg_idx = 0
        for start in range(0, total, frames_per_seg):
            end = min(start + frames_per_seg, total)
            yield seg_idx, start, end, self._slice_features(features, start, end)
            seg_idx += 1

    # ──────────────────── 规则后端 ────────────────────

    def _analyze_rule(
        self, features: AudioFeatures, segment_seconds: float = 1.0
    ) -> list[EmotionVector]:
        """
        基于音频特征的简易规则推理（按片段分析）。
        """
        results: list[EmotionVector] = [None] * features.frame_count  # type: ignore
        total_segs = 0

        for seg_idx, start, end, seg in self._iter_segments(features, segment_seconds):
            avg_volume = float(np.mean(seg.volume)) if seg.volume else 0.0
            avg_energy = float(np.mean(seg.energy)) if seg.energy else 0.0
            avg_pitch = (
                float(np.mean([p for p in seg.pitch if p > 0]))
                if any(p > 0 for p in seg.pitch)
                else 0.0
            )

            emotion = self._rule_based_emotion(avg_volume, avg_energy, avg_pitch)
            print(
                f"[EmotionEngine][rule] seg[{seg_idx}] frames[{start}~{end}] "
                f"avg_vol={avg_volume:.3f} avg_energy={avg_energy:.3f} avg_pitch={avg_pitch:.1f} "
                f"-> dominant={emotion.dominant_emotion()}"
            )

            for i in range(start, end):
                frame_vol = features.volume[i] if i < len(features.volume) else 0.0
                results[i] = EmotionVector(
                    **{k: getattr(emotion, k) for k in EMOTION_KEYS},
                    energy=frame_vol,
                )
            total_segs += 1

        print(
            f"[EmotionEngine][rule] done: {len(results)} frames across {total_segs} segments"
        )
        return results

    def _rule_based_emotion(
        self, volume: float, energy: float, pitch: float
    ) -> EmotionVector:
        """从音频统计量推导 12 维情绪权重。"""
        # 基础维度
        calm = max(0.0, 1.0 - volume - energy)
        excited = min(1.0, (volume + energy) / 2)
        panic = min(1.0, volume * 0.5 + (pitch / 500.0) * 0.5) if pitch > 300 else 0.0
        sad = max(0.0, (1.0 - volume) * 0.5 + (1.0 - energy) * 0.5 - 0.3)
        happy = min(1.0, excited * 0.7 + (pitch / 400.0) * 0.3) if pitch > 200 else 0.0
        angry = (
            min(1.0, volume * 0.6 + energy * 0.4)
            if energy > 0.6 and pitch < 250
            else 0.0
        )
        # 扩展维度
        shy = max(0.0, (1.0 - volume) * 0.3) if volume < 0.3 and energy < 0.3 else 0.0
        surprised = (
            min(1.0, (pitch / 500.0) * 0.7 + energy * 0.3) if pitch > 350 else 0.0
        )
        tender = max(0.0, calm * 0.5 + (1.0 - energy) * 0.3) if volume < 0.4 else 0.0
        smug = 0.0  # 难以从音频推断
        tired = (
            max(0.0, (1.0 - energy) * 0.5 + (1.0 - volume) * 0.3 - 0.2)
            if energy < 0.3
            else 0.0
        )
        confused = 0.0  # 难以从音频推断

        # 归一化
        vals = [
            calm,
            happy,
            excited,
            sad,
            angry,
            panic,
            shy,
            surprised,
            tender,
            smug,
            tired,
            confused,
        ]
        total = sum(vals) or 1.0
        norm = [round(v / total, 3) for v in vals]

        return EmotionVector(
            **{k: norm[i] for i, k in enumerate(EMOTION_KEYS)},
            energy=round((volume + energy) / 2, 3),
        )

    # ──────────────────── OpenAI 后端 ────────────────────

    def _analyze_openai(
        self,
        features: AudioFeatures,
        text: Optional[str] = None,
        segment_seconds: float = 1.0,
    ) -> list[EmotionVector]:
        """
        调用 OpenAI API 进行情绪推理（按片段分析）。
        每个 segment_seconds 时间窗调用一次 AI。
        """
        import openai

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 未设置。请在 .env 文件中配置 OPENAI_API_KEY=sk-xxx"
            )

        base_url = os.environ.get("OPENAI_BASE_URL", None)
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        client = openai.OpenAI(**kwargs)

        emotion_constraint = ""
        if self.available_emotions:
            emotion_constraint = (
                f"\nThe available emotion labels are: {', '.join(self.available_emotions)}. "
                "You MUST pick from these labels only. "
            )

        system_content = (
            "You are an emotion analysis engine. "
            "Given audio features and optional text, output emotion weights as JSON. "
            f"Keys: {', '.join(EMOTION_KEYS)}, energy. "
            "All values are floats between 0 and 1. "
            "The emotion keys should sum to approximately 1.0. "
            "energy represents the audio energy level independently. "
            "Do not output anything else."
            f"{emotion_constraint}"
        )

        results: list[EmotionVector] = [None] * features.frame_count  # type: ignore
        segments = list(self._iter_segments(features, segment_seconds))
        total_segs = len(segments)

        print(
            f"[EmotionEngine][openai] Analyzing {total_segs} segment(s) "
            f"({segment_seconds:.1f}s each), model={self.model or 'gpt-4o-mini'}"
        )

        import json

        for seg_idx, start, end, seg in segments:
            prompt = self._build_emotion_prompt(seg, text)
            print(
                f"[EmotionEngine][openai] seg[{seg_idx}/{total_segs}] "
                f"frames[{start}~{end}] duration={seg.duration:.2f}s calling API..."
            )

            response = client.chat.completions.create(
                model=self.model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            if response is None:
                raise ValueError("OpenAI API 返回了空响应。请检查 API Key 和网络连接。")
            if not response.choices or response.choices[0] is None:
                raise ValueError("OpenAI API 返回了无效的 choices。请重试。")
            if response.choices[0].message is None:
                raise ValueError("OpenAI API 返回了无效的 message。请重试。")

            raw = response.choices[0].message.content
            if not raw:
                raise ValueError("OpenAI API 返回了空的 message 内容。请重试。")

            raw = raw.strip()
            if raw.startswith("```"):
                import re

                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw.strip())
            data = json.loads(raw)

            base_emotion = EmotionVector(
                **{k: data.get(k, 0.0) for k in EMOTION_KEYS},
                energy=data.get("energy", 0.0),
            )
            print(
                f"[EmotionEngine][openai] seg[{seg_idx}] -> dominant={base_emotion.dominant_emotion()} "
                + " ".join(f"{k}={getattr(base_emotion, k):.2f}" for k in EMOTION_KEYS)
            )

            for i in range(start, end):
                frame_vol = features.volume[i] if i < len(features.volume) else 0.0
                results[i] = EmotionVector(
                    **{k: getattr(base_emotion, k) for k in EMOTION_KEYS},
                    energy=frame_vol,
                )

        print(
            f"[EmotionEngine][openai] done: {len(results)} frames across {total_segs} segments"
        )
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

    # ──────────────────── Qwen (DashScope) 后端 ────────────────────

    def _analyze_qwen(
        self,
        features: AudioFeatures,
        text: Optional[str] = None,
        segment_seconds: float = 1.0,
    ) -> list[EmotionVector]:
        """
        调用通义千问 Qwen API 进行情绪推理（按片段分析）。
        每个 segment_seconds 时间窗调用一次 AI。
        """
        import dashscope
        import json

        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未设置。请在 .env 文件中配置 DASHSCOPE_API_KEY=sk-xxx"
            )

        base_url = os.environ.get(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
        )
        dashscope.base_http_api_url = base_url

        emotion_constraint = ""
        if self.available_emotions:
            emotion_constraint = (
                f"\nThe available emotion labels are: {', '.join(self.available_emotions)}. "
                "You MUST pick from these labels only. "
            )

        system_content = (
            "You are an emotion analysis engine. "
            "Given audio features and optional text, output emotion weights as JSON. "
            f"Keys: {', '.join(EMOTION_KEYS)}, energy. "
            "All values are floats between 0 and 1. "
            "The emotion keys should sum to approximately 1.0. "
            "energy represents the audio energy level independently. "
            "Do not output anything else."
            f"{emotion_constraint}"
        )

        results: list[EmotionVector] = [None] * features.frame_count  # type: ignore
        segments = list(self._iter_segments(features, segment_seconds))
        total_segs = len(segments)

        print(
            f"[EmotionEngine][qwen] Analyzing {total_segs} segment(s) "
            f"({segment_seconds:.1f}s each), model={self.model or 'qwen-plus'}"
        )

        for seg_idx, start, end, seg in segments:
            prompt = self._build_emotion_prompt(seg, text)
            print(
                f"[EmotionEngine][qwen] seg[{seg_idx}/{total_segs}] "
                f"frames[{start}~{end}] duration={seg.duration:.2f}s calling API..."
            )

            response = dashscope.Generation.call(
                api_key=api_key,
                model=self.model or "qwen-plus",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                result_format="message",
            )

            if response is None:
                raise ValueError("Qwen API 返回了空响应。请检查 API Key 和网络连接。")
            if not hasattr(response, "output") or response.output is None:
                # 提取实际错误信息
                err_code = getattr(
                    response, "code", getattr(response, "status_code", "unknown")
                )
                try:
                    err_msg = response.message
                except Exception:
                    try:
                        err_msg = response.error  # type: ignore[attr-defined]
                    except Exception:
                        err_msg = str(response)
                raise ValueError(f"Qwen API 调用失败（code={err_code}\uff09: {err_msg}")
            if not response.output.choices or response.output.choices[0] is None:
                raise ValueError("Qwen API 返回了无效的 choices。请重试。")
            if response.output.choices[0].message is None:
                raise ValueError("Qwen API 返回了无效的 message。请重试。")

            raw = response.output.choices[0].message.content
            if not raw:
                raise ValueError("Qwen API 返回了空的 message 内容。请重试。")

            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)

            base_emotion = EmotionVector(
                **{k: data.get(k, 0.0) for k in EMOTION_KEYS},
                energy=data.get("energy", 0.0),
            )
            print(
                f"[EmotionEngine][qwen] seg[{seg_idx}] -> dominant={base_emotion.dominant_emotion()} "
                + " ".join(f"{k}={getattr(base_emotion, k):.2f}" for k in EMOTION_KEYS)
            )

            for i in range(start, end):
                frame_vol = features.volume[i] if i < len(features.volume) else 0.0
                results[i] = EmotionVector(
                    **{k: getattr(base_emotion, k) for k in EMOTION_KEYS},
                    energy=frame_vol,
                )

        print(
            f"[EmotionEngine][qwen] done: {len(results)} frames across {total_segs} segments"
        )
        return results
