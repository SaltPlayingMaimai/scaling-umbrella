"""
Audio Analyzer — 音频特征提取模块。

职责：
  - 加载音频文件（wav / mp3 / ogg）
  - 提取逐帧特征：RMS 音量、基频、能量
  - 简易 VAD（语音活动检测）
  - 输出 AudioFeatures 数据结构

不做：
  - 情绪判断（交给 Emotion Engine）
  - 动画决策（交给 State Engine）
"""

from __future__ import annotations

import numpy as np

from vtuber_engine.models.data_models import AudioFeatures

try:
    import librosa
except ImportError:
    librosa = None


class AudioAnalyzer:
    """从音频文件提取用于角色动画的特征。"""

    def __init__(self, fps: int = 30, sample_rate: int = 22050):
        """
        Args:
            fps: 分析帧率，与最终视频帧率一致。
            sample_rate: 音频重采样率。
        """
        if librosa is None:
            raise ImportError(
                "librosa is required for audio analysis. "
                "Install it with: pip install librosa"
            )
        self.fps = fps
        self.sample_rate = sample_rate

    # ──────────────────── 公共接口 ────────────────────

    def analyze(self, audio_path: str) -> AudioFeatures:
        """
        分析音频文件，返回逐帧特征。

        Args:
            audio_path: 音频文件路径。

        Returns:
            AudioFeatures 数据结构。
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = librosa.get_duration(y=y, sr=sr)

        hop_length = sr // self.fps  # 每帧对应的采样点数

        volume = self._extract_volume(y, hop_length)
        pitch = self._extract_pitch(y, sr, hop_length)
        energy = self._extract_energy(y, hop_length)
        is_speaking = self._detect_speech(volume)

        features = AudioFeatures(
            duration=duration,
            sample_rate=sr,
            fps=self.fps,
            volume=volume,
            pitch=pitch,
            energy=energy,
            speech_rate=self._estimate_speech_rate(is_speaking),
            is_speaking=is_speaking,
        )
        return features

    # ──────────────────── 特征提取 ────────────────────

    def _extract_volume(self, y: np.ndarray, hop_length: int) -> list[float]:
        """RMS 音量，归一化到 0~1。"""
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        # 归一化
        max_rms = rms.max() if rms.max() > 0 else 1.0
        normalized = (rms / max_rms).tolist()
        return normalized

    def _extract_pitch(self, y: np.ndarray, sr: int, hop_length: int) -> list[float]:
        """基频提取（pyin），无声段为 0。"""
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=hop_length,
        )
        # 将 NaN（无声段）替换为 0
        f0 = np.nan_to_num(f0, nan=0.0)
        return f0.tolist()

    def _extract_energy(self, y: np.ndarray, hop_length: int) -> list[float]:
        """短时能量，归一化到 0~1。"""
        # 分帧
        frames = librosa.util.frame(y, frame_length=hop_length, hop_length=hop_length)
        energy = np.mean(frames**2, axis=0)
        max_energy = energy.max() if energy.max() > 0 else 1.0
        normalized = (energy / max_energy).tolist()
        return normalized

    def _detect_speech(
        self, volume: list[float], threshold: float = 0.05
    ) -> list[bool]:
        """
        简易 VAD：音量超过阈值视为正在说话。

        Args:
            volume: 归一化 RMS 音量列表。
            threshold: 判定阈值。
        """
        return [v > threshold for v in volume]

    def _estimate_speech_rate(self, is_speaking: list[bool]) -> float:
        """
        估算语速（说话段占比作为近似）。
        后续可用 Whisper 提取精确音节数。
        """
        if not is_speaking:
            return 0.0
        return sum(is_speaking) / len(is_speaking)
