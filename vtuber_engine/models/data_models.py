"""
统一数据结构定义。
所有模块之间通过这些数据类通信，保持解耦。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from PIL import Image


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────


# 标准情绪维度（12 维）—— 所有模块共享此列表
EMOTION_KEYS = [
    "calm",
    "happy",
    "excited",
    "sad",
    "angry",
    "panic",
    "shy",
    "surprised",
    "tender",
    "smug",
    "tired",
    "confused",
]


def cosine_similarity(vec_a, vec_b) -> float:
    """计算两个情绪向量之间的余弦相似度（忽略 energy 字段）。

    支持 Dict[str, float] 或 EmotionVector 对象。
    返回值范围 [-1, 1]，1 表示完全相同方向。
    """
    keys = EMOTION_KEYS

    def _get(v, k):
        if isinstance(v, dict):
            return v.get(k, 0.0)
        return getattr(v, k, 0.0)

    a_vals = [_get(vec_a, k) for k in keys]
    b_vals = [_get(vec_b, k) for k in keys]

    dot = sum(x * y for x, y in zip(a_vals, b_vals))
    mag_a = math.sqrt(sum(x * x for x in a_vals)) or 1e-9
    mag_b = math.sqrt(sum(x * x for x in b_vals)) or 1e-9
    return dot / (mag_a * mag_b)


# ─────────────────────────────────────────────
# Audio Analyzer 输出
# ─────────────────────────────────────────────


@dataclass
class AudioFeatures:
    """音频分析器提取的逐帧特征。"""

    duration: float = 0.0  # 音频总时长（秒）
    sample_rate: int = 22050  # 采样率
    fps: int = 30  # 分析帧率

    volume: List[float] = field(default_factory=list)  # 每帧 RMS 音量 (0~1)
    pitch: List[float] = field(default_factory=list)  # 每帧基频 Hz（0 表示无声）
    energy: List[float] = field(default_factory=list)  # 每帧能量
    speech_rate: float = 0.0  # 整体语速（音节/秒）
    is_speaking: List[bool] = field(default_factory=list)  # 每帧是否在说话（VAD）

    @property
    def frame_count(self) -> int:
        return len(self.volume)


# ─────────────────────────────────────────────
# Emotion Engine 输出
# ─────────────────────────────────────────────


@dataclass
class EmotionVector:
    """AI 情绪推理结果 — 纯抽象权重，不指定具体表情。12 维情绪 + energy。"""

    calm: float = 0.0
    happy: float = 0.0
    excited: float = 0.0
    sad: float = 0.0
    angry: float = 0.0
    panic: float = 0.0
    shy: float = 0.0
    surprised: float = 0.0
    tender: float = 0.0
    smug: float = 0.0
    tired: float = 0.0
    confused: float = 0.0
    energy: float = 0.0  # 综合能量 0~1

    def dominant_emotion(self) -> str:
        """返回权重最高的情绪标签。"""
        return max(EMOTION_KEYS, key=lambda k: getattr(self, k, 0.0))

    def as_dict(self) -> Dict[str, float]:
        d = {k: getattr(self, k, 0.0) for k in EMOTION_KEYS}
        d["energy"] = self.energy
        return d

    def emotion_only_dict(self) -> Dict[str, float]:
        """只返回情绪维度（不含 energy），用于余弦相似度比较。"""
        return {k: getattr(self, k, 0.0) for k in EMOTION_KEYS}


# ─────────────────────────────────────────────
# State Engine 输出
# ─────────────────────────────────────────────


@dataclass
class CharacterState:
    """角色在某一时刻的完整状态描述。"""

    emotion: str = "calm"  # 主情绪标签
    energy: float = 0.0  # 0~1
    mouth_open: float = 0.0  # 0~1（0=闭口, 1=全开）
    blink_phase: float = 0.0  # 0~1（眨眼动画相位）
    gesture: int = 0  # 动作编号（0=无动作）
    expression_weights: Dict[str, float] = field(default_factory=dict)

    def clone(self) -> "CharacterState":
        return CharacterState(
            emotion=self.emotion,
            energy=self.energy,
            mouth_open=self.mouth_open,
            blink_phase=self.blink_phase,
            gesture=self.gesture,
            expression_weights=dict(self.expression_weights),
        )


# ─────────────────────────────────────────────
# Animation Engine 输出（插值后的状态）
# ─────────────────────────────────────────────


@dataclass
class AnimatedState:
    """经过插值平滑后的角色状态，直接供渲染器使用。"""

    emotion: str = "calm"
    energy: float = 0.0
    mouth_open: float = 0.0
    blink_phase: float = 0.0
    gesture: int = 0
    expression_weights: Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────
# 渲染帧描述
# ─────────────────────────────────────────────


@dataclass
class RenderFrame:
    """单帧渲染指令 — 直接指向一张完整的角色图片。"""

    frame_index: int = 0
    image_key: str = ""  # 图片键名，如 "calm_eo_mo"（calm状态、眼开、嘴开）


# ─────────────────────────────────────────────
# 角色配置
# ─────────────────────────────────────────────


@dataclass
class CharacterConfig:
    """
    角色配置。

    每个表情状态（emotion）对应 4 张图片：
      - 眼开 + 嘴开  (eo_mo)
      - 眼开 + 嘴闭  (eo_mc)
      - 眼闭 + 嘴开  (ec_mo)
      - 眼闭 + 嘴闭  (ec_mc)

    图片键名格式: "{emotion}_eo_mo" / "{emotion}_eo_mc" 等

    emotions 列表由用户上传素材后、AI 识别动态构建，不再预设。
    """

    name: str = "unnamed"
    resolution: tuple = (1080, 1920)  # (宽, 高)

    # 支持的表情状态列表 —— 动态构建，初始为空
    emotions: List[str] = field(default_factory=list)

    # 每个表情对应的情绪向量 {emotion_label: {calm: 0.3, happy: 0.5, ...}}
    # 由图片识别 AI 填充，用于和音频情绪向量做余弦相似度匹配
    emotion_vectors: Dict[str, Dict[str, float]] = field(default_factory=dict)

    mouth_threshold: float = 0.5  # mouth_open > threshold 则张嘴
    blink_interval: float = 3.0  # 眨眼间隔（秒）
    blink_duration: float = 0.15  # 单次眨眼时长（秒）

    @staticmethod
    def image_key(emotion: str, eye_open: bool, mouth_open: bool) -> str:
        """生成图片键名。"""
        eo = "eo" if eye_open else "ec"
        mo = "mo" if mouth_open else "mc"
        return f"{emotion}_{eo}_{mo}"

    def all_image_keys(self) -> List[str]:
        """返回该角色所有需要的图片键名。"""
        keys = []
        for emotion in self.emotions:
            for eye in [True, False]:
                for mouth in [True, False]:
                    keys.append(self.image_key(emotion, eye, mouth))
        return keys

    def add_emotion(
        self, emotion: str, emotion_vector: Optional[Dict[str, float]] = None
    ) -> None:
        """添加一个新的表情状态（去重）。可附带情绪向量。"""
        if emotion not in self.emotions:
            self.emotions.append(emotion)
        if emotion_vector:
            self.emotion_vectors[emotion] = emotion_vector

    def remove_emotion(self, emotion: str) -> None:
        """移除一个表情状态。"""
        if emotion in self.emotions:
            self.emotions.remove(emotion)
        self.emotion_vectors.pop(emotion, None)


# ─────────────────────────────────────────────
# 上传素材容器（Streamlit 运行时使用）
# ─────────────────────────────────────────────


class UploadedAssets:
    """
    存放用户通过 Streamlit 上传的 PIL Image。
    键名格式: "{emotion}_eo_mo" 等，与 CharacterConfig.image_key() 一致。
    不会持久化到磁盘——每次 Streamlit 会话重新上传。
    """

    def __init__(self):
        self.images: Dict[str, Any] = {}  # key → PIL Image (RGBA)

    def put(self, key: str, image) -> None:
        self.images[key] = image

    def get(self, key: str):
        return self.images.get(key)

    def has(self, key: str) -> bool:
        return key in self.images

    def is_complete(self, config: CharacterConfig) -> bool:
        """检查是否所有必需图片都已上传。"""
        return all(self.has(k) for k in config.all_image_keys())

    def missing_keys(self, config: CharacterConfig) -> List[str]:
        """返回尚未上传的图片键名列表。"""
        return [k for k in config.all_image_keys() if not self.has(k)]

    def put_emotion_group(self, emotion: str, images_dict: Dict[str, Any]) -> None:
        """
        一次性存入一组表情的 4 张图片。

        Args:
            emotion: 表情名称（如 "happy"）。
            images_dict: 键为 "eo_mo"/"eo_mc"/"ec_mo"/"ec_mc"，值为 PIL Image。
        """
        for suffix, img in images_dict.items():
            key = f"{emotion}_{suffix}"
            self.images[key] = img

    def remove_emotion_group(self, emotion: str) -> None:
        """移除一组表情的所有图片。"""
        keys_to_remove = [k for k in self.images if k.startswith(f"{emotion}_")]
        for k in keys_to_remove:
            del self.images[k]

    def get_emotion_group(self, emotion: str) -> Dict[str, Any]:
        """获取一组表情的所有图片。"""
        prefix = f"{emotion}_"
        return {k: v for k, v in self.images.items() if k.startswith(prefix)}

    def emotion_group_complete(self, emotion: str) -> bool:
        """检查一组表情的 4 张图片是否都有。"""
        for eye in [True, False]:
            for mouth in [True, False]:
                key = CharacterConfig.image_key(emotion, eye, mouth)
                if not self.has(key):
                    return False
        return True
