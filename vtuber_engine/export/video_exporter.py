"""
Video Exporter — 视频导出模块。

职责：
  - 将帧序列写入临时目录
  - 调用 ffmpeg 编码为视频
  - 合并音轨
  - 输出最终绿幕视频文件

不做：
  - 图像合成（交给 Renderer）
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

try:
    from PIL import Image
except ImportError:
    Image = None


class VideoExporter:
    """将帧序列 + 音频导出为 MP4 视频。"""

    def __init__(
        self,
        fps: int = 30,
        output_dir: str = "output",
        ffmpeg_path: str = "ffmpeg",
    ):
        """
        Args:
            fps: 输出视频帧率。
            output_dir: 输出目录。
            ffmpeg_path: ffmpeg 可执行文件路径。
        """
        self.fps = fps
        self.output_dir = output_dir
        self.ffmpeg_path = ffmpeg_path

    # ──────────────────── 公共接口 ────────────────────

    def export(
        self,
        frames: list,
        audio_bytes: Optional[bytes] = None,
        audio_suffix: str = ".wav",
        output_filename: str = "output.mp4",
    ) -> str:
        """
        导出视频。

        Args:
            frames: PIL Image 列表（帧序列）。
            audio_bytes: 音频文件的内容（可选）。尺对内存，处理时临时落盘到系统临时目录再删除。
            audio_suffix: 音频文件后缀（如 ".wav" ".mp3"）。
            output_filename: 输出文件名。

        Returns:
            输出视频的完整路径。
        """
        self._ensure_ffmpeg()
        os.makedirs(self.output_dir, exist_ok=True)

        output_path = os.path.join(self.output_dir, output_filename)

        # 使用临时目录存放帧图片
        with tempfile.TemporaryDirectory(prefix="vtuber_frames_") as tmp_dir:
            # 1. 写入帧
            self._write_frames(frames, tmp_dir)

            # 2. 编码视频
            video_only = os.path.join(tmp_dir, "video_only.mp4")
            self._encode_video(tmp_dir, video_only)

            # 3. 合并音轨（音频字节临时落盘，合并完就删除）
            if audio_bytes:
                fd, tmp_audio = tempfile.mkstemp(
                    suffix=audio_suffix, prefix="vtuber_audio_"
                )
                try:
                    os.close(fd)
                    with open(tmp_audio, "wb") as af:
                        af.write(audio_bytes)
                    self._merge_audio(video_only, tmp_audio, output_path)
                finally:
                    if os.path.exists(tmp_audio):
                        os.remove(tmp_audio)
            else:
                shutil.copy2(video_only, output_path)

        print(f"[VideoExporter] Exported: {output_path}")
        return output_path

    # ──────────────────── 内部逻辑 ────────────────────

    def _write_frames(self, frames: list, tmp_dir: str) -> None:
        """将帧列表写入临时目录为 PNG 序列。"""
        for i, frame in enumerate(frames):
            filepath = os.path.join(tmp_dir, f"frame_{i:06d}.png")
            # 转换为 RGB（视频不需要 alpha 通道，绿幕已在背景中）
            if hasattr(frame, "convert"):
                rgb_frame = frame.convert("RGB")
                rgb_frame.save(filepath)
            else:
                raise TypeError(f"Expected PIL Image, got {type(frame)}")

    def _encode_video(self, frames_dir: str, output_path: str) -> None:
        """调用 ffmpeg 将帧序列编码为视频。"""
        input_pattern = os.path.join(frames_dir, "frame_%06d.png")

        cmd = [
            self.ffmpeg_path,
            "-y",  # 覆盖输出
            "-framerate",
            str(self.fps),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",  # 兼容性
            "-preset",
            "medium",
            "-crf",
            "18",  # 高质量
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg encoding failed:\n{result.stderr}")

    def _merge_audio(self, video_path: str, audio_path: str, output_path: str) -> None:
        """合并视频和音轨。"""
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",  # 以较短的流为准
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio merge failed:\n{result.stderr}")

    def _ensure_ffmpeg(self) -> None:
        """检查 ffmpeg 是否可用。"""
        try:
            subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                timeout=10,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"ffmpeg not found at '{self.ffmpeg_path}'. "
                "Please install ffmpeg and add it to PATH."
            )
