"""
Video Exporter — 视频导出模块。

职责：
  - 将帧序列写入临时目录
  - 调用 ffmpeg 编码为视频
  - 合并音轨
  - 输出最终绿幕视频文件

性能优化：
  - 自动检测硬件编码器 (NVENC / QSV / AMF)，优先使用 GPU 加速
  - 并行帧→bytes 转换（多线程）
  - 缓冲批量写入 ffmpeg stdin（减少 syscall 开销）

不做：
  - 图像合成（交给 Renderer）
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None


# ─── 硬件编码器优先级 ───
_HW_ENCODERS = [
    ("h264_nvenc", ["-preset", "p4", "-tune", "ll", "-rc", "vbr", "-cq", "23"]),
    ("h264_qsv", ["-preset", "fast", "-global_quality", "23"]),
    ("h264_amf", ["-quality", "speed", "-rc", "cqp", "-qp_i", "23", "-qp_p", "23"]),
]


class VideoExporter:
    """将帧序列 + 音频导出为 MP4 视频。"""

    def __init__(
        self,
        fps: int = 30,
        output_dir: str = "output",
        ffmpeg_path: str = "ffmpeg",
        hw_accel: bool = True,
        max_workers: int = 4,
        pipe_buffer_frames: int = 60,
    ):
        """
        Args:
            fps: 输出视频帧率。
            output_dir: 输出目录。
            ffmpeg_path: ffmpeg 可执行文件路径。
            hw_accel: 是否尝试使用硬件编码（NVENC/QSV/AMF）。
            max_workers: 帧→bytes 并行转换线程数。
            pipe_buffer_frames: 每攒满多少帧才批量写入 ffmpeg stdin。
        """
        self.fps = fps
        self.output_dir = output_dir
        self.ffmpeg_path = ffmpeg_path
        self.hw_accel = hw_accel
        self.max_workers = max_workers
        self.pipe_buffer_frames = pipe_buffer_frames

        # 缓存检测结果
        self._hw_encoder: Optional[Tuple[str, list]] = None
        self._hw_checked = False

    # ──────────────────── 公共接口 ────────────────────

    def export(
        self,
        frames: list,
        audio_bytes: Optional[bytes] = None,
        audio_suffix: str = ".wav",
        output_filename: str = "output.mp4",
        progress_callback=None,
    ) -> str:
        """
        导出视频。

        Args:
            frames: PIL Image 列表（帧序列）。
            audio_bytes: 音频文件的内容（可选）。
            audio_suffix: 音频文件后缀（如 ".wav" ".mp3"）。
            output_filename: 输出文件名。
            progress_callback: 可选回调 fn(current_frame, total_frames)。

        Returns:
            输出视频的完整路径。
        """
        self._ensure_ffmpeg()
        os.makedirs(self.output_dir, exist_ok=True)

        output_path = os.path.join(self.output_dir, output_filename)

        # 使用临时目录存放帧图片
        with tempfile.TemporaryDirectory(prefix="vtuber_frames_") as tmp_dir:
            # 通过 stdin pipe 直接管道到 ffmpeg
            video_only = os.path.join(tmp_dir, "video_only.mp4")
            self._encode_video_pipe(frames, video_only, progress_callback)

            # 2. 合并音轨
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

    def get_encoder_info(self) -> str:
        """返回当前使用的编码器信息（用于 UI 展示）。"""
        hw = self._detect_hw_encoder()
        if hw:
            return f"GPU 硬件编码 ({hw[0]})"
        return "CPU 软编码 (libx264)"

    # ──────────────────── 硬件编码检测 ────────────────────

    def _detect_hw_encoder(self) -> Optional[Tuple[str, list]]:
        """检测可用的硬件编码器（NVENC > QSV > AMF），返回 (编码器名, 参数列表) 或 None。"""
        if self._hw_checked:
            return self._hw_encoder
        self._hw_checked = True

        if not self.hw_accel:
            self._hw_encoder = None
            return None

        for encoder_name, encoder_params in _HW_ENCODERS:
            try:
                result = subprocess.run(
                    [self.ffmpeg_path, "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if encoder_name in result.stdout:
                    # 进一步验证能否实际使用
                    test_cmd = [
                        self.ffmpeg_path,
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "nullsrc=s=64x64:d=0.1",
                        "-c:v",
                        encoder_name,
                        "-f",
                        "null",
                        "-",
                    ]
                    test_r = subprocess.run(
                        test_cmd,
                        capture_output=True,
                        timeout=15,
                    )
                    if test_r.returncode == 0:
                        self._hw_encoder = (encoder_name, encoder_params)
                        print(f"[VideoExporter] 硬件编码器检测成功: {encoder_name}")
                        return self._hw_encoder
            except Exception:
                continue

        print("[VideoExporter] 未检测到硬件编码器，使用 libx264 软编码")
        self._hw_encoder = None
        return None

    # ──────────────────── 内部逻辑 ────────────────────

    @staticmethod
    def _frame_to_rgb_bytes(frame) -> bytes:
        """将单个 PIL Image 转为 RGB bytes（用于并行 map）。"""
        if np is not None:
            arr = np.asarray(frame.convert("RGB"))
            return arr.tobytes()
        return frame.convert("RGB").tobytes()

    def _encode_video_pipe(
        self,
        frames: list,
        output_path: str,
        progress_callback=None,
    ) -> None:
        """
        通过 stdin pipe 将帧列表直接管道给 ffmpeg。

        优化：
          1. 自动选择硬件编码器（NVENC/QSV/AMF）或 libx264
          2. 多线程并行将 PIL→RGB bytes
          3. 缓冲批量写入 stdin（减少系统调用）
        """
        if not frames:
            raise ValueError("No frames to encode")

        w, h = frames[0].size

        # 选择编码器
        hw = self._detect_hw_encoder()
        if hw:
            encoder_name, encoder_params = hw
            codec_args = ["-c:v", encoder_name] + encoder_params
        else:
            codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{w}x{h}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            *codec_args,
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        total = len(frames)
        buf_size = self.pipe_buffer_frames
        frame_byte_size = w * h * 3  # RGB24

        try:
            # 分批处理：每批先并行转换，再批量写入
            for batch_start in range(0, total, buf_size):
                batch_end = min(batch_start + buf_size, total)
                batch = frames[batch_start:batch_end]

                # 并行 PIL → bytes（IO 释放 GIL，Pillow tobytes 同理）
                if len(batch) > 1 and self.max_workers > 1:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                        raw_chunks = list(pool.map(self._frame_to_rgb_bytes, batch))
                else:
                    raw_chunks = [self._frame_to_rgb_bytes(f) for f in batch]

                # 合并为单次大写入（减少 syscall）
                blob = b"".join(raw_chunks)
                proc.stdin.write(blob)

                if progress_callback:
                    progress_callback(batch_end, total)

            proc.stdin.close()
            _, stderr = proc.communicate(timeout=600)
        except Exception as e:
            proc.kill()
            proc.communicate()
            raise RuntimeError(f"ffmpeg pipe encoding failed: {e}")

        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encoding failed:\n{stderr.decode(errors='replace')}"
            )

    def _write_frames(self, frames: list, tmp_dir: str) -> None:
        """备用：将帧列表写入临时目录为 PNG 序列。"""
        for i, frame in enumerate(frames):
            filepath = os.path.join(tmp_dir, f"frame_{i:06d}.png")
            if hasattr(frame, "convert"):
                frame.convert("RGB").save(filepath)
            else:
                raise TypeError(f"Expected PIL Image, got {type(frame)}")

    def _encode_video(self, frames_dir: str, output_path: str) -> None:
        """备用：调用 ffmpeg 将帧序列编码为视频（从磁盘读取 PNG）。"""
        input_pattern = os.path.join(frames_dir, "frame_%06d.png")
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-framerate",
            str(self.fps),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "23",
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
