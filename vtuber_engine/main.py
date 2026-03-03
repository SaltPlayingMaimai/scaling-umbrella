"""
VTuber Engine — 主入口。

推荐使用方式：
  streamlit run vtuber_engine/streamlit_app.py

也支持命令行（需要提前在 assets/ 放好素材）：
  python -m vtuber_engine.main --mode audio --input voice.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ──────────────────── 路径设置 ────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="VTuber Engine — AI 驱动角色动画生成系统"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="启动 Streamlit 可视化界面（推荐）",
    )
    parser.add_argument(
        "--mode",
        choices=["audio", "tts"],
        default="audio",
        help="运行模式：audio=导入配音, tts=文字转语音",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="音频文件路径（audio 模式）",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        help="要转换为语音的文本（tts 模式）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/output.mp4",
        help="输出视频路径",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="视频帧率",
    )
    parser.add_argument(
        "--emotion-backend",
        choices=["rule", "openai"],
        default="rule",
        help="情绪识别后端",
    )

    args = parser.parse_args()

    # ──────────── 启动 Streamlit UI ────────────

    if args.ui:
        import subprocess

        app_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "streamlit_app.py",
        )
        subprocess.run(["streamlit", "run", app_path])
        return

    # ──────────── CLI 模式 ────────────

    if args.mode == "audio" and not args.input:
        parser.error("audio 模式需要 --input 参数指定音频文件")
    if args.mode == "tts" and not args.text:
        parser.error("tts 模式需要 --text 参数指定文本内容")

    from vtuber_engine.config.character_config import create_default_config

    config = create_default_config()

    # TTS
    audio_path = args.input
    if args.mode == "tts":
        audio_path = _run_tts(args.text)

    print(f"\n{'='*50}")
    print(f"[Main] VTuber Engine Pipeline (CLI)")
    print(f"  Mode   : {args.mode}")
    print(f"  Audio  : {audio_path}")
    print(f"  FPS    : {args.fps}")
    print(f"  Output : {args.output}")
    print(f"{'='*50}\n")
    print("[提示] CLI 模式需要在 assets/ 放好素材，推荐使用 Streamlit UI：")
    print("       streamlit run vtuber_engine/streamlit_app.py")
    print()

    t0 = time.time()

    # Step 1: Audio Analysis
    print("[1/6] Analyzing audio...")
    from vtuber_engine.audio.analyzer import AudioAnalyzer

    analyzer = AudioAnalyzer(fps=args.fps)
    audio_features = analyzer.analyze(audio_path)
    print(f"       {audio_features.frame_count} frames, {audio_features.duration:.1f}s")

    # Step 2: Emotion Engine
    print("[2/6] Running emotion analysis...")
    from vtuber_engine.audio.emotion_engine import EmotionEngine

    emotion_engine = EmotionEngine(backend=args.emotion_backend)
    emotion_vectors = emotion_engine.analyze(audio_features)
    dominant = emotion_vectors[0].dominant_emotion() if emotion_vectors else "calm"
    print(f"       Dominant emotion: {dominant}")

    # Step 3: State Engine
    print("[3/6] Computing character states...")
    from vtuber_engine.core.state_engine import StateEngine

    state_engine = StateEngine(config, fps=args.fps)
    states = state_engine.process(audio_features, emotion_vectors)
    print(f"       {len(states)} states generated")

    # Step 4: Animation Engine
    print("[4/6] Smoothing animation...")
    from vtuber_engine.core.animation_engine import AnimationEngine

    anim_engine = AnimationEngine(smoothing=0.25)
    animated_states = anim_engine.process(states)
    print(f"       {len(animated_states)} frames smoothed")

    # Step 5 & 6: Render + Export
    # 在 CLI 模式下需要 UploadedAssets（需要从文件加载）
    print("[5/6] Rendering frames...")
    print("[!] CLI 模式暂不支持自动加载素材，请使用 Streamlit UI。")
    print("    streamlit run vtuber_engine/streamlit_app.py")

    elapsed = time.time() - t0
    print(f"\n[Time] {elapsed:.1f}s")


def _run_tts(text: str) -> str:
    """使用 Edge TTS 将文本转为音频。"""
    import asyncio

    try:
        import edge_tts
    except ImportError:
        raise ImportError(
            "edge-tts is required for TTS mode. "
            "Install it with: pip install edge-tts"
        )

    output_path = os.path.join(PROJECT_ROOT, "output", "tts_audio.mp3")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    async def _generate():
        communicate = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")
        await communicate.save(output_path)

    print(f"[TTS] Generating audio for: {text[:50]}...")
    asyncio.run(_generate())
    print(f"[TTS] Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    main()
