"""
VTuber Engine — Streamlit 可视化界面。

功能：
  1. 上传角色素材（每个表情 4 张图）
  2. 上传音频 / 使用 TTS
  3. 配置参数
  4. 一键生成绿幕视频
  5. 可视化音频分析 & 情绪结果

运行方式：
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
from PIL import Image

# ──────────────────── 路径 ────────────────────

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from vtuber_engine.models.data_models import (
    CharacterConfig,
    UploadedAssets,
)

# ──────────────────── 页面配置 ────────────────────

st.set_page_config(
    page_title="VTuber Engine",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Session State 初始化
# ──────────────────────────────────────────────


def _init_session():
    """初始化 session state。"""
    if "assets" not in st.session_state:
        st.session_state.assets = UploadedAssets()
    if "config" not in st.session_state:
        st.session_state.config = CharacterConfig(
            name="my_oc",
            resolution=(1080, 1920),
            emotions=["calm", "excited", "panic"],
        )
    if "audio_features" not in st.session_state:
        st.session_state.audio_features = None
    if "emotion_vectors" not in st.session_state:
        st.session_state.emotion_vectors = None
    if "video_bytes" not in st.session_state:
        st.session_state.video_bytes = None
    # 音频只存内存字节，不写入项目目录
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None
    if "audio_suffix" not in st.session_state:
        st.session_state.audio_suffix = ".wav"


_init_session()


# ──────────────────────────────────────────────
# 侧栏：参数配置
# ──────────────────────────────────────────────


def _sidebar_config():
    """侧栏参数设置。"""
    st.sidebar.title("⚙️ 参数设置")

    st.sidebar.subheader("角色设置")
    name = st.sidebar.text_input("角色名称", value=st.session_state.config.name)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        width = st.sidebar.number_input(
            "宽度 (px)",
            value=st.session_state.config.resolution[0],
            min_value=100,
            step=10,
        )
    with col2:
        height = st.sidebar.number_input(
            "高度 (px)",
            value=st.session_state.config.resolution[1],
            min_value=100,
            step=10,
        )

    st.sidebar.subheader("表情状态列表")
    emotions_text = st.sidebar.text_area(
        "每行一个表情名称",
        value="\n".join(st.session_state.config.emotions),
        help="例如：calm、excited、panic。每种表情需要上传4张差分图。",
    )
    emotions = [e.strip() for e in emotions_text.strip().split("\n") if e.strip()]

    st.sidebar.subheader("动画参数")
    mouth_threshold = st.sidebar.slider(
        "嘴型阈值",
        0.0,
        1.0,
        value=st.session_state.config.mouth_threshold,
        step=0.05,
        help="音量超过此值时选择「嘴开」的图片",
    )
    blink_interval = st.sidebar.slider(
        "眨眼间隔（秒）",
        0.5,
        10.0,
        value=st.session_state.config.blink_interval,
        step=0.5,
    )
    blink_duration = st.sidebar.slider(
        "眨眼时长（秒）",
        0.05,
        0.5,
        value=st.session_state.config.blink_duration,
        step=0.05,
    )

    fps = st.sidebar.selectbox("视频帧率", [24, 30, 60], index=1)
    smoothing = st.sidebar.slider(
        "动画平滑度",
        0.05,
        1.0,
        value=0.25,
        step=0.05,
        help="越小越平滑（响应慢），越大越灵敏（可能抖动）",
    )

    emotion_backend = st.sidebar.selectbox(
        "情绪识别后端",
        ["rule", "openai"],
        help="rule = 基于规则（免费），openai = 调用 AI API",
    )

    # 更新配置
    st.session_state.config = CharacterConfig(
        name=name,
        resolution=(int(width), int(height)),
        emotions=emotions,
        mouth_threshold=mouth_threshold,
        blink_interval=blink_interval,
        blink_duration=blink_duration,
    )

    return fps, smoothing, emotion_backend


# ──────────────────────────────────────────────
# Tab 1：上传素材
# ──────────────────────────────────────────────


def _tab_upload_assets():
    """素材上传界面。"""
    config = st.session_state.config
    assets = st.session_state.assets

    st.markdown(
        """
        ### 📌 上传规则
        每种 **表情状态** 需要上传 **4 张 PNG 图片**（透明背景最佳）：

        | 图片 | 眼睛 | 嘴巴 | 说明 |
        |------|------|------|------|
        | 图1 | 👁️ 睁开 | 👄 张开 | 说话时 + 眼睛正常 |
        | 图2 | 👁️ 睁开 | 👄 闭合 | 不说话时 + 眼睛正常 |
        | 图3 | 👁️ 闭合 | 👄 张开 | 说话时 + 眨眼瞬间 |
        | 图4 | 👁️ 闭合 | 👄 闭合 | 不说话时 + 眨眼瞬间 |

        > 💡 素材只存在内存中，关闭页面即消失，不会上传到任何地方。
        """
    )

    if not config.emotions:
        st.warning("请先在侧栏设置表情状态列表。")
        return

    for emotion in config.emotions:
        st.subheader(f"🎭 表情: {emotion}")

        cols = st.columns(4)
        combos = [
            ("eo_mo", "眼开 + 嘴开", True, True),
            ("eo_mc", "眼开 + 嘴闭", True, False),
            ("ec_mo", "眼闭 + 嘴开", False, True),
            ("ec_mc", "眼闭 + 嘴闭", False, False),
        ]

        for col, (suffix, label, eye_open, mouth_open) in zip(cols, combos):
            key = CharacterConfig.image_key(emotion, eye_open, mouth_open)
            with col:
                uploaded = st.file_uploader(
                    label,
                    type=["png", "jpg", "jpeg", "webp"],
                    key=f"upload_{key}",
                )
                if uploaded is not None:
                    img = Image.open(uploaded).convert("RGBA")
                    assets.put(key, img)

                # 显示已上传的缩略图
                if assets.has(key):
                    st.image(assets.get(key), width=150, caption=f"✅ {key}")
                else:
                    st.caption(f"⬜ 未上传")

    # 汇总
    st.divider()
    missing = assets.missing_keys(config)
    total_needed = len(config.all_image_keys())
    uploaded_count = total_needed - len(missing)

    if missing:
        st.warning(f"还需上传 {len(missing)}/{total_needed} 张图片")
        with st.expander("查看缺少的图片"):
            for k in missing:
                st.text(f"  ❌ {k}")
    else:
        st.success(f"✅ 所有 {total_needed} 张素材已上传完成！")


# ──────────────────────────────────────────────
# Tab 2：上传音频 / TTS
# ──────────────────────────────────────────────


def _tab_audio():
    """音频输入界面。"""

    mode = st.radio(
        "输入方式", ["🎙️ 上传配音文件", "💬 文字转语音 (TTS)"], horizontal=True
    )

    if mode == "🎙️ 上传配音文件":
        uploaded_audio = st.file_uploader(
            "上传音频文件",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            key="audio_upload",
        )
        if uploaded_audio is not None:
            # 只存内存，不写入项目目录
            st.session_state.audio_bytes = uploaded_audio.getvalue()
            st.session_state.audio_suffix = Path(uploaded_audio.name).suffix
            st.audio(uploaded_audio)
            st.success(f"✅ 音频已加载：{uploaded_audio.name}（仅位于内存）")

    else:  # TTS
        tts_text = st.text_area(
            "输入要朗读的文字", height=100, placeholder="在这里输入文字..."
        )
        tts_voice = st.selectbox(
            "语音选择",
            [
                "zh-CN-XiaoxiaoNeural",
                "zh-CN-YunxiNeural",
                "zh-CN-YunjianNeural",
                "zh-TW-HsiaoChenNeural",
                "en-US-JennyNeural",
                "en-US-GuyNeural",
                "ja-JP-NanamiNeural",
            ],
        )
        if st.button("🔊 生成语音", type="primary"):
            if not tts_text.strip():
                st.error("请输入文字内容。")
            else:
                with st.spinner("正在生成语音..."):
                    audio_bytes = _generate_tts(tts_text, tts_voice)
                    st.session_state.audio_bytes = audio_bytes
                    st.session_state.audio_suffix = ".mp3"
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success("✅ 语音生成完成！（仅位于内存）")


def _generate_tts(text: str, voice: str) -> bytes:
    """使用 Edge TTS 生成语音，返回音频字节（不落盘）。"""
    import asyncio

    try:
        import edge_tts
    except ImportError:
        st.error("请先安装 edge-tts: `pip install edge-tts`")
        st.stop()

    buf = io.BytesIO()

    async def _gen():
        communicate = edge_tts.Communicate(text, voice=voice)
        # stream() 返回 (type, data) 元组，收集 audio 类型的块
        async for chunk_type, chunk_data in communicate.stream():
            if chunk_type == "audio" and chunk_data:
                buf.write(chunk_data)

    asyncio.run(_gen())
    return buf.getvalue()


# ──────────────────────────────────────────────
# Tab 3：生成视频
# ──────────────────────────────────────────────


def _tab_generate(fps: int, smoothing: float, emotion_backend: str):
    """生成视频界面。"""
    config = st.session_state.config
    assets = st.session_state.assets

    # 前置检查
    ready = True
    checks = []

    missing = assets.missing_keys(config)
    if missing:
        checks.append(f"❌ 还差 {len(missing)} 张素材未上传")
        ready = False
    else:
        checks.append(f"✅ 所有素材已就绪")

    if st.session_state.audio_bytes:
        size_kb = len(st.session_state.audio_bytes) / 1024
        checks.append(f"✅ 音频已加载（{size_kb:.0f} KB，仅位于内存）")
    else:
        checks.append(f"❌ 请先上传音频或生成 TTS")
        ready = False

    for c in checks:
        st.markdown(c)

    st.divider()

    if not ready:
        st.info("请先完成上面的步骤。")
        return

    if st.button("🎬 开始生成视频", type="primary", use_container_width=True):
        _run_pipeline(fps, smoothing, emotion_backend)


def _run_pipeline(fps: int, smoothing: float, emotion_backend: str):
    """执行完整生成管线。"""
    config = st.session_state.config
    assets = st.session_state.assets
    audio_bytes: bytes = st.session_state.audio_bytes
    audio_suffix: str = st.session_state.audio_suffix

    progress = st.progress(0, text="准备中...")

    try:
        # Step 1: 音频分析（传入内存字节，不需要文件路径）
        progress.progress(5, text="🔍 分析音频特征...")
        from vtuber_engine.audio.analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer(fps=fps)
        audio_features = analyzer.analyze(audio_bytes)
        st.session_state.audio_features = audio_features

        # Step 2: 情绪分析
        progress.progress(20, text="🧠 AI 情绪识别...")
        from vtuber_engine.audio.emotion_engine import EmotionEngine

        emotion_engine = EmotionEngine(backend=emotion_backend)
        emotion_vectors = emotion_engine.analyze(audio_features)
        st.session_state.emotion_vectors = emotion_vectors

        # Step 3: 状态计算
        progress.progress(35, text="🎯 计算角色状态...")
        from vtuber_engine.core.state_engine import StateEngine

        state_engine = StateEngine(config, fps=fps)
        states = state_engine.process(audio_features, emotion_vectors)

        # Step 4: 动画平滑
        progress.progress(45, text="✨ 动画插值平滑...")
        from vtuber_engine.core.animation_engine import AnimationEngine

        anim_engine = AnimationEngine(smoothing=smoothing)
        animated_states = anim_engine.process(states)

        # Step 5: 渲染帧
        progress.progress(55, text="🖼️ 渲染帧图像...")
        from vtuber_engine.render.renderer import Renderer

        renderer = Renderer(config, assets)

        total_frames = len(animated_states)
        frames = []
        for i, state in enumerate(animated_states):
            frame = renderer.render_frame(state)
            frames.append(frame)
            if i % 10 == 0:
                pct = 55 + int((i / total_frames) * 30)
                progress.progress(pct, text=f"🖼️ 渲染帧 {i}/{total_frames}...")

        # Step 6: 导出视频
        progress.progress(90, text="🎬 导出视频...")
        from vtuber_engine.export.video_exporter import VideoExporter

        output_dir = tempfile.mkdtemp(prefix="vtuber_output_")
        exporter = VideoExporter(fps=fps, output_dir=output_dir)
        output_path = exporter.export(
            frames=frames,
            audio_bytes=audio_bytes,
            audio_suffix=audio_suffix,
            output_filename="output.mp4",
        )

        # 读取视频到内存
        with open(output_path, "rb") as f:
            st.session_state.video_bytes = f.read()

        progress.progress(100, text="✅ 完成！")

        # 显示结果
        st.balloons()
        st.success(
            f"🎉 视频生成完成！ 共 {total_frames} 帧，"
            f"时长 {audio_features.duration:.1f} 秒"
        )

    except Exception as e:
        progress.empty()
        st.error(f"生成失败: {e}")
        import traceback

        st.code(traceback.format_exc())


# ──────────────────────────────────────────────
# Tab 4：预览 & 下载
# ──────────────────────────────────────────────


def _tab_preview():
    """视频预览和下载。"""

    if st.session_state.video_bytes:
        st.video(st.session_state.video_bytes)

        st.download_button(
            label="⬇️ 下载视频 (MP4)",
            data=st.session_state.video_bytes,
            file_name=f"{st.session_state.config.name}_output.mp4",
            mime="video/mp4",
            use_container_width=True,
            type="primary",
        )
    else:
        st.info("还没有生成视频。请先到「生成视频」标签页生成。")

    # 音频分析可视化
    if st.session_state.audio_features is not None:
        _show_analysis()


def _show_analysis():
    """显示音频分析和情绪识别结果的可视化图表。"""
    import numpy as np

    features = st.session_state.audio_features
    emotions = st.session_state.emotion_vectors

    st.subheader("📊 音频分析结果")

    # 音量波形
    if features.volume:
        st.markdown("**音量 (RMS)**")
        st.line_chart(features.volume, height=150)

    # 基频
    if features.pitch:
        st.markdown("**基频 (Hz)**")
        st.line_chart(features.pitch, height=150)

    # 说话段
    if features.is_speaking:
        speaking_float = [1.0 if s else 0.0 for s in features.is_speaking]
        st.markdown("**语音活动 (VAD)**")
        st.bar_chart(speaking_float, height=100)

    # 情绪分析
    if emotions:
        st.subheader("🧠 情绪分析结果")

        # 取中间帧的情绪作为代表
        mid = len(emotions) // 2
        sample = emotions[mid]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**情绪权重（中间时刻）**")
            emotion_data = {
                "情绪": ["calm", "excited", "panic", "sad", "angry", "happy"],
                "权重": [
                    sample.calm,
                    sample.excited,
                    sample.panic,
                    sample.sad,
                    sample.angry,
                    sample.happy,
                ],
            }
            st.bar_chart(emotion_data, x="情绪", y="权重", height=250)

        with col2:
            st.markdown("**主要情绪**")
            dominant = sample.dominant_emotion()
            st.metric("主情绪", dominant)
            st.metric("能量", f"{sample.energy:.2f}")

        # 能量曲线
        if len(emotions) > 1:
            energy_curve = [e.energy for e in emotions]
            st.markdown("**能量曲线（逐帧）**")
            st.line_chart(energy_curve, height=150)


# ──────────────────────────────────────────────
# 主界面
# ──────────────────────────────────────────────


def main():
    st.title("🎭 VTuber Engine")
    st.caption("AI 驱动 · 角色动画生成 · 绿幕视频输出")

    fps, smoothing, emotion_backend = _sidebar_config()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📁 上传素材",
            "🎵 音频输入",
            "🎬 生成视频",
            "📺 预览 & 下载",
        ]
    )

    with tab1:
        _tab_upload_assets()

    with tab2:
        _tab_audio()

    with tab3:
        _tab_generate(fps, smoothing, emotion_backend)

    with tab4:
        _tab_preview()


if __name__ == "__main__":
    main()
