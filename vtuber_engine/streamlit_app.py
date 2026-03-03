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
from dotenv import load_dotenv

# ──────────────────── 路径 & 环境变量 ────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env（项目根目录）
load_dotenv(PROJECT_ROOT / ".env")

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
            emotions=[],  # 动态构建，不再预设
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
    # 逐张上传模式：各 slot 已分配的图片
    if "pending_slot_images" not in st.session_state:
        st.session_state.pending_slot_images = {
            "eo_mo": None, "eo_mc": None, "ec_mo": None, "ec_mc": None
        }
    # 当前正在上传(尚未分配)的单张图
    if "pending_single_image" not in st.session_state:
        st.session_state.pending_single_image = None
    if "pending_single_file_id" not in st.session_state:
        st.session_state.pending_single_file_id = None
    # AI 对当前单张图的分类结果
    if "pending_single_result" not in st.session_state:
        st.session_state.pending_single_result = None  # {"assigned_slot": ..., "probabilities": {...}}
    # 用于重置单张文件上传框
    if "single_upload_round" not in st.session_state:
        st.session_state.single_upload_round = 0
    # upload_round 用于 AI 生成完成后重置文件上传框
    if "upload_round" not in st.session_state:
        st.session_state.upload_round = 0
    # AI 模型选择
    if "text_model" not in st.session_state:
        st.session_state.text_model = "qwen3.5-flash"  # 情緒分析文字模型
    if "vision_model" not in st.session_state:
        st.session_state.vision_model = (
            "qwen-vl-max"  # 图片识别视觉模型（视觉理解模型）
        )


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

    # 显示当前已注册的表情列表（只读，由上传流程驱动）
    st.sidebar.subheader("已注册表情")
    if st.session_state.config.emotions:
        for emo in st.session_state.config.emotions:
            st.sidebar.markdown(f"  🎭 `{emo}`")
    else:
        st.sidebar.caption("暂无表情。请到「上传素材」标签页添加。")

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
        ["qwen", "rule", "openai"],
        index=0,
        help="qwen = 通义千问，rule = 基于规则（免费），openai = OpenAI API",
    )

    st.sidebar.subheader("🖼️ 图片识别后端")
    vision_backend = st.sidebar.selectbox(
        "表情识别后端",
        ["qwen", "openai"],
        index=0,
        help="上传素材时用哪个 AI 识别情緒：通义千问 Qwen VL 或 OpenAI GPT-4o",
    )

    # 后端切换时自动重置为该后端的默认模型
    _TEXT_DEFAULTS = {"qwen": "qwen3.5-flash", "openai": "gpt-4o-mini", "rule": ""}
    _VISION_DEFAULTS = {"qwen": "qwen-vl-max", "openai": "gpt-4o"}
    prev_text_backend = st.session_state.get("_prev_text_backend", emotion_backend)
    prev_vision_backend = st.session_state.get("_prev_vision_backend", vision_backend)
    if emotion_backend != prev_text_backend:
        st.session_state.text_model = _TEXT_DEFAULTS.get(emotion_backend, "")
        st.session_state["_prev_text_backend"] = emotion_backend
    if vision_backend != prev_vision_backend:
        st.session_state.vision_model = _VISION_DEFAULTS.get(vision_backend, "")
        st.session_state["_prev_vision_backend"] = vision_backend

    # 更新配置（保留现有 emotions 列表，不覆盖）
    cfg = st.session_state.config
    cfg.name = name
    cfg.resolution = (int(width), int(height))
    cfg.mouth_threshold = mouth_threshold
    cfg.blink_interval = blink_interval
    cfg.blink_duration = blink_duration

    return fps, smoothing, emotion_backend, vision_backend


# ──────────────────────────────────────────────
# Tab 1：上传素材
# ──────────────────────────────────────────────


def _tab_upload_assets():
    """素材上传界面 — 逐张上传 + AI 返回各 slot 概率 + 手动确认分配。"""
    config = st.session_state.config
    assets = st.session_state.assets

    st.markdown(
        """
        ### 📌 上传规则
        **逐张**上传角色差分图，AI 会对每张分别识别眼睛/嘴巴状态（并返回各状态的概率）。
        分配完 1-4 张后，输入情绪名称并点击「确认添加」。

        > 💡 差分图通常包含：眼开嘴开、眼开嘴闭、眼闭嘴开、眼闭嘴闭 四种组合。
        > 图片只存在内存中，关闭页面即消失，不会上传到任何地方。
        """
    )

    # ── 已有表情组展示 ──
    if config.emotions:
        st.subheader("✅ 已注册的表情组")
        for emotion in config.emotions:
            with st.expander(f"🎭 {emotion}", expanded=False):
                cols = st.columns(4)
                suffixes = [
                    ("eo_mo", "眼开+嘴开"),
                    ("eo_mc", "眼开+嘴闭"),
                    ("ec_mo", "眼闭+嘴开"),
                    ("ec_mc", "眼闭+嘴闭"),
                ]
                for col, (suffix, label) in zip(cols, suffixes):
                    key = f"{emotion}_{suffix}"
                    with col:
                        if assets.has(key):
                            st.image(assets.get(key), width=120, caption=label)
                        else:
                            st.caption(f"⬜ {label}")

                with st.expander("🔍 已存储的 keys", expanded=False):
                    all_slots = ["eo_mo", "eo_mc", "ec_mo", "ec_mc"]
                    for slot in all_slots:
                        k = f"{emotion}_{slot}"
                        status = "✅ 已存储" if assets.has(k) else "❌ 缺失"
                        st.caption(f"`{k}` — {status}")

                if st.button(f"🗑️ 删除「{emotion}」表情组", key=f"del_{emotion}"):
                    config.remove_emotion(emotion)
                    assets.remove_emotion_group(emotion)
                    st.rerun()

    st.divider()

    # ── 新增表情组：逐张上传 ──
    st.subheader("➕ 添加新表情组")

    slot_labels_cn = {
        "eo_mo": "眼开+嘴开",
        "eo_mc": "眼开+嘴闭",
        "ec_mo": "眼闭+嘴开",
        "ec_mc": "眼闭+嘴闭",
    }
    slot_images = st.session_state.pending_slot_images  # {slot: PIL.Image or None}

    # ── 当前已分配的四个槽位展示 ──
    st.markdown("🎯 **待确认的槽位**")
    slot_cols = st.columns(4)
    for col, (slot, cn_lbl) in zip(slot_cols, slot_labels_cn.items()):
        with col:
            img = slot_images.get(slot)
            if img is not None:
                st.image(img, width=130, caption=f"✅ {cn_lbl}")
                if st.button("🗑️ 清除", key=f"clear_slot_{slot}"):
                    st.session_state.pending_slot_images[slot] = None
                    st.rerun()
            else:
                st.caption(f"⬜ {cn_lbl}")
                st.caption("_(未分配)_")

    assigned_slots = [s for s, v in slot_images.items() if v is not None]
    filled_count = len(assigned_slots)

    st.divider()

    # ── 单张上传区域 ──
    st.markdown("📤 **上传一张图片**")
    single_upload_round = st.session_state.single_upload_round
    uploaded_file = st.file_uploader(
        "选择差分图（每次上传一张）",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"single_upload_{single_upload_round}",
    )

    if uploaded_file is not None:
        if st.session_state.pending_single_file_id != uploaded_file.file_id:
            st.session_state.pending_single_image = Image.open(uploaded_file).convert("RGBA")
            st.session_state.pending_single_file_id = uploaded_file.file_id
            st.session_state.pending_single_result = None  # 新图片就清除旧结果

    current_img = st.session_state.pending_single_image

    if current_img is not None:
        col_img, col_ai = st.columns([1, 2])
        with col_img:
            st.image(current_img, width=160, caption="待分类图片")
        with col_ai:
            ai_btn = st.button(
                "🤖 AI 识别此图",
                type="primary",
                use_container_width=True,
            )
            if ai_btn:
                _classify_single_image(current_img)

        # ── 展示 AI 分类结果 ──
        result = st.session_state.pending_single_result
        if result is not None:
            st.markdown("**📊 AI 分类概率**")
            probs = result["probabilities"]
            ai_suggested = result["assigned_slot"]

            prob_cols = st.columns(4)
            for col, (slot, cn_lbl) in zip(prob_cols, slot_labels_cn.items()):
                p = probs.get(slot, 0.0)
                already_filled = slot_images.get(slot) is not None
                badge = "🟩" if slot == ai_suggested else ("🟥" if already_filled else "")
                with col:
                    st.metric(
                        label=f"{cn_lbl} {badge}",
                        value=f"{p*100:.1f}%",
                        delta="AI推荐" if slot == ai_suggested else "",
                    )

            slot_options = list(slot_labels_cn.keys())
            slot_display = [f"{slot_labels_cn[s]} ({s})" for s in slot_options]
            default_idx = slot_options.index(ai_suggested) if ai_suggested in slot_options else 0
            chosen_display = st.selectbox(
                "分配到哪个槽位？（可覆盖 AI 建议）",
                slot_display,
                index=default_idx,
                key="slot_override_select",
            )
            chosen_slot = slot_options[slot_display.index(chosen_display)]

            assign_btn = st.button(
                f"✅ 分配到「{slot_labels_cn[chosen_slot]}」",
                type="primary",
                use_container_width=True,
            )
            if assign_btn:
                st.session_state.pending_slot_images[chosen_slot] = current_img
                st.session_state.pending_single_image = None
                st.session_state.pending_single_file_id = None
                st.session_state.pending_single_result = None
                st.session_state.single_upload_round += 1
                st.rerun()
    else:
        st.info("请上传一张图片，然后点击「AI 识别此图」。")

    st.divider()

    # ── 情绪名 + 确认添加 ──
    st.markdown("**📄 最终确认**")
    manual_label = st.text_input(
        "情绪名称（留空则由 AI 从已上传图片中识别）",
        value="",
        placeholder="例如：happy、calm、angry...",
        key="manual_emotion_label",
    )

    confirm_btn = st.button(
        "🎭 确认添加表情组",
        type="primary",
        disabled=filled_count == 0,
        use_container_width=True,
        help="至少分配一个槽位后才能确认",
    )

    if confirm_btn and filled_count > 0:
        name_override = manual_label.strip().lower() or None
        _confirm_emotion_group(name_override)


def _classify_single_image(image):
    """调用 AI 对单张图片分类，将结果存入 session state。"""
    vision_backend = st.session_state.get("vision_backend", "qwen")

    if vision_backend == "qwen":
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        key_name = "DASHSCOPE_API_KEY"
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        key_name = "OPENAI_API_KEY"

    if not api_key:
        st.error(f"⚠️ 未配置 {key_name}。请在项目根目录的 `.env` 文件中写入 `{key_name}=sk-xxx`。")
        return

    slot_images = st.session_state.pending_slot_images
    already_assigned = [s for s, v in slot_images.items() if v is not None]

    with st.spinner("🤖 AI 正在识别图片..."):
        try:
            from vtuber_engine.audio.image_recognizer import ImageEmotionRecognizer
            recognizer = ImageEmotionRecognizer(
                backend=vision_backend,
                model=st.session_state.vision_model or None,
            )
            result = recognizer.classify_single(
                image=image,
                existing_slots=already_assigned if already_assigned else None,
            )
            st.session_state.pending_single_result = result
        except Exception as e:
            st.error(f"AI 识别失败: {e}")


def _confirm_emotion_group(name_override: str | None = None):
    """当用户点击「确认添加」时：如无情绪名则用 AI 识别，然后存入。"""
    config = st.session_state.config
    slot_images = st.session_state.pending_slot_images
    classified = {k: v for k, v in slot_images.items() if v is not None}

    if not classified:
        st.warning("请至少分配一个槽位的图片。")
        return

    if name_override:
        label = name_override
        _store_emotion_group_single(label, classified)
    else:
        # 用 AI 从已分配图片中识别情绪名
        vision_backend = st.session_state.get("vision_backend", "qwen")
        with st.spinner("🤖 AI 识别情绪名..."):
            try:
                from vtuber_engine.audio.image_recognizer import ImageEmotionRecognizer
                recognizer = ImageEmotionRecognizer(
                    backend=vision_backend,
                    model=st.session_state.vision_model or None,
                )
                ai_emotion, _ = recognizer.classify_and_recognize(
                    images_list=list(classified.values()),
                    existing_emotions=config.emotions,
                )
                label = ai_emotion
                st.success(f"🎭 AI 识别情绪: **{label}**")
            except Exception as e:
                st.error(f"AI 情绪识别失败: {e}")
                return
        _store_emotion_group_single(label, classified)


def _store_emotion_group_single(label: str, classified: dict):
    """将已分类的图片存入 assets 并注册表情，然后清空 pending 状态。"""
    config = st.session_state.config
    assets = st.session_state.assets

    if label in config.emotions:
        st.warning(f"⚠️ 表情「{label}」已存在，将覆盖旧素材。")

    assets.put_emotion_group(label, classified)
    config.add_emotion(label)

    # 清空所有 pending 状态
    st.session_state.pending_slot_images = {
        "eo_mo": None, "eo_mc": None, "ec_mo": None, "ec_mc": None
    }
    st.session_state.pending_single_image = None
    st.session_state.pending_single_file_id = None
    st.session_state.pending_single_result = None
    st.session_state.single_upload_round += 1
    st.session_state.upload_round += 1

    st.success(f"✅ 表情组「{label}」已添加！共 {len(config.emotions)} 组表情。")
    st.rerun()


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

    if not config.emotions:
        checks.append("❌ 还没有上传任何表情组")
        ready = False
    else:
        missing = assets.missing_keys(config)
        if missing:
            checks.append(f"❌ 还差 {len(missing)} 张素材未上传")
            ready = False
        else:
            checks.append(f"✅ 所有素材已就绪（{len(config.emotions)} 组表情）")

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

        emotion_engine = EmotionEngine(
            backend=emotion_backend,
            available_emotions=config.emotions,
            model=st.session_state.text_model or None,
        )
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


# 各后端预设模型列表
_TEXT_MODEL_PRESETS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "o4-mini"],
    "qwen": ["qwen-plus", "qwen3.5-plus", "qwen-max", "qwen-turbo", "qwen3-235b-a22b"],
    "rule": [],
}

_VISION_MODEL_PRESETS: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "qwen": [
        "qwen-vl-max",
        "qwen-vl-plus",
        "qvq-max",
        "qwen3.5-plus",
    ],
}


def _tab_model_config(emotion_backend: str, vision_backend: str):
    """模型配置页：分别设置文字模型和视觉模型的具体名称。"""
    st.markdown(
        """
        ### 🧠 模型配置
        分别配置 **情绪分析（文字模型）** 和 **表情识别（视觉模型）**。
        后端在左侧栏切换，这里只需要填写模型字符串。
        """
    )

    # ── 文字模型（情绪分析） ──
    st.subheader("💬 情绪分析 — 文字模型")
    st.caption(
        f"当前后端: `{emotion_backend}`"
        + (" · rule 模式不需要模型" if emotion_backend == "rule" else "")
    )

    if emotion_backend != "rule":
        text_presets = _TEXT_MODEL_PRESETS.get(emotion_backend, [])
        if text_presets:
            st.markdown("📄 快捷选择")
            preset_cols = st.columns(len(text_presets))
            for col, preset in zip(preset_cols, text_presets):
                with col:
                    active = preset == st.session_state.text_model
                    if st.button(
                        f"✔️ {preset}" if active else preset,
                        key=f"text_preset_{preset}",
                        type="primary" if active else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state.text_model = preset
                        st.rerun()

        new_text = st.text_input(
            "模型名称（可直接输入任意字符串）",
            value=st.session_state.text_model,
            key="text_model_input",
            placeholder="例如: gpt-4o-mini、qwen3.5-plus",
        )
        if new_text.strip() and new_text.strip() != st.session_state.text_model:
            st.session_state.text_model = new_text.strip()
        st.info(f"当前情绪分析模型： **{st.session_state.text_model}**")
    else:
        st.info("🟢 rule 后端不调用任何 API，无需配置模型。")

    st.divider()

    # ── 视觉模型（表情识别） ──
    st.subheader("🖼️ 表情识别 — 视觉模型")
    st.caption(f"当前后端: `{vision_backend}`")

    vision_presets = _VISION_MODEL_PRESETS.get(vision_backend, [])
    if vision_presets:
        st.markdown("📄 快捷选择")
        preset_cols = st.columns(len(vision_presets))
        for col, preset in zip(preset_cols, vision_presets):
            with col:
                active = preset == st.session_state.vision_model
                if st.button(
                    f"✔️ {preset}" if active else preset,
                    key=f"vision_preset_{preset}",
                    type="primary" if active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.vision_model = preset
                    st.rerun()

    new_vision = st.text_input(
        "模型名称（可直接输入任意字符串）",
        value=st.session_state.vision_model,
        key="vision_model_input",
        placeholder="例如: gpt-4o、qwen-vl-max、qwen3.5-plus",
    )
    if new_vision.strip() and new_vision.strip() != st.session_state.vision_model:
        st.session_state.vision_model = new_vision.strip()
    st.info(f"当前表情识别模型： **{st.session_state.vision_model}**")

    st.divider()

    # ── 配置摘要 ──
    st.subheader("📋 当前配置摘要")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("💬 **情绪分析**")
        st.code(
            f"backend : {emotion_backend}\n"
            + (
                f"model   : {st.session_state.text_model}"
                if emotion_backend != "rule"
                else "model   : (rule-based)"
            ),
            language="yaml",
        )
    with col2:
        st.markdown("🖼️ **表情识别**")
        st.code(
            f"backend : {vision_backend}\nmodel   : {st.session_state.vision_model}",
            language="yaml",
        )


# ──────────────────────────────────────────────
# 主界面
# ──────────────────────────────────────────────


def main():
    st.title("🎭 VTuber Engine")
    st.caption("AI 驱动 · 角色动画生成 · 绿幕视频输出")

    fps, smoothing, emotion_backend, vision_backend = _sidebar_config()
    st.session_state.vision_backend = vision_backend

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📁 上传素材",
            "🎵 音频输入",
            "🎞️ 生成视频",
            "📺 预览 & 下载",
            "🧠 模型配置",
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

    with tab5:
        _tab_model_config(emotion_backend, vision_backend)


if __name__ == "__main__":
    main()
