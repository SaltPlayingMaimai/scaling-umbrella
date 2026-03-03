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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    # 情绪分析分段时长（秒）
    if "segment_seconds" not in st.session_state:
        st.session_state.segment_seconds = 12.5
    # 强制切换表情秒数（0 = 不启用）
    if "force_switch_seconds" not in st.session_state:
        st.session_state.force_switch_seconds = 12.0
    # 批量上传模式（主流程）：4 张图同时上传、并行 AI 识别
    if "pending_batch_images" not in st.session_state:
        st.session_state.pending_batch_images = [None, None, None, None]
    if "pending_batch_file_ids" not in st.session_state:
        st.session_state.pending_batch_file_ids = [None, None, None, None]
    if "pending_batch_filenames" not in st.session_state:
        st.session_state.pending_batch_filenames = [None, None, None, None]
    if "pending_batch_results" not in st.session_state:
        st.session_state.pending_batch_results = [None, None, None, None]
    # pending_img_slots[i] = 分配给第 i 张图的 slot 名称，或 None
    if "pending_img_slots" not in st.session_state:
        st.session_state.pending_img_slots = [None, None, None, None]
    # 用于重置文件上传框
    if "batch_upload_round" not in st.session_state:
        st.session_state.batch_upload_round = 0
    # upload_round 保持向下兼容
    if "upload_round" not in st.session_state:
        st.session_state.upload_round = 0
    # 最近一次自动注册的表情名（用于显示成功消息）
    if "_last_registered_emotion" not in st.session_state:
        st.session_state["_last_registered_emotion"] = None
    # 调试数据：emotion -> [ai_result_for_slot0..3]（按 eo_mo/eo_mc/ec_mo/ec_mc 顺序）
    if "emotion_debug_data" not in st.session_state:
        st.session_state.emotion_debug_data = {}
    # AI 模型选择
    if "text_model" not in st.session_state:
        st.session_state.text_model = "qwen-plus"  # 情绪分析文字模型
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

    st.sidebar.subheader("🧠 情绪分析设置")
    segment_seconds = st.sidebar.slider(
        "分析片段时长（秒）",
        min_value=0.1,
        max_value=30.0,
        value=st.session_state.segment_seconds,
        step=0.1,
        format="%.1f",
        help="每隔多少秒调用一次 AI 进行情绪分析。越短越精细但消耗更多 token；默认 1 秒。",
    )
    st.session_state.segment_seconds = segment_seconds

    force_switch_seconds = st.sidebar.slider(
        "强制切换表情（秒）",
        min_value=0.0,
        max_value=30.0,
        value=st.session_state.force_switch_seconds,
        step=0.5,
        format="%.1f",
        help="同一表情持续超过此秒数后，在下一个句首强制切换到不同表情。0 = 不启用。",
    )
    st.session_state.force_switch_seconds = force_switch_seconds

    gesture_min_hold = st.sidebar.slider(
        "动作最小停留（秒）",
        min_value=0.1,
        max_value=10.0,
        value=float(st.session_state.get("gesture_min_hold", 5.0)),
        step=0.1,
        format="%.1f",
        help="动作切换后至少停留多少秒再允许切换下一个。防止动作快速跳变。默认 5秒。",
    )
    st.session_state["gesture_min_hold"] = gesture_min_hold

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
    _TEXT_DEFAULTS = {"qwen": "qwen-plus", "openai": "gpt-4o-mini", "rule": ""}
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

    return (
        fps,
        smoothing,
        emotion_backend,
        vision_backend,
        segment_seconds,
        force_switch_seconds,
        gesture_min_hold,
    )


# ──────────────────────────────────────────────
# Tab 1：上传素材
# ──────────────────────────────────────────────


def _tab_upload_assets():
    """素材上传界面 — 批量上传（最多4张）+ AI 并行识别 + 手动拖拽调整分配。"""
    config = st.session_state.config
    assets = st.session_state.assets

    st.markdown(
        """
        ### 📌 上传规则
        一次上传最多 **4 张**差分图，点击「AI 并行识别全部」后可看到每张图的各状态概率及情绪建议。
        你可以通过下拉框调整每张图的分配槽位，然后确认添加。

        > 💡 差分图通常包含：眼开嘴开、眼开嘴闭、眼闭嘴开、眼闭嘴闭 四种组合。
        > 图片只存在内存中，关闭页面即消失，不会上传到任何地方。
        """
    )

    # ─────────────────────────────────────────
    # 已注册表情组展示（含 AI 调试数据）
    # ─────────────────────────────────────────
    _SLOT_ORDER = ["eo_mo", "eo_mc", "ec_mo", "ec_mc"]
    _SLOT_CN = {
        "eo_mo": "眼开+嘴开",
        "eo_mc": "眼开+嘴闭",
        "ec_mo": "眼闭+嘴开",
        "ec_mc": "眼闭+嘴闭",
    }

    if config.emotions:
        st.subheader("✅ 已注册的表情组")
        for emotion in config.emotions:
            with st.expander(f"🎭 {emotion}", expanded=False):
                # 图片显示
                img_cols = st.columns(4)
                for col, slot in zip(img_cols, _SLOT_ORDER):
                    key = f"{emotion}_{slot}"
                    with col:
                        if assets.has(key):
                            st.image(assets.get(key), width=120, caption=_SLOT_CN[slot])
                        else:
                            st.caption(f"⬜ {_SLOT_CN[slot]}")

                # AI 调试数据
                debug_list = st.session_state.emotion_debug_data.get(emotion)
                # 情绪向量摘要
                ev = config.emotion_vectors.get(emotion)
                if ev:
                    ev_str = " | ".join(
                        f"{k}:{v*100:.0f}%" for k, v in ev.items() if v > 0.01
                    )
                    st.caption(f"🎯 情绪向量: {ev_str}")
                with st.expander("🔍 AI 识别调试数据", expanded=False):
                    if debug_list:
                        dcols = st.columns(4)
                        for col, slot, ai_res in zip(dcols, _SLOT_ORDER, debug_list):
                            with col:
                                st.caption(f"**{_SLOT_CN[slot]}**")
                                if ai_res is not None:
                                    probs = ai_res.get("probabilities", {})
                                    label_s = ai_res.get("label", "?")
                                    eyebrow_s = ai_res.get("eyebrow_analysis", "")
                                    eye_s = ai_res.get("eye_analysis", "")
                                    mouth_s = ai_res.get("mouth_analysis", "")
                                    ev_s = ai_res.get("emotion_vector", {})
                                    ai_slot = ai_res.get("assigned_slot", "?")
                                    st.caption(f"标签: `{label_s}`")
                                    st.caption(f"AI判断: `{ai_slot}`")
                                    if eyebrow_s:
                                        st.caption(f"🤨 {eyebrow_s[:100]}")
                                    if eye_s:
                                        st.caption(f"👁️ {eye_s[:100]}")
                                    if mouth_s:
                                        st.caption(f"👄 {mouth_s[:100]}")
                                    if ev_s:
                                        ev_display = " ".join(
                                            f"{k}:{v*100:.0f}%"
                                            for k, v in ev_s.items()
                                            if v > 0.01
                                        )
                                        st.caption(f"📊 {ev_display}")
                                    for s in _SLOT_ORDER:
                                        p = probs.get(s, 0.0)
                                        filled = int(p * 10)
                                        bar = "█" * filled + "░" * (10 - filled)
                                        mark = " ◀" if s == ai_slot else ""
                                        st.caption(f"`{s}` {p*100:4.1f}% {bar}{mark}")
                                else:
                                    st.caption("_(无数据)_")
                    else:
                        st.caption("_(该表情组无 AI 识别数据)_")

                # ── 编辑区：改名 + 槽位调整
                with st.expander("✏️ 编辑表情组", expanded=False):
                    # 改名
                    st.caption("**重命名**")
                    rcol1, rcol2 = st.columns([3, 1])
                    with rcol1:
                        new_name_input = st.text_input(
                            "新名称",
                            value=emotion,
                            key=f"rename_input_{emotion}",
                            label_visibility="collapsed",
                        )
                    with rcol2:
                        if st.button("保存", key=f"save_rename_{emotion}"):
                            _rename_emotion(emotion, new_name_input.strip().lower())
                            st.rerun()

                    st.divider()
                    # 槽位调整（同组图片之间互换）
                    st.caption("**调整槽位分配**（同组图片之间互换）")
                    current_slot_imgs = {
                        slot: assets.get(f"{emotion}_{slot}")
                        for slot in _SLOT_ORDER
                        if assets.has(f"{emotion}_{slot}")
                    }
                    if current_slot_imgs:
                        img_list = list(current_slot_imgs.items())
                        ecols = st.columns(len(img_list))
                        new_slot_map: dict = {}
                        avail_slots = list(current_slot_imgs.keys())
                        for (orig_s, simg), ecol in zip(img_list, ecols):
                            with ecol:
                                st.image(simg, width=100, caption=_SLOT_CN[orig_s])
                                new_s = st.selectbox(
                                    "→ 槽位",
                                    avail_slots,
                                    index=avail_slots.index(orig_s),
                                    key=f"edit_slot_{emotion}_{orig_s}",
                                    format_func=lambda s: _SLOT_CN.get(s, s),
                                )
                                new_slot_map[orig_s] = new_s
                        if st.button("应用槽位调整", key=f"apply_slots_{emotion}"):
                            _reassign_slots(emotion, current_slot_imgs, new_slot_map)
                            st.rerun()
                    else:
                        st.caption("_(该表情组暂无图片)_")

                # 操作按钮行
                btn_cols = st.columns([1, 1, 2])
                with btn_cols[0]:
                    if st.button(f"🔄 重新分析", key=f"reanalyze_{emotion}"):
                        _reanalyze_emotion_group(emotion)
                        st.rerun()
                with btn_cols[1]:
                    if st.button(f"🗑️ 删除", key=f"del_{emotion}"):
                        config.remove_emotion(emotion)
                        assets.remove_emotion_group(emotion)
                        if emotion in st.session_state.emotion_debug_data:
                            del st.session_state.emotion_debug_data[emotion]
                        st.rerun()

    st.divider()

    # ─────────────────────────────────────────
    # 步骤 1：上传最多 4 张图
    # ─────────────────────────────────────────
    st.subheader("➕ 添加新表情组")
    st.markdown("**📤 步骤 1 — 上传差分图（1-4 张）**")

    # ── 显示最近一次自动注册的提示
    if st.session_state.get("_last_registered_emotion"):
        st.success(
            f"✅ 已自动注册表情组「{st.session_state['_last_registered_emotion']}」！"
        )
        st.session_state["_last_registered_emotion"] = None

    batch_round = st.session_state.batch_upload_round

    # 单个多文件上传器，一次最多选 4 张，减少渲染次数
    uploaded_files = st.file_uploader(
        "选择 1-4 张差分图（可一次性选择多张）",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key=f"batch_multi_{batch_round}",
    )

    # 检测是否有新文件上传
    if uploaded_files:
        uploaded_files = uploaded_files[:4]  # 截断至4张
        new_ids = [f.file_id for f in uploaded_files]
        old_ids = [
            st.session_state.pending_batch_file_ids[i]
            for i in range(len(uploaded_files))
        ]
        if new_ids != old_ids[: len(new_ids)]:
            # 有新文件，重置 pending 状态
            st.session_state.pending_batch_images = [None, None, None, None]
            st.session_state.pending_batch_file_ids = [None, None, None, None]
            st.session_state.pending_batch_results = [None, None, None, None]
            st.session_state.pending_img_slots = [None, None, None, None]
            for idx, f in enumerate(uploaded_files):
                st.session_state.pending_batch_images[idx] = Image.open(f).convert(
                    "RGBA"
                )
                st.session_state.pending_batch_file_ids[idx] = f.file_id
                st.session_state.pending_batch_filenames[idx] = f.name

    # 4 列一页布局：显示预览、文件名、删除按钮
    st.markdown("**📄 已上传的文件**（点击「✕ 删除」移除）")
    preview_cols = st.columns(4)
    for idx in range(4):
        with preview_cols[idx]:
            img = st.session_state.pending_batch_images[idx]
            filename = st.session_state.pending_batch_filenames[idx]
            if img is not None and filename is not None:
                st.image(img, width=120)
                st.caption(f"📄 {filename[:20]}")  # 截断长文件名
                if st.button(f"✕ 删除", key=f"del_img_{idx}", use_container_width=True):
                    st.session_state.pending_batch_images[idx] = None
                    st.session_state.pending_batch_file_ids[idx] = None
                    st.session_state.pending_batch_filenames[idx] = None
                    st.session_state.pending_batch_results[idx] = None
                    st.session_state.pending_img_slots[idx] = None
                    st.rerun()
            else:
                st.caption("⬜ 未上传")

    uploaded_count = sum(
        1 for x in st.session_state.pending_batch_images if x is not None
    )
    if uploaded_count == 0:
        st.info("请至少上传 1 张图片。")
        return

    # AI 识别并自动注册
    st.markdown("**🤖 AI 识别并自动注册**")
    st.caption(
        "AI 自动识别每张图的眼/嘴状态后立即注册。"
        "重名时自动追加后缀（如 `happy_2`）。"
        "注册后可在上方「已注册的表情组」中修改名称或调整槽位。"
    )

    if st.button("🤖 AI 识别并自动注册", type="primary", use_container_width=True):
        _batch_classify_all(batch_round)
        _auto_register_from_pending()
        st.rerun()


# ─────────────────────────────────────────────────────
# Batch 辅助函数
# ─────────────────────────────────────────────────────

_SLOT_ORDER_GLOBAL = ["eo_mo", "eo_mc", "ec_mo", "ec_mc"]


def _batch_classify_all(batch_round: int):
    """对所有已上传的图片并行调用 AI 分类，并自动贪心分配槽位。"""
    vision_backend = st.session_state.get("vision_backend", "qwen")

    if vision_backend == "qwen":
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        key_name = "DASHSCOPE_API_KEY"
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        key_name = "OPENAI_API_KEY"

    if not api_key:
        st.error(
            f"⚠️ 未配置 {key_name}。请在项目根目录的 `.env` 文件中写入 `{key_name}=sk-xxx`。"
        )
        return

    images = st.session_state.pending_batch_images

    with st.spinner("🤖 AI 并行识别中，请稍候..."):
        try:
            from vtuber_engine.audio.image_recognizer import ImageEmotionRecognizer

            recognizer = ImageEmotionRecognizer(
                backend=vision_backend,
                model=st.session_state.vision_model or None,
            )
            results = recognizer.classify_batch_parallel(images)
            st.session_state.pending_batch_results = results
        except Exception as e:
            st.error(f"AI 识别失败: {e}")
            return

    # 贪心自动分配：对所有 (概率, img_idx, slot) 三元组降序排列，无冲突分配
    candidates = []
    for i, result in enumerate(results):
        if result is None or images[i] is None:
            continue
        for slot, p in result["probabilities"].items():
            candidates.append((p, i, slot))
    candidates.sort(reverse=True)

    assigned_imgs: set[int] = set()
    assigned_slots: set[str] = set()
    auto_slots = [None] * 4

    for p, img_idx, slot in candidates:
        if img_idx in assigned_imgs or slot in assigned_slots:
            continue
        if p >= 0.15:  # 最低置信度阈值
            auto_slots[img_idx] = slot
            assigned_imgs.add(img_idx)
            assigned_slots.add(slot)

    st.session_state.pending_img_slots = auto_slots

    print(f"[BatchClassify] auto_slots={auto_slots}")


def _store_emotion_group_batch(label: str, classified: dict, debug_list: list):
    """将分类好的图片存入 assets，注册表情，并清空 pending 状态。"""
    config = st.session_state.config
    assets = st.session_state.assets

    if label in config.emotions:
        st.warning(f"⚠️ 表情「{label}」已存在，将覆盖旧素材。")

    assets.put_emotion_group(label, classified)

    # 从 AI 调试数据中提取情绪向量（优先级 eo_mo > eo_mc > ec_mo > ec_mc）
    _SLOT_ORDER_LOCAL = ["eo_mo", "eo_mc", "ec_mo", "ec_mc"]
    emotion_vector = None
    for slot_idx, slot in enumerate(_SLOT_ORDER_LOCAL):
        if (
            debug_list
            and slot_idx < len(debug_list)
            and debug_list[slot_idx] is not None
        ):
            ev = debug_list[slot_idx].get("emotion_vector")
            if ev:
                emotion_vector = ev
                break

    config.add_emotion(label, emotion_vector=emotion_vector)
    st.session_state.emotion_debug_data[label] = debug_list

    # 清空所有 pending 状态
    st.session_state.pending_batch_images = [None, None, None, None]
    st.session_state.pending_batch_file_ids = [None, None, None, None]
    st.session_state.pending_batch_filenames = [None, None, None, None]
    st.session_state.pending_batch_results = [None, None, None, None]
    st.session_state.pending_img_slots = [None, None, None, None]
    st.session_state.batch_upload_round += 1
    st.session_state.upload_round += 1

    st.success(f"✅ 表情组「{label}」已添加！共 {len(config.emotions)} 组表情。")
    st.rerun()


def _reanalyze_emotion_group(emotion: str):
    """重新对已注册表情组的图片运行 AI 分析，更新情绪向量和调试数据。"""
    config = st.session_state.config
    assets = st.session_state.assets
    vision_backend = st.session_state.get("vision_backend", "qwen")

    _SLOT_ORDER_LOCAL = ["eo_mo", "eo_mc", "ec_mo", "ec_mc"]

    # 收集该表情组的所有图片
    images = []
    for slot in _SLOT_ORDER_LOCAL:
        key = f"{emotion}_{slot}"
        if assets.has(key):
            images.append(assets.get(key))
        else:
            images.append(None)

    valid_images = [img for img in images if img is not None]
    if not valid_images:
        st.warning(f"表情组「{emotion}」没有可分析的图片。")
        return

    try:
        from vtuber_engine.audio.image_recognizer import ImageEmotionRecognizer

        recognizer = ImageEmotionRecognizer(
            backend=vision_backend,
            model=st.session_state.get("vision_model") or None,
        )

        # 对每张有效图片逐张分析
        new_debug_list = []
        for img in images:
            if img is not None:
                result = recognizer.classify_single(img)
                new_debug_list.append(result)
            else:
                new_debug_list.append(None)

        # 更新情绪向量（优先级 eo_mo > eo_mc > ec_mo > ec_mc）
        emotion_vector = None
        for slot_idx, slot in enumerate(_SLOT_ORDER_LOCAL):
            if (
                new_debug_list
                and slot_idx < len(new_debug_list)
                and new_debug_list[slot_idx] is not None
            ):
                ev = new_debug_list[slot_idx].get("emotion_vector")
                if ev:
                    emotion_vector = ev
                    break

        # 更新配置和调试数据
        if emotion_vector:
            config.add_emotion(emotion, emotion_vector=emotion_vector)
        st.session_state.emotion_debug_data[emotion] = new_debug_list
        st.success(f"✅ 表情组「{emotion}」重新分析完成！")

    except Exception as e:
        st.error(f"重新分析失败: {e}")


def _auto_register_from_pending():
    """AI 分析后自动注册表情组；重名时自动追加 _2、_3… 后缀。"""
    config = st.session_state.config
    pending_results = st.session_state.pending_batch_results
    pending_images = st.session_state.pending_batch_images
    pending_slots = st.session_state.pending_img_slots

    _SLOT_ORDER_LOCAL = ["eo_mo", "eo_mc", "ec_mo", "ec_mc"]

    # 构建最终槽位映射
    final_slot_map: dict = {}
    for idx in range(4):
        img = pending_images[idx]
        if img is None:
            continue
        slot = pending_slots[idx]
        if slot is None:
            continue
        final_slot_map[slot] = (idx, img)

    if not final_slot_map:
        return  # 没有任何可分配图片，直接跳过

    # 推断情绪名（和旧逻辑保持一致）
    suggested_emotion = "unknown"
    suggested_ev: dict = {}
    for priority_slot in _SLOT_ORDER_LOCAL:
        if priority_slot in final_slot_map:
            img_idx, _ = final_slot_map[priority_slot]
            r = pending_results[img_idx]
            if r:
                suggested_emotion = r.get("label") or "unknown"
                suggested_ev = r.get("emotion_vector", {})
            break

    # 重名处理：xxx → xxx_2 → xxx_3 …
    label = suggested_emotion
    if label in config.emotions:
        counter = 2
        while f"{label}_{counter}" in config.emotions:
            counter += 1
        label = f"{label}_{counter}"

    classified = {slot: img for slot, (_, img) in final_slot_map.items()}
    debug_list = []
    for slot in _SLOT_ORDER_LOCAL:
        if slot in final_slot_map:
            img_idx, _ = final_slot_map[slot]
            debug_list.append(pending_results[img_idx])
        else:
            debug_list.append(None)

    # 静默注册（不调用 st.rerun，由调用方负责）
    assets = st.session_state.assets
    assets.put_emotion_group(label, classified)
    emotion_vector = suggested_ev or None
    # 从 debug_list 取第一个非空 emotion_vector（更准确）
    for slot_idx, slot in enumerate(_SLOT_ORDER_LOCAL):
        if (
            debug_list
            and slot_idx < len(debug_list)
            and debug_list[slot_idx] is not None
        ):
            ev = debug_list[slot_idx].get("emotion_vector")
            if ev:
                emotion_vector = ev
                break
    config.add_emotion(label, emotion_vector=emotion_vector)
    st.session_state.emotion_debug_data[label] = debug_list

    # 清空 pending 状态
    st.session_state.pending_batch_images = [None, None, None, None]
    st.session_state.pending_batch_file_ids = [None, None, None, None]
    st.session_state.pending_batch_filenames = [None, None, None, None]
    st.session_state.pending_batch_results = [None, None, None, None]
    st.session_state.pending_img_slots = [None, None, None, None]
    st.session_state.batch_upload_round += 1
    st.session_state.upload_round += 1
    st.session_state["_last_registered_emotion"] = label


def _rename_emotion(old_name: str, new_name: str) -> str:
    """重命名已注册表情组；重名时自动加 _2 后缀。返回最终名称。"""
    if not new_name or new_name == old_name:
        return old_name
    config = st.session_state.config
    assets = st.session_state.assets
    _SLOT_ORDER_LOCAL = ["eo_mo", "eo_mc", "ec_mo", "ec_mc"]

    # 处理目标名重复
    final_name = new_name
    if final_name in config.emotions and final_name != old_name:
        counter = 2
        while f"{final_name}_{counter}" in config.emotions:
            counter += 1
        final_name = f"{final_name}_{counter}"

    # 迁移图片
    slot_imgs = {
        slot: assets.get(f"{old_name}_{slot}")
        for slot in _SLOT_ORDER_LOCAL
        if assets.has(f"{old_name}_{slot}")
    }
    assets.remove_emotion_group(old_name)
    assets.put_emotion_group(final_name, slot_imgs)

    # 更新 config
    if old_name in config.emotions:
        idx = config.emotions.index(old_name)
        config.emotions[idx] = final_name
    if old_name in config.emotion_vectors:
        config.emotion_vectors[final_name] = config.emotion_vectors.pop(old_name)

    # 迁移调试数据
    if old_name in st.session_state.emotion_debug_data:
        st.session_state.emotion_debug_data[final_name] = (
            st.session_state.emotion_debug_data.pop(old_name)
        )
    return final_name


def _reassign_slots(emotion: str, slot_images: dict, new_assignments: dict):
    """将已注册表情组的各图片重新分配到新的槽位（就地修改）。"""
    assets = st.session_state.assets
    # 构建 new_slot -> image 映射（同一 new_slot 有多张时保留最后一张）
    new_slot_to_img: dict = {}
    for orig_slot, new_slot in new_assignments.items():
        img = slot_images[orig_slot]
        new_slot_to_img[new_slot] = img
    # 先清空再写入
    assets.remove_emotion_group(emotion)
    assets.put_emotion_group(emotion, new_slot_to_img)


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


def _tab_generate(
    fps: int,
    smoothing: float,
    emotion_backend: str,
    segment_seconds: float,
    force_switch_seconds: float = 0.0,
    gesture_min_hold: float = 1.5,
):
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

    # 显示情绪分析片段配置
    st.caption(
        f"情绪分析：每 **{segment_seconds:.1f} 秒** 一个片段 | "
        f"后端：{emotion_backend} | 帧率：{fps} fps"
    )

    st.divider()

    if not ready:
        st.info("请先完成上面的步骤。")
        return

    if st.button("🎬 开始生成视频", type="primary", use_container_width=True):
        _run_pipeline(
            fps,
            smoothing,
            emotion_backend,
            segment_seconds,
            force_switch_seconds,
            gesture_min_hold,
        )


def _run_pipeline(
    fps: int,
    smoothing: float,
    emotion_backend: str,
    segment_seconds: float = 1.0,
    force_switch_seconds: float = 0.0,
    gesture_min_hold: float = 1.5,
):
    """执行完整生成管线（已优化：并行渲染 + 硬件编码 + 精细进度）。"""
    config = st.session_state.config
    assets = st.session_state.assets
    audio_bytes: bytes = st.session_state.audio_bytes
    audio_suffix: str = st.session_state.audio_suffix

    progress = st.progress(0, text="准备中...")
    perf_info = st.empty()  # 性能信息占位

    import time as _time

    t_start = _time.perf_counter()
    timings: dict = {}

    try:
        # Step 1: 音频分析
        progress.progress(5, text="🔍 分析音频特征...")
        t0 = _time.perf_counter()
        from vtuber_engine.audio.analyzer import AudioAnalyzer

        analyzer = AudioAnalyzer(fps=fps)
        audio_features = analyzer.analyze(audio_bytes)
        st.session_state.audio_features = audio_features
        timings["音频分析"] = _time.perf_counter() - t0

        # Step 2: 情绪分析
        progress.progress(15, text="🧠 AI 情绪识别...")
        t0 = _time.perf_counter()
        from vtuber_engine.audio.emotion_engine import EmotionEngine

        emotion_engine = EmotionEngine(
            backend=emotion_backend,
            available_emotions=config.emotions,
            model=st.session_state.text_model or None,
        )
        emotion_vectors = emotion_engine.analyze(
            audio_features,
            segment_seconds=segment_seconds,
        )
        st.session_state.emotion_vectors = emotion_vectors
        timings["情绪分析"] = _time.perf_counter() - t0

        # Step 3: 状态计算
        progress.progress(30, text="🎯 计算角色状态...")
        t0 = _time.perf_counter()
        from vtuber_engine.core.state_engine import StateEngine

        state_engine = StateEngine(
            config,
            fps=fps,
            force_switch_seconds=force_switch_seconds,
            gesture_min_hold_seconds=gesture_min_hold,
        )
        states = state_engine.process(audio_features, emotion_vectors)
        timings["状态计算"] = _time.perf_counter() - t0

        # Step 4: 动画平滑
        progress.progress(40, text="✨ 动画插值平滑...")
        t0 = _time.perf_counter()
        from vtuber_engine.core.animation_engine import AnimationEngine

        anim_engine = AnimationEngine(smoothing=smoothing)
        animated_states = anim_engine.process(states)
        timings["动画平滑"] = _time.perf_counter() - t0

        # Step 5: 渲染帧（利用优化后的 render_sequence：去重 + 并行）
        progress.progress(50, text="🖼️ 渲染帧图像...")
        t0 = _time.perf_counter()
        from vtuber_engine.render.renderer import Renderer

        renderer = Renderer(config, assets)
        total_frames = len(animated_states)

        def _render_progress(current, total):
            pct = 50 + int((current / max(total, 1)) * 25)
            progress.progress(min(pct, 75), text=f"🖼️ 渲染帧 {current}/{total}...")

        frames = renderer.render_sequence(
            animated_states, progress_callback=_render_progress
        )
        timings["帧渲染"] = _time.perf_counter() - t0

        # Step 6: 导出视频（硬件编码 + 缓冲写入）
        progress.progress(78, text="🎬 导出视频...")
        t0 = _time.perf_counter()
        from vtuber_engine.export.video_exporter import VideoExporter

        output_dir = tempfile.mkdtemp(prefix="vtuber_output_")
        exporter = VideoExporter(fps=fps, output_dir=output_dir)
        encoder_info = exporter.get_encoder_info()

        def _export_progress(current, total):
            pct = 78 + int((current / max(total, 1)) * 18)
            progress.progress(
                min(pct, 96), text=f"🎬 编码帧 {current}/{total} [{encoder_info}]..."
            )

        output_path = exporter.export(
            frames=frames,
            audio_bytes=audio_bytes,
            audio_suffix=audio_suffix,
            output_filename="output.mp4",
            progress_callback=_export_progress,
        )
        timings["视频导出"] = _time.perf_counter() - t0

        # 读取视频到内存
        with open(output_path, "rb") as f:
            st.session_state.video_bytes = f.read()

        total_time = _time.perf_counter() - t_start
        progress.progress(100, text="✅ 完成！")

        # 显示结果 + 性能信息
        st.balloons()
        st.success(
            f"🎉 视频生成完成！ 共 {total_frames} 帧，"
            f"时长 {audio_features.duration:.1f} 秒，"
            f"总耗时 {total_time:.1f} 秒"
        )

        # 性能分析面板
        with st.expander("⏱️ 性能详情", expanded=False):
            cols = st.columns(len(timings))
            for col, (step_name, step_time) in zip(cols, timings.items()):
                with col:
                    st.metric(step_name, f"{step_time:.1f}s")
            st.caption(
                f"编码器: {encoder_info} | "
                f"帧率: {fps} fps | "
                f"分辨率: {config.resolution[0]}×{config.resolution[1]} | "
                f"独立帧: {len(renderer._frame_cache)}/{total_frames}"
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
        _mpl_line(list(features.volume), height=150)

    # 基频
    if features.pitch:
        st.markdown("**基频 (Hz)**")
        _mpl_line(list(features.pitch), height=150, color="#ff7f0e")

    # 说话段
    if features.is_speaking:
        speaking_float = [1.0 if s else 0.0 for s in features.is_speaking]
        st.markdown("**语音活动 (VAD)**")
        _mpl_bar(speaking_float, height=100, color="#2ca02c")

    # 情绪分析
    if emotions:
        st.subheader("🧠 情绪分析结果")

        # 取中间帧的情绪作为代表
        mid = len(emotions) // 2
        sample = emotions[mid]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**情绪权重（中间时刻）**")
            from vtuber_engine.models.data_models import EMOTION_KEYS

            ev_values = [getattr(sample, k, 0.0) for k in EMOTION_KEYS]
            _mpl_hbar(EMOTION_KEYS, ev_values, height=250)

        with col2:
            st.markdown("**主要情绪**")
            dominant = sample.dominant_emotion()
            st.metric("主情绪", dominant)
            st.metric("能量", f"{sample.energy:.2f}")

        # 能量曲线
        if len(emotions) > 1:
            energy_curve = [e.energy for e in emotions]
            st.markdown("**能量曲线（逐帧）**")
            _mpl_line(energy_curve, height=150, color="#9467bd")


# ─────────────────────────────────────────────────────
# Matplotlib 图表辅助（规避 pyarrow DLL 问题）
# ─────────────────────────────────────────────────────


def _mpl_line(data: list, height: int = 150, color: str = "#1f77b4"):
    """用 matplotlib 画折线图，替代 st.line_chart（避免 pyarrow）。"""
    fig, ax = plt.subplots(figsize=(8, max(1.2, height / 100)))
    ax.plot(data, linewidth=0.8, color=color)
    ax.margins(x=0.01)
    ax.set_xticks([])
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _mpl_bar(data: list, height: int = 100, color: str = "#1f77b4"):
    """用 matplotlib 画柱状图，替代 st.bar_chart（避免 pyarrow）。"""
    fig, ax = plt.subplots(figsize=(8, max(1.0, height / 100)))
    ax.bar(range(len(data)), data, width=1.0, color=color)
    ax.margins(x=0.01)
    ax.set_xticks([])
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _mpl_hbar(keys: list, values: list, height: int = 250):
    """用 matplotlib 画水平条形图（情绪权重），替代 st.bar_chart with x/y（避免 pyarrow）。"""
    fig, ax = plt.subplots(figsize=(5, max(2.0, height / 100)))
    y_pos = range(len(keys))
    ax.barh(list(y_pos), values, color="#1f77b4")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(keys, fontsize=8)
    ax.set_xlim(0, max(values) * 1.1 + 0.01)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# 各后端预设模型列表
_TEXT_MODEL_PRESETS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "o4-mini"],
    "qwen": ["qwen-plus", "qwen-max", "qwen-turbo", "qwen3-235b-a22b", "qwen3-32b"],
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

    (
        fps,
        smoothing,
        emotion_backend,
        vision_backend,
        segment_seconds,
        force_switch_seconds,
        gesture_min_hold,
    ) = _sidebar_config()
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
        _tab_generate(
            fps,
            smoothing,
            emotion_backend,
            segment_seconds,
            force_switch_seconds,
            gesture_min_hold,
        )

    with tab4:
        _tab_preview()

    with tab5:
        _tab_model_config(emotion_backend, vision_backend)


if __name__ == "__main__":
    main()
