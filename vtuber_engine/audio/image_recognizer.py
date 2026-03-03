"""
Image Emotion Recognizer — 使用 AI Vision API 识别表情图片的情绪。

职责：
  - 接收一组角色差分图片（4 张）
  - 调用 AI Vision 分析这些图片表达的是什么情绪
  - 返回英文情绪标签（如 "happy", "angry", "calm" 等）

支持后端：
  - "openai"  — OpenAI GPT-4o Vision
  - "qwen"    — 通义千问 Qwen VL（通过 DashScope）

使用方式：
  recognizer = ImageEmotionRecognizer(backend="qwen")
  emotion_label = recognizer.recognize(images)
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# 自动加载项目根目录下的 .env
load_dotenv()


SYSTEM_PROMPT = (
    "You are a character emotion classifier for a VTuber/Live2D-style avatar system. "
    "The user will show you character expression sprites (variants of open/closed eyes and mouth). "
    "Your task is to determine the single emotion these sprites represent.\n\n"
    "Rules:\n"
    "1. Determine the emotion PRIMARILY from the [eyes open + mouth open] sprite, "
    "as it best shows the full facial expression. "
    "If that sprite is absent, use [eyes open + mouth closed] instead.\n"
    "2. **Eye-priority**: Focus on the EYES first — eye shape, eyelid position, pupil size, "
    "eyebrow angle are the primary indicators of emotion. "
    "Mouth is secondary; only use it to refine or disambiguate.\n"
    "3. Reply with ONLY a single English word — the emotion label (lowercase).\n"
    "4. Common labels: calm, happy, excited, sad, angry, panic, surprised, shy, smug, tired, etc.\n"
    "5. If existing emotions are provided, avoid duplicating them unless the image clearly matches.\n"
    "6. Do NOT output any explanation, punctuation, or extra text — just the one word.\n"
)

# 标准情绪维度列表（与 data_models.EMOTION_KEYS 对齐）
from vtuber_engine.models.data_models import EMOTION_KEYS

EMOTION_DIMENSIONS = EMOTION_KEYS

CLASSIFY_SYSTEM_PROMPT = (
    "You are a character sprite classifier for a VTuber/Live2D avatar system.\n"
    "The user will provide 1-4 numbered character expression images — all variants of the SAME emotion.\n"
    "Your tasks:\n"
    "1. Assign each image to the correct slot:\n"
    "   eo_mo = eyes open  + mouth open\n"
    "   eo_mc = eyes open  + mouth closed\n"
    "   ec_mo = eyes closed + mouth open\n"
    "   ec_mc = eyes closed + mouth closed\n"
    "2. Analyze the expression and output an emotion_vector from the eo_mo image.\n"
    "   If no eo_mo image exists, use eo_mc; if neither exists, use any available image.\n\n"
    "**Facial analysis priority (high to low):**\n"
    "- EYEBROWS (30%): angle (raised/furrowed/flat), curvature, asymmetry.\n"
    "  Raised brows = surprise/excitement; furrowed brows = anger/concentration; soft arch = calm/tender.\n"
    "- EYES (35%): openness, pupil size/shape (sparkles/stars/hearts = excitement/love),\n"
    "  eyelid droop (tired/sad), tear marks, eye direction, iris patterns or symbols.\n"
    "- MOUTH (20%): open/closed, smile/frown/pout, teeth visibility, tongue out.\n"
    "- OTHER (15%): blush marks (shy/embarrassed), sweat drops (panic/nervous), overall face tilt.\n"
    "Report observations in eyebrow_analysis, eye_analysis, mouth_analysis fields.\n\n"
    "Respond ONLY with valid JSON — no markdown fences, no comments, no extra text:\n"
    '{"label": "<unique_compound_label>", '
    '"eyebrow_analysis": "<brief eyebrow description>", '
    '"eye_analysis": "<brief eye expression description>", '
    '"mouth_analysis": "<brief mouth expression description>", '
    '"emotion_vector": {'
    '"calm": <0-1>, "happy": <0-1>, "excited": <0-1>, "sad": <0-1>, '
    '"angry": <0-1>, "panic": <0-1>, "shy": <0-1>, "surprised": <0-1>, '
    '"tender": <0-1>, "smug": <0-1>, "tired": <0-1>, "confused": <0-1>}, '
    '"assignments": {"eo_mo": <int_or_null>, "eo_mc": <int_or_null>, '
    '"ec_mo": <int_or_null>, "ec_mc": <int_or_null>}}\n\n'
    "Field rules:\n"
    "- label: short unique identifier, 1-3 lowercase words joined by underscores. "
    "  Append a distinguishing modifier when the primary emotion may be duplicated "
    "  (e.g. happy_bright, happy_shy, calm_tired, angry_tense). Single word is fine if unique.\n"
    "- eyebrow_analysis: 1 short sentence about eyebrow shape, angle, and what it conveys.\n"
    "- eye_analysis: 1 short sentence about eye state including any special symbols or patterns inside the eyes.\n"
    "- mouth_analysis: 1 short sentence about the mouth expression.\n"
    "- emotion_vector: percentage weights for ALL 12 emotion dimensions. "
    "  All values 0.0-1.0, they should sum to approximately 1.0. "
    "  Use fine-grained distinctions: e.g. a shy smile is NOT just happy—it should have "
    "  high shy + some happy + some calm. An excited grin should be high excited + some happy.\n"
    "Detection rules:\n"
    "- Eyes open:  irises and pupils clearly visible; eyelids raised.\n"
    "- Eyes closed: eyelids are shut or nearly shut, irises not visible.\n"
    "- Mouth open:  lips are parted, teeth or tongue visible, or a clear gap between lips.\n"
    "- Mouth closed: lips pressed together with no gap.\n"
    "- ec_mo (eyes closed + mouth open) is a valid and common state — look carefully for it.\n"
    "  A sprite with drooping/closed eyes AND an open mouth should be assigned to ec_mo.\n"
    "- Each image index (1, 2, 3 ...) may appear AT MOST ONCE across all assignments.\n"
    "- Use null if no image clearly matches that slot.\n"
    "- If an existing emotions list is provided, avoid duplicating those labels.\n"
)

SINGLE_CLASSIFY_SYSTEM_PROMPT = (
    "You are a character sprite classifier for a VTuber/Live2D avatar system.\n"
    "The user provides a SINGLE character expression image.\n"
    "Your tasks:\n"
    "1. Classify this image into the best-matching slot.\n"
    "2. Estimate the probability for each of the four possible slots.\n"
    "3. Output an emotion_vector representing the blend of emotions.\n\n"
    "**Facial analysis priority (high to low):**\n"
    "- EYEBROWS (30%): angle (raised/furrowed/flat), curvature, asymmetry.\n"
    "  Raised brows = surprise/excitement; furrowed brows = anger/concentration; soft arch = calm/tender.\n"
    "- EYES (35%): openness, pupil size/shape (sparkles/stars/hearts = excitement/love),\n"
    "  eyelid droop (tired/sad), tear marks, eye direction, iris patterns or symbols.\n"
    "- MOUTH (20%): open/closed, smile/frown/pout, teeth visibility, tongue out.\n"
    "- OTHER (15%): blush marks (shy/embarrassed), sweat drops (panic/nervous), overall face tilt.\n"
    "Report observations in eyebrow_analysis, eye_analysis, mouth_analysis fields.\n\n"
    "Slot definitions:\n"
    "   eo_mo = eyes open  + mouth open\n"
    "   eo_mc = eyes open  + mouth closed\n"
    "   ec_mo = eyes closed + mouth open\n"
    "   ec_mc = eyes closed + mouth closed\n\n"
    "Detection rules:\n"
    "- Eyes open:  irises and pupils clearly visible; eyelids raised.\n"
    "- Eyes closed: eyelids are shut or nearly shut, irises not visible.\n"
    "- Mouth open:  lips are parted, teeth or tongue visible, or a clear gap between lips.\n"
    "- Mouth closed: lips pressed together with no gap.\n"
    "- ec_mo (eyes closed + mouth open) is a valid and common state.\n\n"
    "Respond ONLY with valid JSON — no markdown fences, no comments, no extra text:\n"
    '{"assigned_slot": "<best_slot>", '
    '"probabilities": {"eo_mo": <0-1>, "eo_mc": <0-1>, '
    '"ec_mo": <0-1>, "ec_mc": <0-1>}, '
    '"label": "<unique_compound_label>", '
    '"eyebrow_analysis": "<brief eyebrow description>", '
    '"eye_analysis": "<brief eye expression description>", '
    '"mouth_analysis": "<brief mouth expression description>", '
    '"emotion_vector": {'
    '"calm": <0-1>, "happy": <0-1>, "excited": <0-1>, "sad": <0-1>, '
    '"angry": <0-1>, "panic": <0-1>, "shy": <0-1>, "surprised": <0-1>, '
    '"tender": <0-1>, "smug": <0-1>, "tired": <0-1>, "confused": <0-1>}}\n\n'
    "Output rules:\n"
    "- assigned_slot: the slot key with the highest probability.\n"
    "- probabilities: must sum to ~1.0. A clearly open mouth must NOT have high\n"
    "  probability in mc (mouth-closed) slots.\n"
    "- label: short unique identifier (1-3 words, underscores, no spaces). "
    "  Append a modifier when the primary emotion may be ambiguous or duplicated "
    "  (e.g. happy_bright, happy_shy, calm_tired, angry_tense). Single word is fine if unique.\n"
    "- eyebrow_analysis: 1 short sentence about eyebrow expression.\n"
    "- eye_analysis: 1 short sentence about eye expression including any special patterns.\n"
    "- mouth_analysis: 1 short sentence about the mouth expression.\n"
    "- emotion_vector: percentage weights for ALL 12 emotion dimensions "
    "  (calm, happy, excited, sad, angry, panic, shy, surprised, tender, smug, tired, confused). "
    "  All values 0.0-1.0, sum to ~1.0. Use fine-grained distinctions.\n"
)

IMAGE_LABELS = {
    "eo_mo": "eyes open + mouth open",
    "eo_mc": "eyes open + mouth closed",
    "ec_mo": "eyes closed + mouth open",
    "ec_mc": "eyes closed + mouth closed",
}


def _pil_to_base64(image) -> str:
    """将 PIL Image 转为 base64 data URI。"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _clean_label(raw: str) -> str:
    """从 AI 原始输出中提取干净的复合标签（如 happy、happy_bright）。"""
    label = raw.strip().lower()
    label = label.strip(".,!?\"' ")
    # 移除可能包裹的引号
    label = label.strip("'\"")
    if not label:
        return "unknown"
    # 如果已经是下划线连接形式（如 happy_bright），直接保留
    # 如果是空格连接（如 "happy bright"），转为下划线，最多取前 3 个词
    parts = label.replace("-", "_").split()
    parts = [p for p in parts if p.isalpha() or ("_" in p)]
    if not parts:
        return "unknown"
    # 最多保留三段
    return "_".join(parts[:3])


# ──────────────────────────────────────────────
# OpenAI 后端
# ──────────────────────────────────────────────


def _recognize_openai(
    images: Dict[str, Any],
    existing_emotions: Optional[List[str]] = None,
    model: str = "gpt-4o",
) -> str:
    """通过 OpenAI Vision API 识别情绪。"""
    print(
        f"[ImageRecognizer][OpenAI] _recognize_openai called, model={model}, "
        f"images_slots={[k for k,v in images.items() if v is not None]}, "
        f"existing_emotions={existing_emotions}"
    )
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package is required. Install with: pip install openai"
        )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 未设置。请在 .env 文件中配置。")

    base_url = os.environ.get("OPENAI_BASE_URL", None)
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = openai.OpenAI(**kwargs)

    # 构建多图 message
    content_parts: list = []

    existing_info = ""
    if existing_emotions:
        existing_info = (
            f"\nAlready used emotions: {', '.join(existing_emotions)}. "
            "Try to pick a different label.\n"
        )

    content_parts.append(
        {
            "type": "text",
            "text": (
                "Here are 4 character expression sprites for one emotion state. "
                "They show the same emotion with eye-open/closed and mouth-open/closed variants."
                f"{existing_info}"
                "\nWhat emotion do these sprites represent? Reply with ONE word only."
            ),
        }
    )

    for suffix in ["eo_mo", "eo_mc", "ec_mo", "ec_mc"]:
        img = images.get(suffix)
        if img is not None:
            content_parts.append({"type": "text", "text": f"[{IMAGE_LABELS[suffix]}]:"})
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _pil_to_base64(img), "detail": "low"},
                }
            )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ],
        max_tokens=20,
        temperature=0.1,
    )

    if response is None:
        raise ValueError("OpenAI API 返回了空响应。请检查 API Key 和网络连接。")
    if not response.choices or response.choices[0] is None:
        raise ValueError("OpenAI API 返回了无效的 choices。请重试。")
    if response.choices[0].message is None:
        raise ValueError("OpenAI API 返回了无效的 message。请重试。")

    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI API 返回了空的 message 内容。请重试。")

    result = _clean_label(content)
    print(f"[ImageRecognizer][OpenAI][recognize] raw_response={repr(content)!r}")
    print(f"[ImageRecognizer][OpenAI][recognize] cleaned_result='{result}'")
    return result


# ──────────────────────────────────────────────
# Qwen (DashScope) 后端
# ──────────────────────────────────────────────


def _recognize_qwen(
    images: Dict[str, Any],
    existing_emotions: Optional[List[str]] = None,
    model: str = "qwen-vl-max",
) -> str:
    """通过通义千问 DashScope MultiModalConversation 识别情绪。"""
    print(
        f"[ImageRecognizer][Qwen] _recognize_qwen called, model={model}, "
        f"images_slots={[k for k,v in images.items() if v is not None]}, "
        f"existing_emotions={existing_emotions}"
    )
    try:
        import dashscope
        from dashscope import MultiModalConversation
    except ImportError:
        raise ImportError(
            "dashscope package is required. Install with: pip install dashscope"
        )

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 未设置。请在 .env 文件中配置。")

    # 设置 base_url（默认北京地域）
    base_url = os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
    )
    dashscope.base_http_api_url = base_url

    # 构建 content 列表（Qwen VL 格式：image + text 混合）
    existing_info = ""
    if existing_emotions:
        existing_info = (
            f"\nAlready used emotions: {', '.join(existing_emotions)}. "
            "Try to pick a different label.\n"
        )

    user_content: list = []

    # 添加图片（base64 data URI）
    for suffix in ["eo_mo", "eo_mc", "ec_mo", "ec_mc"]:
        img = images.get(suffix)
        if img is not None:
            user_content.append({"image": _pil_to_base64(img)})
            user_content.append({"text": f"[{IMAGE_LABELS[suffix]}]"})

    # 添加文字提示
    user_content.append(
        {
            "text": (
                "Above are 4 character expression sprites for one emotion state. "
                "They show the same emotion with eye-open/closed and mouth-open/closed variants."
                f"{existing_info}"
                "\nWhat emotion do these sprites represent? Reply with ONE word only."
            ),
        }
    )

    messages = [
        {"role": "system", "content": [{"text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

    response = MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=messages,
    )

    if response is None:
        raise ValueError("Qwen API 返回了空响应。请检查 API Key 和网络连接。")

    output = getattr(response, "output", None)
    if output is None:
        code = (
            response.get("code", "")
            if hasattr(response, "get")
            else getattr(response, "code", "")
        )
        message = (
            response.get("message", "")
            if hasattr(response, "get")
            else getattr(response, "message", "")
        )
        raise ValueError(f"Qwen API 调用失败（code={code}）: {message}")

    choices = getattr(output, "choices", None)
    if not choices:
        raise ValueError("Qwen API 返回了无效的 choices。")

    content_data = choices[0].message.content
    if isinstance(content_data, list):
        raw_text = (
            content_data[0].get("text", "")
            if isinstance(content_data[0], dict)
            else str(content_data[0])
        )
    else:
        raw_text = str(content_data) if content_data else ""

    if not raw_text:
        raise ValueError("Qwen API 返回的文本内容为空。")

    result = _clean_label(raw_text)
    print(f"[ImageRecognizer][Qwen][recognize] raw_response={repr(raw_text)!r}")
    print(f"[ImageRecognizer][Qwen][recognize] cleaned_result='{result}'")
    return result


# ──────────────────────────────────────────────
# 自动分类辅助
# ──────────────────────────────────────────────

import json as _json
import re as _re


def _parse_classify_response(
    raw_text: str,
    images_list: List[Any],
) -> tuple:
    """
    解析 AI 返回的 JSON，映射图片索引到分类键。

    Returns:
        (label: str, emotion_vector: dict, classified: dict)
        label: 唯一复合标签（如 happy_bright）
        emotion_vector: 情绪百分比向量 {calm: 0.1, happy: 0.6, ...}
        classified: {slot: PIL.Image or None}
    """
    # 去除可能的 markdown 代码块
    text = raw_text.strip()
    if text.startswith("```"):
        text = _re.sub(r"^```[a-z]*\n?", "", text)
        text = _re.sub(r"\n?```$", "", text.strip())

    # 提取 JSON
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        # 如果有多余内容，尝试找第一个 {...}
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if m:
            data = _json.loads(m.group())
        else:
            raise ValueError(f"AI 返回的内容无法解析为 JSON：{raw_text[:200]}")

    # label 字段（向量驱动，无文字情绪）
    raw_label = data.get("label", "unknown")
    label = _clean_label(raw_label)

    # 解析情绪向量（百分比）
    raw_ev = data.get("emotion_vector", {})
    emotion_vector: dict = {}
    for dim in EMOTION_DIMENSIONS:
        emotion_vector[dim] = float(raw_ev.get(dim, 0.0))
    # 归一化
    ev_total = sum(emotion_vector.values()) or 1.0
    emotion_vector = {k: round(v / ev_total, 3) for k, v in emotion_vector.items()}

    eyebrow_analysis = data.get("eyebrow_analysis", "")
    eye_analysis = data.get("eye_analysis", "")
    mouth_analysis = data.get("mouth_analysis", "")

    print(
        f"[ImageRecognizer][classify] label='{label}', "
        f"emotion_vector={emotion_vector}, "
        f"eyebrow='{eyebrow_analysis[:60]}', "
        f"eye='{eye_analysis[:60]}', mouth='{mouth_analysis[:60]}'"
    )

    assignments: dict = data.get("assignments", {})

    classified: dict = {slot: None for slot in IMAGE_LABELS}
    for slot, idx in assignments.items():
        if idx is None or slot not in IMAGE_LABELS:
            continue
        idx_int = int(idx) - 1  # 转为 0-based
        if 0 <= idx_int < len(images_list):
            classified[slot] = images_list[idx_int]

    return label, emotion_vector, classified


def _classify_openai(
    images_list: List[Any],
    existing_emotions: Optional[List[str]] = None,
    model: str = "gpt-4o",
) -> tuple:
    """通过 OpenAI Vision 自动分类每张图并识别情绪。"""
    print(
        f"[ImageRecognizer][OpenAI] _classify_openai called, model={model}, "
        f"n_images={len(images_list)}, existing_emotions={existing_emotions}"
    )
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package is required. Install with: pip install openai"
        )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 未设置。请在 .env 文件中配置。")

    base_url = os.environ.get("OPENAI_BASE_URL", None)
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = openai.OpenAI(**kwargs)

    existing_info = ""
    if existing_emotions:
        existing_info = f" Already used emotions: {', '.join(existing_emotions)}. Try to pick a different label."

    content_parts: list = [
        {
            "type": "text",
            "text": (
                f"Here are {len(images_list)} character expression sprite(s) (numbered)."
                f"{existing_info}"
                " Classify each by eye/mouth state and determine the emotion."
            ),
        }
    ]

    for i, img in enumerate(images_list, start=1):
        content_parts.append({"type": "text", "text": f"Image {i}:"})
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": _pil_to_base64(img), "detail": "low"},
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ],
        max_tokens=300,
        temperature=0.1,
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
    print(f"[ImageRecognizer][OpenAI][classify] raw_response={repr(raw)!r}")
    label, emotion_vector, classified2 = _parse_classify_response(raw, images_list)
    print(
        f"[ImageRecognizer][OpenAI][classify] label='{label}', "
        f"emotion_vector={emotion_vector}, "
        f"assignments={{ {', '.join(f'{k}: img#{[id(v) for v in images_list].index(id(classified2[k]))+1 if classified2.get(k) is not None else None}' for k in IMAGE_LABELS)} }}"
    )
    return label, emotion_vector, classified2


def _classify_qwen(
    images_list: List[Any],
    existing_emotions: Optional[List[str]] = None,
    model: str = "qwen-vl-max",
) -> tuple:
    """通过通义千问 Qwen VL 自动分类每张图并识别情绪。"""
    print(
        f"[ImageRecognizer][Qwen] _classify_qwen called, model={model}, "
        f"n_images={len(images_list)}, existing_emotions={existing_emotions}"
    )
    try:
        import dashscope
        from dashscope import MultiModalConversation
    except ImportError:
        raise ImportError(
            "dashscope package is required. Install with: pip install dashscope"
        )

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 未设置。请在 .env 文件中配置。")

    base_url = os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
    )
    dashscope.base_http_api_url = base_url

    existing_info = ""
    if existing_emotions:
        existing_info = f" Already used emotions: {', '.join(existing_emotions)}. Try to pick a different label."

    user_content: list = []
    for i, img in enumerate(images_list, start=1):
        user_content.append({"image": _pil_to_base64(img)})
        user_content.append({"text": f"Image {i}:"})

    user_content.append(
        {
            "text": (
                f"Above are {len(images_list)} character expression sprite(s) (numbered)."
                f"{existing_info}"
                " Classify each by eye/mouth state and determine the emotion."
            )
        }
    )

    messages = [
        {"role": "system", "content": [{"text": CLASSIFY_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

    response = MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=messages,
    )

    if response is None:
        raise ValueError("Qwen API 返回了空响应。请检查 API Key 和网络连接。")

    output = getattr(response, "output", None)

    if output is None:
        # 提取 API 返回的错误信息
        code = (
            response.get("code", "")
            if hasattr(response, "get")
            else getattr(response, "code", "")
        )
        message = (
            response.get("message", "")
            if hasattr(response, "get")
            else getattr(response, "message", "")
        )
        raise ValueError(f"Qwen API 调用失败（code={code}）: {message}")

    choices = getattr(output, "choices", None)
    if not choices:
        raise ValueError(f"Qwen API 返回了无效的 choices。")

    content_data = choices[0].message.content
    # Qwen VL 返回的 content 可能是 list[dict] 或 str
    if isinstance(content_data, list):
        raw = (
            content_data[0].get("text", "")
            if isinstance(content_data[0], dict)
            else str(content_data[0])
        )
    else:
        raw = str(content_data) if content_data else ""

    if not raw:
        raise ValueError("Qwen API 返回的文本内容为空。")

    print(f"[ImageRecognizer][Qwen][classify] raw_response={repr(raw)!r}")
    label, emotion_vector, classified2 = _parse_classify_response(raw, images_list)
    print(
        f"[ImageRecognizer][Qwen][classify] label='{label}', "
        f"emotion_vector={emotion_vector}, "
        f"assignments={{ {', '.join(f'{k}: img#{[id(v) for v in images_list].index(id(classified2[k]))+1 if classified2.get(k) is not None else None}' for k in IMAGE_LABELS)} }}"
    )
    return label, emotion_vector, classified2


# ──────────────────────────────────────────────
# 单图分类（逐张上传模式）
# ──────────────────────────────────────────────


def _parse_single_classify_response(raw_text: str) -> dict:
    """
    解析单图分类的 AI JSON 响应。

    Returns:
        {
            "assigned_slot": str,
            "probabilities": {slot: float},
            "label": str,               # 唯一复合标签，如 happy_bright
            "eyebrow_analysis": str,     # 眉毛分析
            "eye_analysis": str,         # 眼部分析
            "mouth_analysis": str,       # 嘴部分析
            "emotion_vector": dict,      # 情绪百分比向量 (12 维)
        }
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = _re.sub(r"^```[a-z]*\n?", "", text)
        text = _re.sub(r"\n?```$", "", text.strip())
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if m:
            data = _json.loads(m.group())
        else:
            raise ValueError(f"AI 返回的内容无法解析为 JSON：{raw_text[:200]}")

    assigned = data.get("assigned_slot", "eo_mc")
    probs_raw: dict = data.get("probabilities", {})
    # 确保所有 slot 都有值
    probs = {slot: float(probs_raw.get(slot, 0.0)) for slot in IMAGE_LABELS}
    # 归一化
    total = sum(probs.values()) or 1.0
    probs = {k: round(v / total, 4) for k, v in probs.items()}
    if assigned not in IMAGE_LABELS:
        assigned = max(probs, key=probs.get)
    # label: 直接取 label 字段
    raw_label = data.get("label", "unknown")
    label = _clean_label(raw_label)
    eyebrow_analysis: str = data.get("eyebrow_analysis", "")
    eye_analysis: str = data.get("eye_analysis", "")
    mouth_analysis: str = data.get("mouth_analysis", "")

    # 解析情绪向量
    raw_ev = data.get("emotion_vector", {})
    emotion_vector: dict = {}
    for dim in EMOTION_DIMENSIONS:
        emotion_vector[dim] = float(raw_ev.get(dim, 0.0))
    ev_total = sum(emotion_vector.values()) or 1.0
    emotion_vector = {k: round(v / ev_total, 3) for k, v in emotion_vector.items()}

    return {
        "assigned_slot": assigned,
        "probabilities": probs,
        "label": label,
        "eyebrow_analysis": eyebrow_analysis,
        "eye_analysis": eye_analysis,
        "mouth_analysis": mouth_analysis,
        "emotion_vector": emotion_vector,
    }


def _classify_single_openai(
    image: Any,
    existing_slots: Optional[List[str]] = None,
    model: str = "gpt-4o",
) -> dict:
    """通过 OpenAI Vision 对单张图片分类（返回 slot + 概率）。"""
    print(
        f"[ImageRecognizer][OpenAI] _classify_single_openai called, model={model}, "
        f"existing_slots={existing_slots}"
    )
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package is required. Install with: pip install openai"
        )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 未设置。请在 .env 文件中配置。")

    base_url = os.environ.get("OPENAI_BASE_URL", None)
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)

    existing_info = ""
    if existing_slots:
        existing_info = f" Already assigned slots: {', '.join(existing_slots)}. Avoid assigning to those if possible."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SINGLE_CLASSIFY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Classify this sprite image.{existing_info}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _pil_to_base64(image), "detail": "low"},
                    },
                ],
            },
        ],
        max_tokens=300,
        temperature=0.1,
    )

    if not response or not response.choices:
        raise ValueError("OpenAI API 返回了空响应。")
    raw = response.choices[0].message.content or ""
    print(f"[ImageRecognizer][OpenAI][classify_single] raw_response={repr(raw)!r}")
    result = _parse_single_classify_response(raw)
    print(
        f"[ImageRecognizer][OpenAI][classify_single] assigned_slot='{result['assigned_slot']}', "
        f"label='{result['label']}', "
        f"emotion_vector={result.get('emotion_vector', {})}, "
        f"eyebrow='{result.get('eyebrow_analysis', '')[:60]}', "
        f"eye='{result.get('eye_analysis', '')[:60]}', "
        f"mouth='{result.get('mouth_analysis', '')[:60]}', "
        f"probabilities={result['probabilities']}"
    )
    return result


def _classify_single_qwen(
    image: Any,
    existing_slots: Optional[List[str]] = None,
    model: str = "qwen-vl-max",
) -> dict:
    """通过 Qwen VL 对单张图片分类（返回 slot + 概率）。"""
    print(
        f"[ImageRecognizer][Qwen] _classify_single_qwen called, model={model}, "
        f"existing_slots={existing_slots}"
    )
    try:
        import dashscope
        from dashscope import MultiModalConversation
    except ImportError:
        raise ImportError(
            "dashscope package is required. Install with: pip install dashscope"
        )

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 未设置。请在 .env 文件中配置。")

    base_url = os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
    )
    dashscope.base_http_api_url = base_url

    existing_info = ""
    if existing_slots:
        existing_info = f" Already assigned slots: {', '.join(existing_slots)}. Avoid assigning to those if possible."

    user_content: list = [
        {"image": _pil_to_base64(image)},
        {"text": f"Classify this sprite image.{existing_info}"},
    ]
    messages = [
        {"role": "system", "content": [{"text": SINGLE_CLASSIFY_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

    response = MultiModalConversation.call(
        api_key=api_key, model=model, messages=messages
    )
    if response is None:
        raise ValueError("Qwen API 返回了空响应。")

    output = getattr(response, "output", None)
    if output is None:
        code = (
            getattr(response, "code", "")
            if not hasattr(response, "get")
            else response.get("code", "")
        )
        msg = (
            getattr(response, "message", "")
            if not hasattr(response, "get")
            else response.get("message", "")
        )
        raise ValueError(f"Qwen API 调用失败（code={code}）: {msg}")

    choices = getattr(output, "choices", None)
    if not choices:
        raise ValueError("Qwen API 返回了无效的 choices。")

    content_data = choices[0].message.content
    if isinstance(content_data, list):
        raw = (
            content_data[0].get("text", "")
            if isinstance(content_data[0], dict)
            else str(content_data[0])
        )
    else:
        raw = str(content_data) if content_data else ""

    if not raw:
        raise ValueError("Qwen API 返回的文本内容为空。")

    print(f"[ImageRecognizer][Qwen][classify_single] raw_response={repr(raw)!r}")
    result = _parse_single_classify_response(raw)
    print(
        f"[ImageRecognizer][Qwen][classify_single] assigned_slot='{result['assigned_slot']}', "
        f"label='{result['label']}', "
        f"emotion_vector={result.get('emotion_vector', {})}, "
        f"eyebrow='{result.get('eyebrow_analysis', '')[:60]}', "
        f"eye='{result.get('eye_analysis', '')[:60]}', "
        f"mouth='{result.get('mouth_analysis', '')[:60]}', "
        f"probabilities={result['probabilities']}"
    )
    return result


# ──────────────────────────────────────────────
# 统一入口
# ──────────────────────────────────────────────


class ImageEmotionRecognizer:
    """使用 AI Vision 识别一组表情素材对应的情绪。"""

    def __init__(self, backend: str = "openai", model: Optional[str] = None):
        """
        Args:
            backend: "openai" 或 "qwen"。
            model: 指定模型名。None 时使用后端默认值。
        """
        self.backend = backend
        if model:
            self.model = model
        else:
            self.model = "gpt-4o" if backend == "openai" else "qwen-vl-max"

    def recognize(
        self,
        images: Dict[str, Any],
        existing_emotions: Optional[List[str]] = None,
    ) -> str:
        """
        识别一组表情图片的情绪。

        Args:
            images: 键为 "eo_mo"/"eo_mc"/"ec_mo"/"ec_mc"，值为 PIL Image。
            existing_emotions: 当前已有的情绪标签列表，帮助 AI 避免重复。

        Returns:
            情绪标签字符串（如 "happy"）。
        """
        if self.backend == "openai":
            return _recognize_openai(images, existing_emotions, self.model)
        elif self.backend == "qwen":
            return _recognize_qwen(images, existing_emotions, self.model)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def classify_and_recognize(
        self,
        images_list: List[Any],
        existing_emotions: Optional[List[str]] = None,
    ) -> tuple:
        """
        自动将一组无标签图片分类到眼/嘴状态槽，并识别情绪。

        Args:
            images_list: PIL Image 列表（顺序任意，最多 4 张）。
            existing_emotions: 当前已有的情绪标签列表，帮助 AI 避免重复。

        Returns:
            (label: str, emotion_vector: dict, classified: dict)
            label: 唯一复合标签（如 happy_bright）
            emotion_vector: 情绪百分比向量 {calm: 0.1, happy: 0.6, ...}
            classified 键为 "eo_mo"/"eo_mc"/"ec_mo"/"ec_mc"，值为 PIL Image 或 None。
        """
        if not images_list:
            raise ValueError("images_list 不能为空")
        if self.backend == "openai":
            return _classify_openai(images_list, existing_emotions, self.model)
        elif self.backend == "qwen":
            return _classify_qwen(images_list, existing_emotions, self.model)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def classify_single(
        self,
        image: Any,
        existing_slots: Optional[List[str]] = None,
    ) -> dict:
        """
        对单张图片进行眼/嘴状态分类，返回最优 slot 及各 slot 的概率。

        Args:
            image: PIL Image 对象。
            existing_slots: 已分配的 slot 列表（辅助 AI 避免重复分配）。

        Returns:
            {
                "assigned_slot": "eo_mo",  # 最高概率的 slot
                "probabilities": {
                    "eo_mo": 0.85,
                    "eo_mc": 0.08,
                    "ec_mo": 0.04,
                    "ec_mc": 0.03,
                }
            }
        """
        print(
            f"[ImageRecognizer] classify_single called, backend={self.backend}, "
            f"model={self.model}, existing_slots={existing_slots}"
        )
        if self.backend == "openai":
            return _classify_single_openai(image, existing_slots, self.model)
        elif self.backend == "qwen":
            return _classify_single_qwen(image, existing_slots, self.model)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def classify_batch_parallel(
        self,
        images: List[Any],
    ) -> List[Optional[dict]]:
        """并行对最多 4 张图片分类，每张独立调用 AI。

        Args:
            images: 列表，元素为 PIL Image 或 None（None 的位置不调用）。

        Returns:
            与输入等长的列表，元素为分类结果字典或 None。
            每个结果： {"assigned_slot": ..., "probabilities": {...}, "label": ..., "emotion_vector": ...}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(
            f"[ImageRecognizer] classify_batch_parallel called, backend={self.backend}, "
            f"model={self.model}, n_images={sum(1 for x in images if x is not None)}"
        )

        results: List[Optional[dict]] = [None] * len(images)
        tasks = [(i, img) for i, img in enumerate(images) if img is not None]

        def _classify_one(idx: int, img: Any):
            return idx, self.classify_single(image=img, existing_slots=None)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_classify_one, i, img): i for i, img in tasks}
            for future in as_completed(futures):
                try:
                    idx, res = future.result()
                    results[idx] = res
                    print(
                        f"[ImageRecognizer] batch parallel: img#{idx+1} done -> "
                        f"slot='{res['assigned_slot']}' label='{res.get('label', '?')}'"
                    )
                except Exception as exc:
                    img_idx = futures[future]
                    print(
                        f"[ImageRecognizer] batch parallel: img#{img_idx+1} FAILED: {exc}"
                    )
                    results[img_idx] = None

        return results
