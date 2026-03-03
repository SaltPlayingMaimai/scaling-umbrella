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
    "2. Reply with ONLY a single English word — the emotion label (lowercase).\n"
    "3. Common labels: calm, happy, excited, sad, angry, panic, surprised, shy, smug, tired, etc.\n"
    "4. If existing emotions are provided, avoid duplicating them unless the image clearly matches.\n"
    "5. Do NOT output any explanation, punctuation, or extra text — just the one word.\n"
)

CLASSIFY_SYSTEM_PROMPT = (
    "You are a character sprite classifier for a VTuber/Live2D avatar system.\n"
    "The user will provide 1-4 numbered character expression images — all variants of the SAME emotion.\n"
    "Your tasks:\n"
    "1. Assign each image to the correct slot:\n"
    "   eo_mo = eyes open  + mouth open\n"
    "   eo_mc = eyes open  + mouth closed\n"
    "   ec_mo = eyes closed + mouth open\n"
    "   ec_mc = eyes closed + mouth closed\n"
    "2. Determine the emotion label from the eo_mo image (eyes open + mouth open) ONLY.\n"
    "   If no eo_mo image exists, use eo_mc; if neither exists, use any available image.\n\n"
    "Respond ONLY with valid JSON — no markdown fences, no comments, no extra text:\n"
    '{"emotion": "<single_word_lowercase>", "assignments": '
    '{"eo_mo": <1-based_int_or_null>, "eo_mc": <1-based_int_or_null>, '
    '"ec_mo": <1-based_int_or_null>, "ec_mc": <1-based_int_or_null>}}\n\n'
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
    "3. Identify the emotion expressed in this image.\n\n"
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
    '"probabilities": {"eo_mo": <0.0-1.0>, "eo_mc": <0.0-1.0>, '
    '"ec_mo": <0.0-1.0>, "ec_mc": <0.0-1.0>}, '
    '"emotion": "<single_word_lowercase>"}\n\n'
    "Output rules:\n"
    "- assigned_slot: the slot key with the highest probability.\n"
    "- probabilities: must sum to ~1.0. A clearly open mouth must NOT have high\n"
    "  probability in mc (mouth-closed) slots.\n"
    "- emotion: a single lowercase English word for the facial expression\n"
    "  (e.g. calm, happy, excited, sad, angry, panic, surprised, shy, smug, tired).\n"
    "  For eyes-closed images, infer from mouth shape and overall expression.\n"
    "  Do NOT use slot names (eo_mo, eo_mc, ec_mo, ec_mc) as emotion labels.\n"
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
    """从 AI 原始输出中提取干净的单词标签。"""
    label = raw.strip().lower()
    label = label.strip(".,!?\"' ")
    label = label.split()[0] if label else "unknown"
    return label


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
        (emotion: str, classified: dict)  其中 classified = {slot: PIL.Image or None}
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

    emotion = _clean_label(data.get("emotion", "unknown"))
    assignments: dict = data.get("assignments", {})

    classified: dict = {slot: None for slot in IMAGE_LABELS}
    for slot, idx in assignments.items():
        if idx is None or slot not in IMAGE_LABELS:
            continue
        idx_int = int(idx) - 1  # 转为 0-based
        if 0 <= idx_int < len(images_list):
            classified[slot] = images_list[idx_int]

    return emotion, classified


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
        max_tokens=120,
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
    result = _parse_classify_response(raw, images_list)
    print(
        f"[ImageRecognizer][OpenAI][classify] emotion='{result[0]}', "
        f"assignments={{ {', '.join(f'{k}: img#{[id(v) for v in images_list].index(id(result[1][k]))+1 if result[1].get(k) is not None else None}' for k in IMAGE_LABELS)} }}"
    )
    return result


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
    result = _parse_classify_response(raw, images_list)
    print(
        f"[ImageRecognizer][Qwen][classify] emotion='{result[0]}', "
        f"assignments={{ {', '.join(f'{k}: img#{[id(v) for v in images_list].index(id(result[1][k]))+1 if result[1].get(k) is not None else None}' for k in IMAGE_LABELS)} }}"
    )
    return result


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
            "emotion": str,
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
    emotion = _clean_label(data.get("emotion", "unknown"))
    return {"assigned_slot": assigned, "probabilities": probs, "emotion": emotion}


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
        max_tokens=120,
        temperature=0.1,
    )

    if not response or not response.choices:
        raise ValueError("OpenAI API 返回了空响应。")
    raw = response.choices[0].message.content or ""
    print(f"[ImageRecognizer][OpenAI][classify_single] raw_response={repr(raw)!r}")
    result = _parse_single_classify_response(raw)
    print(
        f"[ImageRecognizer][OpenAI][classify_single] assigned_slot='{result['assigned_slot']}', "
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
            (emotion: str, classified: dict)
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
            每个结果： {"assigned_slot": ..., "probabilities": {...}, "emotion": ...}
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
                        f"slot='{res['assigned_slot']}' emotion='{res['emotion']}'"
                    )
                except Exception as exc:
                    img_idx = futures[future]
                    print(
                        f"[ImageRecognizer] batch parallel: img#{img_idx+1} FAILED: {exc}"
                    )
                    results[img_idx] = None

        return results
