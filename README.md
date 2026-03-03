# VTuber Engine — AI 驱动角色动画生成系统

> AI 理解情绪 → 状态引擎决策 → 动画引擎平滑 → 渲染器输出 → 绿幕视频

面向 **视频制作** 的离线角色动画生成系统。
支持自定义 OC 表情差分（眼开/眼闭 × 嘴开/嘴闭）、AI 音频情绪识别、TTS 接入。
通过 **Streamlit 网页界面** 操作，素材只在内存中使用，**不会上传到任何地方**。

---

## 🚀 给代码小白的快速上手指南

### 你需要准备什么

1. **Python 3.10+**（[下载地址](https://www.python.org/downloads/)，安装时勾选"Add to PATH"）
2. **ffmpeg**（视频处理工具，[下载地址](https://ffmpeg.org/download.html)，需要加入系统 PATH）
3. **你的角色素材图片**（PNG 格式，透明背景最佳）

### 素材怎么准备？

你的角色的每种 **表情** 需要画 **4 张** 差分图：

| 图片 | 眼睛 | 嘴巴 | 什么时候用 |
|------|------|------|------------|
| 图1 | 👁️ 睁开 | 👄 张开 | 角色说话时，眼睛正常 |
| 图2 | 👁️ 睁开 | 👄 闭合 | 角色不说话时，眼睛正常 |
| 图3 | 👁️ 闭合 | 👄 张开 | 说话 + 眨眼瞬间 |
| 图4 | 👁️ 闭合 | 👄 闭合 | 不说话 + 眨眼瞬间 |

**举例**：如果你设了 3 种表情（calm / excited / panic），就需要 3 × 4 = 12 张图。

> 💡 图片不需要提前放到项目文件夹里！每次打开工具时直接在网页上传就好。
> 关闭网页后图片自动消失，不会留在项目里，也不会被 git 上传。

### 三步运行

```bash
# 第 1 步：安装依赖（只需运行一次）
pip install -r requirements.txt

# 第 2 步：启动网页界面
streamlit run vtuber_engine/streamlit_app.py

# 第 3 步：在浏览器里操作（会自动打开）
#   → 上传素材图片
#   → 上传配音 或 输入文字生成语音
#   → 点击"生成视频"
#   → 下载绿幕 MP4
```

---

## 📦 各模块做什么？（通俗版）

把整个系统想象成一条 **流水线**，你的音频从开始到最后经过每一站：

```
你的配音 / TTS 文字
       ↓
🔍 Audio Analyzer   ←  "听"出音量大小、有没有在说话
       ↓
🧠 Emotion Engine   ←  AI 判断这段音频是什么情绪
       ↓
🎯 State Engine     ←  根据情绪 + 音量，决定角色"状态参数"
       ↓
✨ Animation Engine  ←  让状态变化变得平滑自然
       ↓
🖼️ Renderer         ←  根据参数选出对应的图片，放到绿幕上
       ↓
🎬 Video Exporter   ←  把所有帧拼成视频 + 合上配音
       ↓
📺 你的绿幕视频.mp4
```

### 每个模块详细说明

| 模块 | 文件 | 通俗解释 |
|------|------|---------|
| **Audio Analyzer** | `audio/analyzer.py` | 就像看音频的"波形图"，找出哪些地方在说话、声音多大 |
| **Emotion Engine** | `audio/emotion_engine.py` | 根据音量、语速等判断情绪。输出 "60%平静+30%开心+10%紧张" 这样的权重，不直接选表情 |
| **State Engine** | `core/state_engine.py` | 收到情绪信息后，决定"嘴要张开多少""眼睛该不该眨""用哪种表情"。还能防止表情疯狂跳来跳去 |
| **Animation Engine** | `core/animation_engine.py` | 让变化更自然——嘴从闭到开不是"啪"一下切换，而是有个过渡 |
| **Renderer** | `render/renderer.py` | 根据"眼开+嘴开"之类的参数，从你上传的4张图里选出正确的那张，放到绿色背景上 |
| **Video Exporter** | `export/video_exporter.py` | 把几百帧图片拼成 MP4 视频，再把配音合进去 |
| **Streamlit App** | `streamlit_app.py` | 你实际操作的网页界面：上传图片、上传音频、调参数、生成视频、下载 |
| **Data Models** | `models/data_models.py` | 规定了各模块之间传递数据的格式，像部门之间的"表单模板" |
| **Config** | `config/character_config.py` | 角色设置：分辨率、有哪些表情、眨眼频率等 |

### 关键设计理念

> **AI 不直接选图片。**

AI 只说："这段音频 60% 是 calm，能量 0.7"。
然后 State Engine 说："好，emotion=calm，嘴巴开度=0.8"。
然后 Renderer 说："calm + 嘴开 + 眼开 → 我选 calm_eo_mo 这张图"。

这样做的好处：你以后加 20 种表情，AI 和引擎的代码完全不用改，只需要多上传图片。

---

## 🎛️ Streamlit 网页界面说明

### 侧边栏（左边）
- **角色名称**：随便起，会用在导出文件名上
- **分辨率**：你素材图片的尺寸（宽 × 高）
- **表情状态列表**：每行一个表情名，默认 calm / excited / panic，可以改成任何名字
- **嘴型阈值**：音量超过多少算"张嘴"（0.5 = 中间值，越小越灵敏）
- **眨眼间隔/时长**：多久眨一次、每次眨多长时间
- **动画平滑度**：越小越丝滑但反应慢，越大越灵敏但可能抖
- **情绪后端**：rule = 免费规则判断，openai = 调用 AI 接口（需 API key）

### 四个标签页
1. **📁 上传素材** — 每种表情上传 4 张图，有缩略图预览，底部提示还差几张
2. **🎵 音频输入** — 上传配音文件（wav/mp3等），或输入文字让 TTS 朗读
3. **🎬 生成视频** — 检查清单 + 一键生成（有进度条）
4. **📺 预览&下载** — 在线观看视频 + 音频分析图表 + 下载按钮

---

## 项目结构

```
scaling-umbrella/
├── vtuber_engine/
│   ├── streamlit_app.py        # ← 启动这个！Streamlit 网页界面
│   ├── main.py                 # CLI 入口（备用）
│   ├── audio/
│   │   ├── analyzer.py         # 音频特征提取
│   │   └── emotion_engine.py   # AI 情绪推理
│   ├── core/
│   │   ├── state_engine.py     # 角色状态决策
│   │   └── animation_engine.py # 动画参数插值
│   ├── models/
│   │   └── data_models.py      # 数据结构定义
│   ├── render/
│   │   └── renderer.py         # 图像合成
│   ├── export/
│   │   └── video_exporter.py   # 视频导出
│   └── config/
│       ├── character_config.py # 角色配置加载
│       └── default_config.yaml # 默认参数
├── requirements.txt
└── README.md
```

> 注意：没有 `assets/` 文件夹！素材通过 Streamlit 网页上传，只存在内存中。

---

## 技术栈

| 用途 | 库 |
|------|-----|
| 网页界面 | `streamlit` |
| 音频分析 | `librosa` |
| TTS | `edge-tts` |
| 情绪推理 | 内置规则 / `openai`（可选） |
| 图像合成 | `Pillow` |
| 视频导出 | `ffmpeg` |
| 配置 | `pyyaml` |

---

## 开发路线图

### 阶段 1 — MVP（当前）
- [x] 项目结构 + 完整管线
- [x] Streamlit 可视化界面
- [x] 每表情4图上传（眼×嘴组合）
- [x] 音频特征提取（RMS + VAD）
- [x] 基础情绪识别（rule 后端）
- [x] 嘴型切换 + 眨眼系统
- [x] 动画插值平滑
- [x] 绿幕视频输出
- [x] TTS 接入（Edge TTS）
- [x] 音频/情绪可视化图表

### 阶段 2 — 增强
- [ ] OpenAI 情绪后端
- [ ] 多角色切换
- [ ] 动作层（抬手等）
- [ ] 批量视频生成
- [ ] 时间轴精细编辑

### 阶段 3 — 引擎升级
- [ ] 接入 Unity / Godot
- [ ] Python ↔ 引擎 WebSocket
- [ ] 类 Live2D 动画控制
- [ ] 实时音频驱动

### 阶段 4 — 工具化
- [ ] 可视化调参 GUI
- [ ] 风格预设系统
- [ ] 插件系统

---

## License

Private Project
