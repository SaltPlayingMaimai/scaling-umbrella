"""
弹跳验证脚本 —— 模拟 AI 输出，数学验证 30fps vs 60fps 弹跳行为。

用法:
    python tests/test_bounce_verification.py

验证项:
  1. 帧率一致性：30fps 和 60fps 在相同时间点的弹跳值应接近
  2. 连续性：相邻帧的弹跳变化量不超过阈值（无跳变）
  3. 振幅范围：弹跳值始终在 [0, bounce_amplitude] 内
  4. 衰减正确性：停止说话后弹跳平滑衰减到 0
  5. 灵动模式：burst 周期和 cooldown 计时正确
  6. 弹跳曲线数学性质：周期边界 C¹ 连续（导数=0）

输出: 每项测试 PASS/FAIL + 详细数据
"""

from __future__ import annotations

import math
import os
import sys

# 确保能 import vtuber_engine
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from vtuber_engine.core.animation_engine import AnimationEngine
from vtuber_engine.models.data_models import CharacterState


# ═══════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════


def make_speaking_states(
    duration_sec: float,
    fps: int,
    mouth_open: float = 0.6,
    volume: float = 0.5,
) -> list[CharacterState]:
    """生成持续说话的 CharacterState 序列。"""
    n = int(duration_sec * fps)
    return [
        CharacterState(emotion="calm", energy=0.5, mouth_open=mouth_open)
        for _ in range(n)
    ]


def make_silence_states(duration_sec: float, fps: int) -> list[CharacterState]:
    """生成静音的 CharacterState 序列。"""
    n = int(duration_sec * fps)
    return [
        CharacterState(emotion="calm", energy=0.0, mouth_open=0.0)
        for _ in range(n)
    ]


def make_speech_pattern(fps: int) -> list[CharacterState]:
    """
    模拟真实说话模式：
      0.0-0.5s  静音
      0.5-3.0s  说话
      3.0-3.5s  停顿
      3.5-6.0s  说话
      6.0-8.0s  静音衰减
    """
    states: list[CharacterState] = []
    states += make_silence_states(0.5, fps)
    states += make_speaking_states(2.5, fps, mouth_open=0.6)
    states += make_silence_states(0.5, fps)
    states += make_speaking_states(2.5, fps, mouth_open=0.7)
    states += make_silence_states(2.0, fps)
    return states


def run_engine(
    fps: int,
    states: list[CharacterState],
    lively_mode: bool = False,
    bounce_frequency: float = 1.0,
    bounce_amplitude: float = 30.0,
) -> list[float]:
    """运行 AnimationEngine 并返回 bounce_offset 序列。"""
    engine = AnimationEngine(
        smoothing=0.25,
        fps=fps,
        bounce_enabled=True,
        bounce_frequency=bounce_frequency,
        bounce_amplitude=bounce_amplitude,
        squash_stretch_factor=0.1,
        bounce_lively_mode=lively_mode,
        lively_burst_min=3,
        lively_burst_max=3,  # 固定 3 次以便验证
        lively_cooldown_min=2.0,
        lively_cooldown_max=2.0,  # 固定 2 秒以便验证
    )
    animated = engine.process(states)
    return [a.bounce_offset for a in animated]


def run_engine_full(
    fps: int,
    states: list[CharacterState],
    **kwargs,
) -> list[dict]:
    """运行 AnimationEngine 并返回完整状态序列。"""
    engine = AnimationEngine(
        smoothing=0.25,
        fps=fps,
        bounce_enabled=True,
        bounce_frequency=kwargs.get("bounce_frequency", 1.0),
        bounce_amplitude=kwargs.get("bounce_amplitude", 30.0),
        squash_stretch_factor=0.1,
        bounce_lively_mode=kwargs.get("lively_mode", False),
        lively_burst_min=3,
        lively_burst_max=3,
        lively_cooldown_min=2.0,
        lively_cooldown_max=2.0,
    )
    animated = engine.process(states)
    return [
        {
            "bounce_offset": a.bounce_offset,
            "squash_stretch": a.squash_stretch,
            "mouth_open": a.mouth_open,
        }
        for a in animated
    ]


# ═══════════════════════════════════════════════════════════
#  测试 1：弹跳曲线数学性质
# ═══════════════════════════════════════════════════════════


def test_bounce_curve_math():
    """验证 _bounce_curve 的数学性质。"""
    print("\n" + "=" * 60)
    print("测试 1：弹跳曲线数学性质")
    print("=" * 60)

    errors = []

    # 地面点（t=0）
    h0, v0 = AnimationEngine._bounce_curve(0.0)
    if abs(h0) > 1e-9:
        errors.append(f"  t=0 时 height 应为 0，实际 {h0:.6f}")
    if abs(v0) > 1e-9:
        errors.append(f"  t=0 时 velocity 应为 0，实际 {v0:.6f}")

    # 上升/下降分界点（t=5/7）—— 峰值
    RISE = 5.0 / 7.0
    h_peak, v_peak = AnimationEngine._bounce_curve(RISE - 1e-9)
    if abs(h_peak - 1.0) > 0.01:
        errors.append(f"  峰值 height 应接近 1.0，实际 {h_peak:.6f}")
    if abs(v_peak) > 0.02:
        errors.append(f"  峰值 velocity 应接近 0，实际 {v_peak:.6f}")

    # 接近周期末尾的检查（t → 1.0）
    h_end, v_end = AnimationEngine._bounce_curve(0.9999)
    if abs(h_end) > 0.01:
        errors.append(f"  周期末尾 height 应接近 0，实际 {h_end:.6f}")
    if abs(v_end) > 0.02:
        errors.append(f"  周期末尾 velocity 应接近 0，实际 {v_end:.6f}")

    # 上升段 height 单调递增
    prev_h = 0.0
    for i in range(1, 100):
        t = (i / 100) * RISE
        h, _ = AnimationEngine._bounce_curve(t)
        if h < prev_h - 1e-6:
            errors.append(f"  上升段 t={t:.3f} 处 height 非单调递增: {prev_h:.6f} → {h:.6f}")
            break
        prev_h = h

    # 下降段 height 单调递减
    prev_h = 1.0
    for i in range(1, 100):
        t = RISE + (i / 100) * (1.0 - RISE)
        h, _ = AnimationEngine._bounce_curve(t)
        if h > prev_h + 1e-6:
            errors.append(f"  下降段 t={t:.3f} 处 height 非单调递减: {prev_h:.6f} → {h:.6f}")
            break
        prev_h = h

    # C¹ 连续性检查：数值导数在分界点处左右极限应接近
    eps = 1e-6
    h_left, _ = AnimationEngine._bounce_curve(RISE - eps)
    h_right, _ = AnimationEngine._bounce_curve(RISE + eps)
    h_center, _ = AnimationEngine._bounce_curve(RISE)
    deriv_left = (h_center - h_left) / eps
    deriv_right = (h_right - h_center) / eps
    if abs(deriv_left - deriv_right) > 0.1:
        errors.append(
            f"  分界点 C¹ 不连续: 左导数={deriv_left:.4f}, 右导数={deriv_right:.4f}"
        )

    if errors:
        print("FAIL:")
        for e in errors:
            print(e)
    else:
        print("PASS: 弹跳曲线满足所有数学性质")
        print(f"  [OK] 地面点: h={h0:.6f}, v={v0:.6f}")
        print(f"  [OK] 峰值点: h={h_peak:.6f}, v={v_peak:.6f}")
        print(f"  [OK] 上升段单调递增, 下降段单调递减")
        print(f"  [OK] C1 连续: 左导数={deriv_left:.4f}, 右导数={deriv_right:.4f}")

    return len(errors) == 0


# ═══════════════════════════════════════════════════════════
#  测试 2：帧率一致性（30fps vs 60fps 弹跳对齐）
# ═══════════════════════════════════════════════════════════


def test_fps_consistency():
    """验证 30fps 和 60fps 在相同时间点的弹跳值接近。"""
    print("\n" + "=" * 60)
    print("测试 2：帧率一致性（30fps vs 60fps）")
    print("=" * 60)

    # 持续说话 5 秒
    states_30 = make_speaking_states(5.0, 30)
    states_60 = make_speaking_states(5.0, 60)

    bounces_30 = run_engine(30, states_30, bounce_frequency=1.0)
    bounces_60 = run_engine(60, states_60, bounce_frequency=1.0)

    errors = []
    max_diff = 0.0
    diff_samples = []

    # 比较相同时间点（每 1/30 秒，即 30fps 的每帧 = 60fps 的每 2 帧）
    for i in range(len(bounces_30)):
        t = i / 30.0
        j = i * 2  # 60fps 中对应的帧
        if j >= len(bounces_60):
            break
        b30 = bounces_30[i]
        b60 = bounces_60[j]
        diff = abs(b30 - b60)
        max_diff = max(max_diff, diff)
        diff_samples.append((t, b30, b60, diff))

    # 允许的最大差异（像素）
    # 由于 lerp 离散化在不同采样率下的固有误差，允许一定容差
    # 30fps 采样 1/30s 间隔 vs 60fps 采样 1/60s 间隔，lerp 追逐路径略有不同
    TOLERANCE = 3.5  # 像素

    # 稳态差异（跳过前 30 帧的暖启动期）
    steady_diffs = [d[3] for d in diff_samples[30:]]
    avg_steady_diff = sum(steady_diffs) / max(len(steady_diffs), 1)
    max_steady_diff = max(steady_diffs) if steady_diffs else 0

    if max_steady_diff > TOLERANCE:
        errors.append(
            f"  稳态最大差异 {max_steady_diff:.2f}px > 容差 {TOLERANCE}px"
        )

    # 打印采样对比
    print(f"  时间点采样（每 0.5 秒）:")
    for t, b30, b60, diff in diff_samples:
        if abs(t % 0.5) < 0.02 or t < 0.1:
            print(f"    t={t:5.2f}s  30fps={b30:7.2f}  60fps={b60:7.2f}  diff={diff:.2f}px")

    print(f"\n  稳态平均差异: {avg_steady_diff:.2f}px")
    print(f"  稳态最大差异: {max_steady_diff:.2f}px")
    print(f"  全程最大差异: {max_diff:.2f}px")

    if errors:
        print("FAIL:")
        for e in errors:
            print(e)
    else:
        print(f"PASS: 30fps/60fps 弹跳差异在 {TOLERANCE}px 容差内")

    return len(errors) == 0


# ═══════════════════════════════════════════════════════════
#  测试 3：连续性（无跳变）
# ═══════════════════════════════════════════════════════════


def test_continuity():
    """验证弹跳曲线无帧间跳变。"""
    print("\n" + "=" * 60)
    print("测试 3：连续性（无帧间跳变）")
    print("=" * 60)

    errors = []

    for fps in [30, 60]:
        states = make_speech_pattern(fps)
        bounces = run_engine(fps, states, bounce_frequency=1.0, bounce_amplitude=30.0)

        max_delta = 0.0
        max_delta_frame = 0

        for i in range(1, len(bounces)):
            delta = abs(bounces[i] - bounces[i - 1])
            if delta > max_delta:
                max_delta = delta
                max_delta_frame = i

        # 最大允许帧间变化：
        # 在 30fps 下，振幅 30px，1Hz 频率，
        # 最快变化在下降段（2/7 周期 ≈ 0.286s ≈ 8.6 帧跨越 30px）
        # max_delta ≈ 30 / 8.6 ≈ 3.5 px/frame @30fps
        # 加上 lerp 平滑实际应更小，但留足余量
        MAX_DELTA = 8.0  # px per frame（宽松阈值）

        if max_delta > MAX_DELTA:
            errors.append(
                f"  {fps}fps: 帧 {max_delta_frame} 处跳变 {max_delta:.2f}px > {MAX_DELTA}px"
            )

        print(f"  {fps}fps: 最大帧间变化 = {max_delta:.2f}px (帧 {max_delta_frame})")

        # 打印极端变化点附近的上下文
        if max_delta > 2.0:
            start = max(0, max_delta_frame - 3)
            end = min(len(bounces), max_delta_frame + 4)
            print(f"    上下文 (帧 {start}~{end-1}):")
            for j in range(start, end):
                marker = " <--" if j == max_delta_frame else ""
                t = j / fps
                print(f"      [{j:4d}] t={t:5.3f}s bounce={bounces[j]:7.2f}{marker}")

    if errors:
        print("FAIL:")
        for e in errors:
            print(e)
    else:
        print("PASS: 所有帧率下弹跳无跳变")

    return len(errors) == 0


# ═══════════════════════════════════════════════════════════
#  测试 4：振幅范围
# ═══════════════════════════════════════════════════════════


def test_amplitude_range():
    """验证弹跳值始终在合理范围内。"""
    print("\n" + "=" * 60)
    print("测试 4：振幅范围")
    print("=" * 60)

    errors = []
    AMPLITUDE = 30.0

    for fps in [30, 60]:
        states = make_speaking_states(5.0, fps, mouth_open=0.8)
        bounces = run_engine(fps, states, bounce_amplitude=AMPLITUDE)

        bmin = min(bounces)
        bmax = max(bounces)

        # 弹跳应在 [负小值, AMPLITUDE] 范围内
        # 由于 lerp 追逐，可能短暂超调但不应超出 AMPLITUDE * 1.1
        if bmin < -AMPLITUDE * 0.1:
            errors.append(f"  {fps}fps: 最小值 {bmin:.2f} < -{AMPLITUDE * 0.1:.1f}")
        if bmax > AMPLITUDE * 1.1:
            errors.append(f"  {fps}fps: 最大值 {bmax:.2f} > {AMPLITUDE * 1.1:.1f}")

        print(f"  {fps}fps: 范围 [{bmin:.2f}, {bmax:.2f}] (目标幅度 {AMPLITUDE}px)")

    if errors:
        print("FAIL:")
        for e in errors:
            print(e)
    else:
        print("PASS: 弹跳值在合理范围内")

    return len(errors) == 0


# ═══════════════════════════════════════════════════════════
#  测试 5：衰减正确性
# ═══════════════════════════════════════════════════════════


def test_decay_correctness():
    """验证停止说话后弹跳平滑衰减到 0，且 30/60fps 衰减时间一致。"""
    print("\n" + "=" * 60)
    print("测试 5：衰减正确性")
    print("=" * 60)

    errors = []

    # 说话 2 秒 + 静音 3 秒
    for fps in [30, 60]:
        states = make_speaking_states(2.0, fps) + make_silence_states(3.0, fps)
        results = run_engine_full(fps, states)
        bounces = [r["bounce_offset"] for r in results]
        mouths = [r["mouth_open"] for r in results]

        speech_end_frame = int(2.0 * fps)

        # 找到 mouth_open 实际衰减到 < 0.05 的帧（bounce 只在此之后进入 decay 分支）
        actual_decay_start = None
        for i in range(speech_end_frame, len(mouths)):
            if mouths[i] < 0.05:
                actual_decay_start = i
                break

        if actual_decay_start is None:
            errors.append(f"  {fps}fps: mouth_open 未衰减到 0.05 以下")
            continue

        print(f"  {fps}fps: mouth_open 在帧 {actual_decay_start} ({actual_decay_start/fps:.3f}s) 降到 <0.05")

        # 找到弹跳归零的帧
        zero_frame = None
        for i in range(actual_decay_start, len(bounces)):
            if bounces[i] == 0.0:
                zero_frame = i
                break

        if zero_frame is None:
            errors.append(f"  {fps}fps: 弹跳未归零")
            zero_time = None
        else:
            zero_time = (zero_frame - actual_decay_start) / fps
            print(f"  {fps}fps: 从 decay 开始到归零耗时 {zero_time:.3f}s (帧 {zero_frame - actual_decay_start})")

            # 验证 decay 期间单调递减（从 actual_decay_start 开始，不含 mouth_open 平滑期）
            for i in range(actual_decay_start + 1, zero_frame):
                if bounces[i] > bounces[i - 1] + 0.01:  # 小容差
                    errors.append(
                        f"  {fps}fps: 衰减期帧 {i} 非单调: {bounces[i-1]:.2f} → {bounces[i]:.2f}"
                    )
                    break

    # 比较 30fps 和 60fps 的衰减时间
    times = {}
    for fps in [30, 60]:
        states = make_speaking_states(2.0, fps) + make_silence_states(3.0, fps)
        results = run_engine_full(fps, states)
        bounces = [r["bounce_offset"] for r in results]
        mouths = [r["mouth_open"] for r in results]
        # 找到 mouth_open 实际低于 0.05 的帧
        decay_start = None
        for i in range(int(2.0 * fps), len(mouths)):
            if mouths[i] < 0.05:
                decay_start = i
                break
        if decay_start is not None:
            for i in range(decay_start, len(bounces)):
                if bounces[i] == 0.0:
                    times[fps] = (i - decay_start) / fps
                    break

    if 30 in times and 60 in times:
        diff = abs(times[30] - times[60])
        print(f"  衰减时间差: |{times[30]:.3f} - {times[60]:.3f}| = {diff:.3f}s")
        if diff > 0.2:
            errors.append(f"  衰减时间差 {diff:.3f}s > 0.2s（帧率不一致）")

    if errors:
        print("FAIL:")
        for e in errors:
            print(e)
    else:
        print("PASS: 衰减正确且帧率一致")

    return len(errors) == 0


# ═══════════════════════════════════════════════════════════
#  测试 6：灵动模式行为
# ═══════════════════════════════════════════════════════════


def test_lively_mode():
    """验证灵动模式的 burst + cooldown 行为。"""
    print("\n" + "=" * 60)
    print("测试 6：灵动模式行为")
    print("=" * 60)

    errors = []

    # 持续说话 10 秒，burst=3次，cooldown=2秒，frequency=1Hz
    # 预期行为：
    #   0~3s  第一轮 burst（3 次跳）
    #   3~5s  cooldown（衰减到待机）
    #   5~8s  第二轮 burst（3 次跳）
    #   8~10s cooldown（衰减到待机）
    for fps in [30, 60]:
        states = make_speaking_states(10.0, fps, mouth_open=0.6)
        bounces = run_engine(fps, states, lively_mode=True, bounce_frequency=1.0)

        # 分析弹跳活动区间
        # "活动" = bounce > 1.0 的区间
        active_regions = []
        in_active = False
        region_start = 0
        for i, b in enumerate(bounces):
            if b > 1.0 and not in_active:
                in_active = True
                region_start = i
            elif b <= 1.0 and in_active:
                in_active = False
                active_regions.append((region_start / fps, i / fps))
        if in_active:
            active_regions.append((region_start / fps, len(bounces) / fps))

        print(f"  {fps}fps: 弹跳活动区间:")
        for start, end in active_regions:
            duration = end - start
            print(f"    {start:.2f}s ~ {end:.2f}s (持续 {duration:.2f}s)")

        # 应有至少 2 个活动区间（2 轮 burst）
        if len(active_regions) < 2:
            errors.append(f"  {fps}fps: 仅检测到 {len(active_regions)} 个活动区间，预期 ≥ 2")
        else:
            # 第 1 和第 2 个 burst 之间应有约 2 秒的 cooldown
            gap = active_regions[1][0] - active_regions[0][1]
            if gap < 1.0:
                errors.append(
                    f"  {fps}fps: burst 间隔 {gap:.2f}s < 1.0s（cooldown 不足）"
                )
            print(f"    burst 间 cooldown: {gap:.2f}s (目标 ~2.0s)")

    if errors:
        print("FAIL:")
        for e in errors:
            print(e)
    else:
        print("PASS: 灵动模式 burst/cooldown 行为正确")

    return len(errors) == 0


# ═══════════════════════════════════════════════════════════
#  测试 7：完整管线模拟（说话模式）
# ═══════════════════════════════════════════════════════════


def test_full_pipeline_simulation():
    """模拟真实说话模式下的完整弹跳行为。"""
    print("\n" + "=" * 60)
    print("测试 7：完整管线模拟")
    print("=" * 60)

    errors = []

    for fps in [30, 60]:
        states = make_speech_pattern(fps)
        results = run_engine_full(fps, states, bounce_frequency=1.0)

        total = len(results)
        bounces = [r["bounce_offset"] for r in results]
        mouths = [r["mouth_open"] for r in results]

        # 统计
        active_frames = sum(1 for b in bounces if abs(b) > 0.5)
        max_bounce = max(bounces)
        min_bounce = min(bounces)
        peak_frame = bounces.index(max_bounce)
        peak_time = peak_frame / fps

        print(f"\n  {fps}fps ({total} 帧, {total / fps:.1f}s):")
        print(f"    活跃帧: {active_frames}/{total} ({active_frames/total*100:.1f}%)")
        print(f"    弹跳范围: [{min_bounce:.2f}, {max_bounce:.2f}]")
        print(f"    峰值时刻: {peak_time:.2f}s (帧 {peak_frame})")

        # 打印时间线
        print(f"    时间线:")
        for sec in range(int(total / fps) + 1):
            frame = sec * fps
            if frame < total:
                b = bounces[frame]
                m = mouths[frame]
                bar = "#" * int(abs(b))
                print(f"      {sec:2d}s: bounce={b:7.2f} mouth={m:.2f} |{bar}")

        # 验证：静音期弹跳应为 0
        silence_start = int(6.0 * fps)
        silence_end = min(int(8.0 * fps), total)
        if silence_end <= silence_start:
            continue
        late_silence = bounces[silence_end - 1] if silence_end - 1 < total else 0
        if abs(late_silence) > 0.5:
            errors.append(
                f"  {fps}fps: 静音末尾弹跳 {late_silence:.2f} 未归零"
            )

    if errors:
        print("\nFAIL:")
        for e in errors:
            print(e)
    else:
        print("\nPASS: 完整管线模拟正确")

    return len(errors) == 0


# ═══════════════════════════════════════════════════════════
#  测试 8：_adjust_speed / _adjust_decay 数学验证
# ═══════════════════════════════════════════════════════════


def test_fps_adjustment_math():
    """验证帧率校正函数的数学正确性。"""
    print("\n" + "=" * 60)
    print("测试 8：帧率校正函数数学验证")
    print("=" * 60)

    errors = []

    # 创建不同帧率的引擎
    engines = {fps: AnimationEngine(fps=fps) for fps in [24, 30, 60, 120]}

    # 测试 _adjust_speed: 给定初始值 100，经过 1 秒后剩余值应相同
    base_speed = 0.35
    initial = 100.0
    TARGET_TIME = 1.0

    print(f"  lerp speed={base_speed} @30fps设计, 初始值={initial}, 经过 {TARGET_TIME}s:")
    remainders = {}
    for fps, eng in engines.items():
        adj_speed = eng._adjust_speed(base_speed)
        val = initial
        for _ in range(int(fps * TARGET_TIME)):
            val = val * (1.0 - adj_speed)
        remainders[fps] = val
        print(f"    {fps:3d}fps: adjusted_speed={adj_speed:.6f}, 剩余={val:.6f}")

    # 所有帧率的剩余值应接近
    ref = remainders[30]
    for fps, rem in remainders.items():
        diff = abs(rem - ref)
        if diff > 0.1:
            errors.append(
                f"  {fps}fps 剩余 {rem:.6f} vs 30fps {ref:.6f}, 差异 {diff:.6f} > 0.1"
            )

    # 测试 _adjust_decay: 给定初始值 100，经过 1 秒后剩余值应相同
    base_decay = 0.85
    print(f"\n  decay factor={base_decay} @30fps设计, 初始值={initial}, 经过 {TARGET_TIME}s:")
    remainders2 = {}
    for fps, eng in engines.items():
        adj_decay = eng._adjust_decay(base_decay)
        val = initial
        for _ in range(int(fps * TARGET_TIME)):
            val *= adj_decay
        remainders2[fps] = val
        print(f"    {fps:3d}fps: adjusted_decay={adj_decay:.6f}, 剩余={val:.6f}")

    ref2 = remainders2[30]
    for fps, rem in remainders2.items():
        diff = abs(rem - ref2)
        if diff > 0.1:
            errors.append(
                f"  decay {fps}fps 剩余 {rem:.6f} vs 30fps {ref2:.6f}, 差异 {diff:.6f} > 0.1"
            )

    if errors:
        print("FAIL:")
        for e in errors:
            print(e)
    else:
        print("PASS: 帧率校正函数数学正确")

    return len(errors) == 0


# ═══════════════════════════════════════════════════════════
#  测试 9：重复帧检测（视觉卡顿诊断）
# ═══════════════════════════════════════════════════════════


def test_repeated_frames():
    """检测活跃说话期间是否有连续相同的弹跳值（导致视觉卡顿）。"""
    print("\n" + "=" * 60)
    print("测试 9：重复帧检测（卡顿诊断）")
    print("=" * 60)

    errors = []

    for fps in [30, 60]:
        states = make_speaking_states(5.0, fps, mouth_open=0.6)
        bounces = run_engine(fps, states, bounce_frequency=1.0)

        # 跳过前 10 帧（初始加速期）
        active_bounces = bounces[10:]

        # 检查整数四舍五入后的重复帧
        int_bounces = [int(round(b)) for b in active_bounces]
        repeat_count = 0
        max_repeat = 0
        current_repeat = 0
        for i in range(1, len(int_bounces)):
            if int_bounces[i] == int_bounces[i - 1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
                repeat_count += 1
            else:
                current_repeat = 0

        repeat_pct = repeat_count / max(len(int_bounces) - 1, 1) * 100

        print(f"  {fps}fps: 重复帧 {repeat_count}/{len(int_bounces)-1} ({repeat_pct:.1f}%)")
        print(f"    最长连续重复: {max_repeat} 帧")

        # 60fps 下由于精度更高，允许更多短暂重复（在波峰/谷值附近）
        # 但不应有超过 5 帧的连续重复
        MAX_CONSECUTIVE = 5
        if max_repeat > MAX_CONSECUTIVE:
            errors.append(
                f"  {fps}fps: 最长连续重复 {max_repeat} 帧 > {MAX_CONSECUTIVE}"
            )

    if errors:
        print("FAIL:")
        for e in errors:
            print(e)
    else:
        print("PASS: 无明显卡顿风险")

    return len(errors) == 0


# ═══════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   VTuber Engine — 弹跳验证套件                          ║")
    print("║   模拟 AI 输出 → 数学验证 30fps/60fps 弹跳行为          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    tests = [
        ("弹跳曲线数学性质", test_bounce_curve_math),
        ("帧率一致性", test_fps_consistency),
        ("连续性", test_continuity),
        ("振幅范围", test_amplitude_range),
        ("衰减正确性", test_decay_correctness),
        ("灵动模式", test_lively_mode),
        ("完整管线模拟", test_full_pipeline_simulation),
        ("帧率校正数学", test_fps_adjustment_math),
        ("重复帧检测", test_repeated_frames),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 汇总
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        icon = "[OK]" if passed else "[NG]"
        print(f"  {icon} {status}: {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("所有测试通过！弹跳在各帧率下行为一致且正确。")
    else:
        print("存在失败项，请根据上方详细输出排查。")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
