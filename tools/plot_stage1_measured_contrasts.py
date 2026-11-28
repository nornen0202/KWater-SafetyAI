"""Render measured Fold 2 BF16 curriculum and local NF4 resolution contrasts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

from plot_stage1_fold2_matched import BLUE, DPI, GREEN, INK, MUTED, PAPER, ROOT, RUST, signed


MIXED = ROOT / "results/fold2-matched-4b-8b-epoch1-20261128.json"
SEOUL = ROOT / "results/stage1-pure-vs-mixed-fold2-20261128.json"
LOCAL = ROOT / "results/stage1-local-nf4-resolution-55-20261129.json"
OUTPUT = ROOT / "reports/figures/stage1-measured-contrasts-20261129.png"


@dataclass(frozen=True)
class Comparison:
    delta: float
    lo: float
    hi: float
    errors: tuple[tuple[int, int, int], tuple[int, int, int]]


def verified_panel(panel, arms, photos, groups, normal_n, severe_n):
    first, second = (panel["metrics"][arm] for arm in arms)
    active = panel["paired_group_bootstrap"]["active_per_arm"]
    delta = active["macro_f1_delta_b_minus_a"]
    lo, hi = active["percentile_95_ci"]
    if ((panel["photos"], panel["manifest_groups"], active["groups"], active["resamples"])
            != (photos, groups, groups, 2000)
            or not all(math.isfinite(value) for value in (lo, delta, hi)) or not -1 <= lo <= hi <= 1
            or not math.isclose(delta, second["macro_f1"] - first["macro_f1"], rel_tol=1e-12, abs_tol=1e-12)
            or any(arm["no_hazard_photos"] != normal_n or arm["severity4"]["photos"] != severe_n
                   for arm in (first, second))):
        raise ValueError(f"Paired bootstrap or cohort differs for the fixed {photos}-photo panel")
    errors = ((first["no_hazard_false_alarms"], second["no_hazard_false_alarms"], normal_n),
              (first["severity4"]["complete_misses"], second["severity4"]["complete_misses"], severe_n))
    if any(not (0 <= before <= total and 0 <= after <= total) for before, after, total in errors):
        raise ValueError("Error counts exceed the fixed denominators")
    return Comparison(delta, lo, hi, errors)


def load_comparisons(seoul_path, local_path):
    mixed_data = MIXED.read_bytes()
    mixed = json.loads(mixed_data)
    seoul = json.loads(seoul_path.read_bytes())
    local = json.loads(local_path.read_bytes())
    control = mixed["generated"]
    candidate = seoul["pure_inference"]
    controls = local["controls"]
    frozen = mixed["recipe"]["training"]
    revision = mixed["recipe"]["base_revision"]
    cache_key = f"fold2-8b:{revision}:{candidate['adapter_sha256']}:{frozen['prompt_sha256']}:bf16:px393216"
    if (seoul["mixed_score_sha256"] != sha256(mixed_data).hexdigest()
            or seoul["mixed_predictions_sha256"] != mixed["8b_predictions_sha256"]
            or seoul["same_schedule_examples_and_applied_lr"] is not True
            or candidate["adapter_sha256"] == control["adapter_sha256"]
            or candidate["eval_order_sha256"] != control["eval_order_sha256"]
            or candidate["cache_key"] != cache_key
            or [(s["start"], s["end"], s["stats"]["initial_batch_size"]) for s in candidate["segments"]]
                != [(0, 44, 4), (44, 197, 2)]
            or seoul["mask_sha256"] != frozen["mask_sha256"]
            or controls["mask_sha256"] != frozen["mask_sha256"]
            or controls["base_revision"] != revision or controls["prompt_sha256"] != frozen["prompt_sha256"]
            or controls["adapter_sha256"] != control["adapter_sha256"]
            or controls["precision"] != "nf4_double_quant_bf16_compute" or controls["photos"] != 55
            or local["caps"] != [393216, 655360] or local["identical_nonpixel_cache_identity"] is not True):
        raise ValueError("Expected the measured Seoul BF16 and RTX 3060 NF4 Fold 2 contrasts")
    panels = seoul["panels"]
    comparisons = (verified_panel(panels["primary_197"], ("control", "candidate"), 197, 38, 67, 59),
                   verified_panel(panels["sensitivity_55"], ("control", "candidate"), 55, 34, 12, 15),
                   verified_panel(local["quality"], ("393216", "655360"), 55, 34, 12, 15))
    low, high = (local["measurement"][str(cap)]["stats"] for cap in local["caps"])
    if (any(arm["photos"] != 55 or arm["initial_batch_size"] != 1 or arm["final_effective_batch_size"] != 1
            or not (0 <= arm["strict_vlm_schema_success"] <= 55 and 0 <= arm["token_cap_hits"] <= 55)
            for arm in (low, high)) or local["cap_failure_crosscheck"]["any_token_cap_hit_photos"] != high["token_cap_hits"]
            or low["token_cap_hits"] != 0):
        raise ValueError("Expected complete batch-1 NF4 raw generation counts")
    raw = tuple((arm["strict_vlm_schema_success"], arm["token_cap_hits"]) for arm in (low, high))
    return comparisons, raw


def draw(comparisons, raw, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    fonts = Path("C:/Windows/Fonts")
    for path in (fonts / "malgun.ttf", fonts / "malgunbd.ttf"):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Malgun Gothic", "Noto Sans CJK KR", "DejaVu Sans"],
        "font.size": 11, "axes.unicode_minus": False, "text.color": INK, "axes.labelcolor": INK,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "xtick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (forest, errors) = plt.subplots(1, 2, figsize=(12.3, 5.55), sharey=True,
                                        gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.251, right=.972, bottom=.233, top=.748, wspace=.17)
    limit = max(.03, 1.13 * max(abs(value) for panel in comparisons for value in (panel.lo, panel.delta, panel.hi)))
    forest.set(xlim=(-limit, limit), ylim=(-.46, 4.03), xlabel="활성 Macro-F1 차이 (변경 - 기준)")
    forest.set_title("쌍체 점 추정치와 95% 구간", fontsize=12.5, loc="left", weight="bold", pad=13)
    forest.axvline(0, color="#647d72", lw=1.05)
    forest.grid(axis="x", color="#dce7e1", lw=.65)
    forest.set_axisbelow(True)
    forest.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2.5, 5, 10], symmetric=True))
    forest.xaxis.set_major_formatter(FuncFormatter(lambda value, _: signed(value, 2) if abs(value) > 1e-10 else "0"))
    forest.spines["left"].set_visible(False)
    forest.spines["bottom"].set_color("#9aac9f")
    forest.tick_params(axis="y", left=False, labelleft=False)
    errors.set_xlim(0, 1)
    errors.set_title("사진 수 기준 오류", fontsize=12.5, loc="left", weight="bold", pad=13)
    errors.set_axis_off()
    for x, label in ((.25, "정상 오경보"), (.75, "심각도 4 완전 미탐")):
        errors.text(x, 3.82, label, ha="center", va="center", size=10.6, weight="bold")

    y_transform = forest.get_yaxis_transform()
    for y, heading, subtitle, color in (
        (3.73, "서울 BF16 · 커리큘럼", "혼합 v3 → Pure-Inspect v3", GREEN),
        (.98, "로컬 NF4 · 해상도", "393,216 → 655,360픽셀", BLUE),
    ):
        forest.text(-.60, y, heading, transform=y_transform, size=12, weight="bold", color=color, va="center")
        forest.text(-.60, y - .31, subtitle, transform=y_transform, size=9.7, color=MUTED, va="center")

    for axis in (forest, errors):
        axis.axhline(1.16, color="#c8d8cf", linewidth=.8)

    for panel, (y, label, color) in zip(comparisons, (
        (3.05, "전체 197장 · 38그룹", GREEN), (1.98, "민감도 55장 · 34그룹", GREEN),
        (0, "민감도 55장 · 34그룹", BLUE),
    ), strict=True):
        forest.text(-.60, y, label, transform=y_transform, size=10.4, va="center")
        forest.hlines(y, panel.lo, panel.hi, color=color, linewidth=2.25)
        forest.vlines((panel.lo, panel.hi), y - .075, y + .075, color=color, linewidth=1.6)
        forest.plot(panel.delta, y, "o", ms=7.5, color=color)
        forest.text(0, y + .21, f"{signed(panel.delta, 4)} [{signed(panel.lo, 4)}, {signed(panel.hi, 4)}]",
                    ha="center", va="bottom", color=color, size=10.4, weight="bold",
                    bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 2.5})
        errors.axhspan(y - .36, y + .37, color="#e7f0ea" if color == GREEN else "#e6eff3")
        for x, (before, after, total) in zip((.25, .75), panel.errors, strict=True):
            delta_n = after - before
            status_color = GREEN if delta_n < 0 else RUST if delta_n > 0 else MUTED
            errors.text(x, y + .115, f"{before}/{total} → {after}/{total}", ha="center", va="center", size=12)
            errors.text(x, y - .17, f"{signed(delta_n, 0) if delta_n else '0'}장", color=status_color,
                        ha="center", va="center", size=10.3, weight="bold")

    fig.text(.041, .921, "Stage 1 · Fold 2 쌍체 판독 결과", size=20, weight="bold")
    fig.text(.041, .856, "각 비교의 기준과 변경 설정에서 같은 사진을 짝지어 계산했다.", size=11.1, color=MUTED)
    fig.text(.041, .123, "95% 구간은 현장 그룹 쌍체 재표집 2,000회 기준이다. 전체 197장에는 교차 현장이 있고 민감도 55장에는 이전 ROI 검토 사진 16장이 남는다.",
             size=9.5, color=MUTED)
    fig.text(.041, .067,
             f"로컬 NF4 원생성 · 토큰 상한 도달 {raw[0][1]} → {raw[1][1]}장 · 엄격 스키마 통과 {raw[0][0]}/55 → {raw[1][0]}/55장. 채점은 55쌍 모두 포함.",
             size=9.5, color=MUTED)
    fig.savefig(output, dpi=DPI, metadata={"Software": "tools/plot_stage1_measured_contrasts.py"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seoul", type=Path, default=SEOUL)
    parser.add_argument("--local", type=Path, default=LOCAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        comparisons, raw = load_comparisons(args.seoul, args.local)
    except FileNotFoundError as error:
        parser.error(f"Measured aggregate required at {error.filename}")
    draw(comparisons, raw, args.output)


if __name__ == "__main__":
    main()
