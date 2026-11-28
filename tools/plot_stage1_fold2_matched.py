"""Draw the measured Fold 2 first-epoch 4B/8B comparison from its aggregate receipt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "results/fold2-matched-4b-8b-epoch1-20261128.json"
CONTROL = ROOT / "results/fold2-v3-4b-epoch1-20261128.json"
OUTPUT = ROOT / "reports/figures/stage1-fold2-matched-4b-8b-20261128.png"
INK, MUTED = "#1b3831", "#49635a"
GREEN, BLUE, RUST = "#216653", "#3979a4", "#aa623d"
PAPER = "#f5f8f5"
DPI = 190


def signed(value, decimals):
    return f"{value:+.{decimals}f}"


def load_panels(path):
    receipt = json.loads(path.read_bytes())
    control = json.loads(CONTROL.read_bytes())
    generation = receipt["generated"]
    if (receipt["ordered_training_step_controls_match"] is not True
            or receipt["4b_predictions_sha256"] != control["inference"]["prediction_sha256"]
            or not generation["cache_key"].endswith(":bf16:px393216")
            or [(s["start"], s["end"]) for s in generation["segments"]] != [(0, 44), (44, 197)]):
        raise ValueError("Expected the measured matched Fold 2 first-epoch BF16 receipt")

    panels = []
    for key, count, groups, normal_n, severe_n in (
        ("primary_197", 197, 38, 67, 59), ("sensitivity_55", 55, 34, 12, 15),
    ):
        panel = receipt["panels"][key]
        bootstrap = panel["paired_group_bootstrap"]
        active = bootstrap["active_per_arm"]
        first, second = (panel["metrics"][model] for model in ("4b", "8b"))
        delta = active["macro_f1_delta_b_minus_a"]
        lo, hi = active["percentile_95_ci"]
        first_f1, second_f1 = first["macro_f1"], second["macro_f1"]
        if ((panel["photos"], panel["manifest_groups"], active["groups"], bootstrap["resamples"], active["resamples"])
                != (count, groups, groups, 2000, 2000)
                or not math.isclose(delta, second_f1 - first_f1, rel_tol=1e-12, abs_tol=1e-12)
                or not all(math.isfinite(x) for x in (delta, lo, hi)) or lo > hi
                or any(arm["no_hazard_photos"] != normal_n or arm["severity4"]["photos"] != severe_n for arm in (first, second))):
            raise ValueError(f"Measured counts or paired bootstrap differ from the fixed {count}-photo panel")
        errors = (
            (first["no_hazard_false_alarms"], second["no_hazard_false_alarms"], normal_n),
            (first["severity4"]["complete_misses"], second["severity4"]["complete_misses"], severe_n),
        )
        if any(not (0 <= before <= total and 0 <= after <= total) for before, after, total in errors):
            raise ValueError("Error counts exceed fixed denominators")
        panels.append((count, groups, first_f1, second_f1, delta, lo, hi, errors))
    return panels


def draw(panels, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    malgun = Path("C:/Windows/Fonts/malgun.ttf")
    if malgun.exists():
        for path in (malgun, malgun.with_name("malgunbd.ttf")):
            if path.is_file():
                font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Malgun Gothic", "Noto Sans CJK KR", "DejaVu Sans"],
        "font.size": 11, "axes.unicode_minus": False, "text.color": INK, "axes.labelcolor": INK,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (forest, errors) = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True,
                                        gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=.213, right=.967, bottom=.255, top=.724, wspace=.24)
    extent = max(abs(v) for panel in panels for v in panel[4:7])
    limit = max(.03, extent * 1.12)
    forest.set(xlim=(-limit, limit), ylim=(-.55, 1.58), xlabel="활성 Macro-F1 차이  (8B - 4B)")
    forest.set_title("위험 유형 탐지", fontsize=13, loc="left", weight="bold", pad=14)
    forest.axvline(0, color="#627b70", lw=1.05)
    forest.grid(axis="x", color="#dce7e1", lw=.65)
    forest.set_axisbelow(True)
    forest.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2.5, 5, 10], symmetric=True))
    forest.xaxis.set_major_formatter(FuncFormatter(lambda value, _: signed(value, 2) if abs(value) > 1e-10 else "0"))
    forest.spines["left"].set_visible(False)
    forest.spines["bottom"].set_color("#9aac9f")
    forest.tick_params(axis="y", length=0, pad=10)

    errors.set_xlim(0, 1)
    errors.set_title("사진 수 기준 오류", fontsize=13, loc="left", weight="bold", pad=14)
    errors.set_axis_off()
    for x, header in ((.25, "정상 오경보"), (.75, "심각도 4 완전 미탐")):
        errors.text(x, 1.44, header, ha="center", va="center", size=10.5, weight="bold")

    labels = []
    for y, panel, heading, color in zip((1, 0), panels, ("전체", "민감도"), (GREEN, BLUE), strict=True):
        count, groups, first_f1, second_f1, delta, lo, hi, counts = panel
        labels.append(f"{heading} {count}장 · {groups}그룹\n{first_f1:.3f} → {second_f1:.3f}")
        forest.hlines(y, lo, hi, color=color, linewidth=2.2)
        forest.vlines((lo, hi), y - .047, y + .047, color=color, linewidth=1.6)
        forest.plot(delta, y, "o", ms=7.5, color=color)
        forest.text(0, y + .20, f"{signed(delta, 4)} [{signed(lo, 4)}, {signed(hi, 4)}]",
                    ha="center", va="bottom", color=color, size=10.5, weight="bold",
                    bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 3})
        errors.axhspan(y - .34, y + .26, color="#e7f0ea")
        for x, (before, after, total) in zip((.25, .75), counts, strict=True):
            diff = after - before
            status_color = GREEN if diff < 0 else RUST if diff > 0 else MUTED
            errors.text(x, y + .08, f"{before}/{total} → {after}/{total}", ha="center", va="center", size=12)
            errors.text(x, y - .18, f"{signed(diff, 0) if diff else '0'}장", color=status_color,
                        ha="center", va="center", size=10.5, weight="bold")
    forest.set_yticks([1, 0], labels)
    fig.text(.046, .904, "Fold 2 · 4B와 8B 첫 에폭 비교", size=19, weight="bold")
    fig.text(.046, .826, "같은 v3 혼합 학습 예제와 393,216픽셀 BF16 판독 · 4B Windows RTX 3060 / 8B Linux L4",
             size=11.2, color=MUTED)
    fig.text(.046, .122, "점은 쌍체 점 추정치, 가로선은 현장 그룹 재표집 2,000회로 계산한 95% 구간이다.", size=10)
    fig.text(.046, .065, "197장에는 확인된 교차 현장이 있고, 민감도 55장에는 이전 ROI 검토 사진 16장이 남는다.", size=9.5, color=MUTED)
    fig.savefig(output, dpi=DPI, metadata={"Software": "tools/plot_stage1_fold2_matched.py"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    draw(load_panels(args.receipt), args.output)


if __name__ == "__main__":
    main()
