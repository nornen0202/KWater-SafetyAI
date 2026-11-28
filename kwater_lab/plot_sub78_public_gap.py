"""Render the dated Public leaderboard gap from public aggregate scores."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def comparison_plot(aggregate, destination):
    pairs = ("6_to_7", "6_to_8", "7_to_8")
    groups = (
        ("same_code_set", "같은 코드 집합", "#dae3ec"),
        ("empty_to_hazard", "정상 → 위험", "#ae4c2d"),
        ("hazard_to_empty", "위험 → 정상", "#287963"),
        ("hazard_code_changed", "위험 코드 변경", "#2d5479"),
    )
    counts = aggregate["comparisons"]
    if aggregate["photos"] != 200 or any(sum(counts[key].values()) != 200 for key in pairs):
        raise ValueError("Every paired result must cover 200 images")
    fig, ax = plt.subplots(figsize=(11, 4.6), layout="constrained")
    ax.set_title("같은 사진 200장에서 달라진 판정", loc="left", fontsize=24, weight="bold", pad=20)
    ax.set_yticks(range(len(pairs)), [key.replace("_to_", "차 → ") + "차" for key in pairs])
    for index, key in enumerate(pairs):
        start = 0
        for group, label, color in groups:
            count = counts[key][group]
            ax.barh(index, count, left=start, color=color, height=0.58, label=label if index == 0 else None)
            if count:
                ax.text(start + count / 2, index, str(count), ha="center", va="center",
                        color="#1d2a36" if group == "same_code_set" else "white", weight="bold", fontsize=11)
            start += count
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=12)
    ax.set_xlabel("사진 수 · 전량 학습에 포함된 개발 사진의 출력 변화", fontsize=12)
    ax.set_xlim(0, 200)
    ax.invert_yaxis()
    fig.savefig(destination, dpi=170)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seventh", type=float)
    parser.add_argument("--eighth", type=float)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/malgun.ttf"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--comparisons", type=Path)
    args = parser.parse_args()
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=args.font).get_name()
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
    if args.comparisons:
        comparison_plot(json.loads(args.comparisons.read_bytes()), args.output or "reports/figures/sub78-output-changes-20261128.png")
        return
    scores = [42.04, 16.12, 42.04, 42.04, 47.35, 53.65, args.seventh, args.eighth]
    fig, ax = plt.subplots(figsize=(10.6, 6.4), layout="constrained")
    ax.set_title("공개 점수와 현재 순위 진입선", loc="left", fontsize=23, weight="bold", pad=29)
    ax.text(0, 1.015, "11월 28일 12:33 KST 관측 · 우리 팀 27위 / 133팀",
            transform=ax.transAxes, color="#35455c", fontsize=13)
    for score, label, color in [(55.89, "20위  55.89", "#35705e"), (60.19, "10위  60.19", "#9b3c25")]:
        ax.axhline(score, color=color, linewidth=1.6, linestyle="--")
        ax.text(8.46, score, label, color=color, va="center", fontsize=13, bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    for index, score in enumerate(scores, 1):
        if score is None:
            ax.text(index, 4, "미측정", ha="center", color="#5b687a", fontsize=12)
        else:
            ax.bar(index, score, width=0.55, color="#194772" if index >= 6 else "#6c9bb7")
            ax.text(index, score - 2.7, f"{score:.2f}", color="white", weight="bold", ha="center", va="top", fontsize=12)
    ax.set(xticks=range(1, 9), xticklabels=[f"{i}차" for i in range(1, 9)], ylim=(0, 67),
           xlim=(0.45, 9.54), ylabel="Public 점수")
    ax.set_xlabel("점선은 11월 28일 관측값이며, 과거 제출 시점의 순위 진입선과 다를 수 있습니다.", labelpad=12, fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e6ecf1")
    fig.savefig(args.output or "reports/figures/sub78-public-gap-20261128.png", dpi=170)


if __name__ == "__main__":
    main()
