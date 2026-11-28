"""Render the dated Public leaderboard gap from public aggregate scores."""
import argparse
from datetime import datetime
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


def diagnostics_plot(aggregate, destination):
    arms = ("6", "7", "8", "8B")
    labels = ["6차 4B 단독", "7차 전량 4B 하이브리드", "비교안 전량 4B 단독", "8B 단독"]
    colors = ["#537894", "#286850", "#a86835", "#263c73"]
    checks = aggregate["full_panel_label_diagnostics"]
    times = [aggregate["official_kit_total_seconds"][arm] for arm in arms]
    columns = (
        ("정상 오경보", [checks["arms"][arm]["safe_false_alarms"] for arm in arms],
         checks["official_no_visible_hazard"], "장"),
        ("중대 4 위험 완전 미탐", [checks["arms"][arm]["severity4_complete_misses"] for arm in arms],
         checks["official_severity4_photos"], "장"),
        ("공식 키트 설치·실행", times, aggregate["research_8b"]["official_kit"]["limit_seconds"], "초"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.3), sharey=True, layout="constrained")
    fig.suptitle("같은 개발 사진 200장 · 출력과 서울 L4 실행시간", fontsize=23, weight="bold")
    for ax, (title, values, denominator, unit) in zip(axes, columns):
        ax.set_title(title, fontsize=15, weight="bold")
        ax.barh(range(len(arms)), values, color=colors, height=0.55)
        for index, value in enumerate(values):
            inside = unit == "초" and value >= denominator / 2
            x = value - denominator / 45 if inside else value + denominator / 45
            ax.text(x, index, f"{value:,}/{denominator:,}{unit}",
                    color="white" if inside else "#213344", ha="right" if inside else "left",
                    va="center", fontsize=12, weight="bold")
        ax.set_xlim(0, denominator * 1.21)
        ax.grid(axis="x", color="#e6edf1")
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[-1].axvline(columns[-1][2], color="#9b3c25", linestyle="--", linewidth=1.5)
    axes[0].invert_yaxis()
    fig.supxlabel("개발 사진은 일부 또는 전부가 학습에 사용됐다. 품질 수치는 이 패널의 출력 진단이다.", fontsize=12)
    fig.savefig(destination, dpi=170)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seventh", type=float)
    parser.add_argument("--eighth", type=float)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/malgun.ttf"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--comparisons", type=Path)
    parser.add_argument("--four-arm", type=Path, help="Four-arm 200-photo timing and diagnostic JSON")
    args = parser.parse_args()
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=args.font).get_name()
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
    if args.comparisons:
        comparison_plot(json.loads(args.comparisons.read_bytes()), args.output or "reports/figures/sub78-output-changes-20261128.png")
        return
    if args.four_arm:
        diagnostics_plot(json.loads(args.four_arm.read_bytes()),
                         args.output or "reports/figures/sub78-four-arm-200-20261128.png")
        return
    receipt = json.loads(Path("results/public_submission_007.json").read_bytes())
    observation = receipt.get("scored_team_leaderboard_observation") or receipt["team_leaderboard_observation"]
    timestamp = datetime.fromisoformat(observation["observed_at_kst"])
    seventh = receipt["public_score"] if args.seventh is None else args.seventh
    scores = [42.04, 16.12, 42.04, 42.04, 47.35, 53.65, seventh, args.eighth]
    fig, ax = plt.subplots(figsize=(10.6, 6.4), layout="constrained")
    ax.set_title("공개 점수와 현재 순위 진입선", loc="left", fontsize=23, weight="bold", pad=29)
    subtitle = (f"{timestamp.month}월 {timestamp.day}일 {timestamp:%H:%M} KST 관측 · "
                f"우리 팀 {observation['team_public_rank']}위 / {observation['ranked_teams']}팀")
    ax.text(0, 1.015, subtitle,
            transform=ax.transAxes, color="#35455c", fontsize=13)
    for rank, color in ((20, "#35705e"), (10, "#9b3c25")):
        score = observation[f"top_{rank}_cutoff"]
        label = f"{rank}위  {score:.2f}"
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
