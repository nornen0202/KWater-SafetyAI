"""Plot the measured full1045 gradients and the same-gradient counterfactual."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    data = json.loads(Path("results/sub78-tail-clipping-audit-20261128.json").read_bytes())
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    rows = data["full_logs_80_85_and_170"]
    keys = [str(x["step"]) for x in rows]
    history = [x["grad_norm"] for x in rows]
    after = [min(1, x) for x in history]
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=165)
    x = range(len(rows))
    ax.bar([i - .17 for i in x], history, width=.32, color="#84a99e", label="로그 · 클리핑 전")
    ax.bar([i + .17 for i in x], after, width=.32, color="#20604f", label="기존 · 클리핑 후 근사")
    for i in (5, 6):
        v = data["analytic_gradient"][keys[i]]["counterfactual_postclip_if_identical_raw_gradient"]
        ax.scatter(i + .17, v, marker="D", color="#df6d43", zorder=5,
                   label="동일 원시 기울기 ÷ 16을 가정" if i == 5 else None)
        ax.annotate(f"{v:.3f}", (i + .17, v), xytext=(15, 0), textcoords="offset points", va="center")
    ax.axhline(1, color="#777", linestyle="--", linewidth=1)
    ax.set(xticks=list(x), xticklabels=keys, xlabel="업데이트 스텝", ylabel="전역 L2 기울기 노름",
           title="전량 1,045장 학습의 마지막 누적 배치")
    ax.set_ylim(0, 2.65)
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.text(.1, .015, "스텝 85·170은 각각 예제 1건이다. 기울기 노름으로 AdamW 파라미터 갱신량이나 미탐의 원인은 판정할 수 없다.", fontsize=9)
    fig.subplots_adjust(bottom=.20)
    fig.savefig("reports/figures/sub78-tail-gradient-audit-20261128.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
