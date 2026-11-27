"""Plot paired changes and group-bootstrap intervals from the 4B/8B Fold 4 score receipt."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


def main():
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "results/paired-qwen-4b-8b-fold4-20261128.json").read_bytes())
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Malgun Gothic", "Noto Sans CJK KR", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 11})
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    for cohort, labels, color in (("primary_175", (3, 2), "#185d68"), ("sensitivity_107", (1, 0), "#a6581f")):
        for epoch, y in enumerate(labels, 1):
            comparison = result["panels"][cohort]["comparisons"][f"4b_epoch{epoch}_vs_8b_epoch{epoch}"]
            delta = comparison["macro_f1_delta_b_minus_a"]
            low, high = comparison["percentile_95_ci"]
            ax.errorbar(delta, y, xerr=[[delta-low], [high-delta]], fmt="o", ms=8, color=color,
                        elinewidth=2.6, capsize=5)
            ax.text(delta, y+.24, f"{delta:+.3f}", ha="center", va="bottom", color=color, fontsize=10)
    ax.set_yticks([3, 2, 1, 0], ["175장 · 첫 에폭", "175장 · 둘째 에폭", "107장 · 첫 에폭", "107장 · 둘째 에폭"])
    ax.axvline(0, color="#304450", lw=1.2)
    ax.grid(axis="x", color="#dce2e6", lw=.65)
    ax.set_axisbelow(True)
    ax.set(xlim=(-.15, .24), ylim=(-.55, 3.7), xlabel="macro-F1 변화량  (8B - 4B)")
    ax.xaxis.set_major_locator(MultipleLocator(.05))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#87939a")
    ax.tick_params(axis="y", length=0, pad=8)
    fig.subplots_adjust(left=.19, right=.97, top=.75, bottom=.21)
    fig.text(.055, .92, "8B의 점 추정치는 네 비교에서 모두 높았다", fontsize=16, weight="bold", color="#172832")
    fig.text(.055, .835, "가로선은 현장 그룹을 단위로 2,000회 재표집한 95% 구간이다. 네 구간 모두 0을 포함한다.",
             fontsize=10, color="#455861")
    fig.savefig(root / "reports/figures/qwen-scale-fold4-20261128.png", dpi=165, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
