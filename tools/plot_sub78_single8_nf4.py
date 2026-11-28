"""Draw the measured fixed-draft NF4 verifier replay from its aggregate."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/malgun.ttf"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures/sub78-single8-verifier-nf4-20261128.png"))
    args = parser.parse_args()
    result = json.loads(Path("results/sub78-single8-verifier-nf4.json").read_bytes())
    if (result["draft_precision"], result["verifier_precision"]) != ("bf16", "nf4"):
        raise ValueError("Expected the measured BF16 draft and local NF4 verifier")
    panels = list(result["panels"].values())
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=args.font).get_name(),
                         "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False})
    fig = plt.figure(figsize=(13.8, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, .20], width_ratios=[1.1, 1.05])
    ax, table, footer = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    fig.suptitle("고정 BF16 초벌에 8B NF4 검증기를 붙인 결과", fontsize=23, weight="bold")
    colors = {"epoch-1": "#645e78", "epoch-2": "#147359"}
    for i, panel in enumerate(panels):
        for arm, offset in (("epoch-1", -.14), ("epoch-2", .14)):
            p = panel["paired"][f"{arm}_vs_eight_b_standalone"]
            delta = p["macro_f1_delta_b_minus_a"]
            lo, hi = p["percentile_95_ci"]
            ax.errorbar(delta, i + offset, xerr=[[delta-lo], [hi-delta]], fmt="o", markersize=7,
                        capsize=5, color=colors[arm], label=f"검증기 E{arm[-1]}" if i == 0 else None)
        p = panel["paired"]["epoch-2_vs_eight_b_standalone"]
        delta = p["macro_f1_delta_b_minus_a"]
        ax.text(delta, i+.28, f"{delta:+.3f}", ha="center", va="center", color=colors["epoch-2"])
    ax.axvline(0,color="#4b5563",linewidth=1,linestyle="--")
    ax.set(xlim=(-.02,.105), ylim=(-.44,2.84), xlabel="초벌 대비 Macro-F1 차이 · 현장 단위 95% 구간")
    ax.set_yticks(range(3), [f"{p['photos']}장 / {p['groups']}현장" for p in panels])
    ax.invert_yaxis()
    ax.legend(loc="lower right", ncols=2, frameon=False)
    ax.grid(axis="x",color="#e1e7ed")
    ax.set_axisbelow(True)
    ax.set_title("F1 변화와 불확실성", loc="left", fontsize=17, weight="bold")
    table.axis("off")
    table.set_title("정상 오경보 / 중대 4 위험 완전 미탐", loc="left", fontsize=17, weight="bold")
    cells=[]
    for panel in panels:
        row=[str(panel["photos"])+"장"]
        for arm in ("sixth","eight_b_standalone","epoch-1","epoch-2"):
            m=panel["metrics"][arm]
            row.append(f"{m['no_hazard_false_alarms']}/{m['no_hazard_photos']}  /  "
                       f"{m['severity4']['complete_misses']}/{m['severity4']['photos']}")
        cells.append(row)
    t=table.table(cellText=cells,colLabels=["표본","6차","BF16 초벌","NF4 E1","NF4 E2"],
                  cellLoc="center",bbox=[0,.30,1,.58],colWidths=[.13,.20,.23,.22,.22])
    t.auto_set_font_size(False)
    t.set_fontsize(11.5)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor("#d7e1e8")
        cell.set_facecolor("#edf4f2" if c==4 else "#f5f7fa" if r==0 else "white")
    primary=panels[0]["decisions"]
    base,e2=primary["epoch-1"],primary["epoch-2"]
    table.text(.02,.17,f"175장 승인 {base['accepted_after_gate']} → {e2['accepted_after_gate']}건 "
               f"(일치 {e2['accepted_gold_true']} · 불일치 {e2['accepted_gold_false']})",
               fontsize=13,weight="bold")
    footer.axis("off")
    footer.text(0,.73,"로컬 RTX 3060 · 고정 52쌍 전체 생성 × 2 · 기존 4B 문턱 전이",fontsize=13,weight="bold")
    footer.text(0,.33,"모든 E2 개선 구간이 0을 포함한다. 107장 중대 미탐 19/39는 6차 18/39보다 많다. BF16 검증기와 공식 200장 배포 시간은 미측정이다.",fontsize=11)
    fig.savefig(args.output,dpi=160)


if __name__ == "__main__":
    main()
