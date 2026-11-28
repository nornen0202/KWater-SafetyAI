"""Redraw the measured tail replay from public aggregate receipts."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/malgun.ttf"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures/sub78-tail-replay-measured-20261128.png"))
    args = parser.parse_args()
    raw = Path("results/sub78-tail-replay-20261128.json").read_bytes()
    result = json.loads(raw)
    proof = json.loads(Path("results/sub78-tail-replay-figure-evidence-20261128.json").read_bytes())
    if hashlib.sha256(raw).hexdigest() != proof["score_receipt_sha256"]:
        raise ValueError("Mismatched score and independent figure evidence")
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=args.font).get_name(),
                         "font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#355a7b", "#286a52"]
    names = ["분모 1 · 원래 방식", "분모 16 · 고정"]
    fig = plt.figure(figsize=(13.1, 7.6), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.32])
    ax1, ax2, ax3, ax4 = (fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2))
    note = fig.add_subplot(grid[2, :])
    note.axis("off")
    fig.suptitle("마지막 한 번의 갱신 · 분모 실측 대조", fontsize=24, weight="bold")
    gradient = proof["gradient_norms"]
    movement = result["adapter_movement"]
    for ax, values, title, unit, precision in (
        (ax1, [gradient[a]["preclip_l2_logged"] for a in ("native85", "fixed85")],
         "클리핑 전 기울기", "기록된 기울기 L2", (".3f", ".4f")),
        (ax2, [gradient[a]["postclip_l2_approx"] for a in ("native85", "fixed85")],
         "클리핑 후 기울기", "기울기 L2 · 기록값과 상한 1.0에서 산출", (".3f", ".4f")),
        (ax3, [movement[f"prefix84_to_{a}_delta_l2"] for a in ("native85", "fixed85")],
         "실제 어댑터 갱신량", "84 → 85 스텝 가중치 차이의 L2", (".5f", ".5f")),
    ):
        ax.barh(range(2), values, color=colors, height=.55)
        ax.set_yticks(range(2), names)
        ax.set(xlim=(0, 1.29 * max(values)), xlabel=unit)
        ax.set_title(title, loc="left", weight="bold")
        ax.invert_yaxis()
        for i, (value, spec) in enumerate(zip(values, precision, strict=True)):
            label = ("약 " if ax is ax2 else "") + format(value, spec)
            ax.text(value + .025 * max(values), i, label, va="center", weight="bold")
    base, fixed = (result["metrics"][a] for a in ("native85", "fixed85"))
    changed = result["fixed_vs_native"]
    ax4.axis("off")
    ax4.set_title("같은 30장의 위험 판정", loc="left", weight="bold")
    lines = [
        f"정상 사진 오경보  {base['normal_false_alarms']} → {fixed['normal_false_alarms']} / {result['normal_photos']}장",
        f"중대 위험 완전 미탐  {base['severity4_complete_misses']} → {fixed['severity4_complete_misses']} / {result['severity4_photos']}장",
        f"위험 없음 출력  {base['no_hazard_outputs']} → {fixed['no_hazard_outputs']} / {result['photos']}장",
        f"위험 없음 → 위험 {changed['empty_to_hazard']}장 · 반대 {changed['hazard_to_empty']}장",
        f"코드 집합 변경 {changed['all_changed_code_sets']} / {result['photos']}장 · 정답 코드 +{changed['gold_codes_added']} / -{changed['gold_codes_lost']}",
    ]
    for i, line in enumerate(lines):
        ax4.text(.02, .88 - .17*i, line, fontsize=14, va="top", transform=ax4.transAxes)
    note.text(0, .78, f"고유 사진 {proof['unique_training_exposed_photos']}장 · 현장 {proof['unique_groups']}개 · 모두 훈련에 사용된 진단 표본",
              fontsize=13, weight="bold")
    gen = result["generation_provenance"]["arms"]
    note.text(0, .33, f"원본과 분모 1 재생의 어댑터 SHA 동일. 원본 신규 {gen['original85']['newly_generated_rows']}건, "
              f"분모 1 재생은 같은 {gen['native85']['reused_rows']}건 재사용, 분모 16 신규 {gen['fixed85']['newly_generated_rows']}건.",
              fontsize=11)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
