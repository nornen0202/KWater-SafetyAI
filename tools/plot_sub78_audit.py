"""Draw Sub 7/8 figures from committed aggregate JSON only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"
MALGUN = Path("C:/Windows/Fonts/malgun.ttf")
if MALGUN.exists():
    font_manager.fontManager.addfont(str(MALGUN))
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False, "text.color": "#1b3831",
                     "axes.labelcolor": "#1b3831", "figure.facecolor": "#f5f8f5",
                     "axes.facecolor": "#f5f8f5", "axes.spines.top": False, "axes.spines.right": False})
GREEN, BLUE, RUST = "#216653", "#3979a4", "#aa623d"


def load(name):
    return json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))


def export(fig, name):
    fig.savefig(FIG / f"sub78-{name}.png", dpi=190, bbox_inches="tight", pad_inches=.22,
                metadata={"Software": "tools/plot_sub78_audit.py"})
    plt.close(fig)


def quality():
    mixed = load("sub78-8b-verifier-audit-20261128")["panels"]
    fixed = load("fold4-v3-fixed-verifier-paired-code-evaluation")["panels"]
    stock = load("stock-27b-fold4-20261128")["panels"]
    labels = ["6차 4B 1에폭", "4B 1에폭 + 고정 검증", "8B 1에폭 단독", "8B + 4B 검증 캐시*", "27B Q4 단독"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.subplots_adjust(wspace=.23, left=.06, right=.98, top=.83, bottom=.18)
    for ax, key in zip(axes, ("primary_175", "sensitivity_107")):
        m = mixed[key]
        idxs = (m["sixth"], fixed[key]["arm_metrics"]["epoch1_fixed2"], m["eight_b_standalone"],
                m["unmeasured_omitted"], stock[key]["arm_metrics"]["stock_27b_q4"])
        lo, hi = m["exact_assignment_ranges"]["macro_f1"]
        sev_lo, sev_hi = m["exact_assignment_ranges"]["severity4_complete_misses"]
        ax.set_xlim(0, 1)
        ax.set_ylim(-.65, 5)
        ax.axis("off")
        ax.text(.0, 4.75, f"{m['photos']}장 · {m['groups']}그룹", size=14, weight="bold")
        for x, label in [(0.46, "Macro-F1"), (.72, "정상 오경보"), (.90, "심각도 4 미탐")]:
            ax.text(x, 4.26, label, ha="center", size=9, weight="bold")
        for i, (name, val) in enumerate(zip(labels, idxs)):
            y = 3.57-.92*i
            if i % 2 == 0:
                ax.axhspan(y-.32, y+.32, facecolor="#e7f0ea")
            ax.text(.01, y, name, va="center", size=10, weight="bold" if i == 3 else "normal")
            macro = f"{lo:.6f}–{hi:.6f}" if i == 3 else f"{val['macro_f1']:.4f}"
            ax.text(.46, y, macro, ha="center", va="center", size=10, color=GREEN)
            ax.text(.72, y, f"{val['no_hazard_false_alarms']}/{val['no_hazard_photos']}", ha="center", va="center")
            sev = val["severity4"]
            severe = f"{sev_lo}–{sev_hi}/{sev['photos']}" if i == 3 else f"{sev['complete_misses']}/{sev['photos']}"
            ax.text(.90, y, severe, ha="center", va="center")
    fig.suptitle("Fold 4에서 정확도·오경보·중대 미탐 함께 보기", fontsize=18, weight="bold", x=.06, ha="left")
    fig.text(.06, .08, "* 8B 초벌 + 4B 검증 캐시. 미측정 5건에 대한 32가지 승인 조합의 최솟값~최댓값이다.", size=10)
    fig.text(.06, .034, "정상·심각도 4는 서로 다른 분모다. 오른쪽 패널은 교차 유사 현장을 제외했다.", size=10)
    export(fig, "fold4-quality-20261128")


def timing():
    source = load("seoul-l4-roadmap-experiments-20261128")
    sixth = source["accepted_sixth_public200"]
    seventh = source["full1045_seventh_public200"]["kit_clock"]
    big = next(p["official_clock"] for p in source["gguf_27b_official"]["panels"].values() if p.get("official_clock", {}).get("total_seconds") == 1591)
    spend = source["spend_estimate_krw"]
    fig, (ax, cost) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=.14, right=.98, top=.76, bottom=.21, wspace=.72)
    labels = ["6차 4B", "전량 4B 후보", "27B Q4"]
    timing_rows = [(sixth["install_seconds"], sixth["execution_seconds"]),
                   (seventh["install_seconds"], seventh["execution_seconds"]),
                   (big["install_seconds"], big["execution_seconds"])]
    for i, (install, infer) in enumerate(timing_rows):
        available = 1800-install-infer
        ax.barh(i, install, color=RUST)
        ax.barh(i, infer, left=install, color=GREEN)
        ax.barh(i, available, left=install+infer, color="#dce5df")
        ax.text((install+infer)/2, i, f"{install}+{infer} = {install+infer:,}초", color="white", ha="center", va="center")
        ax.text(1800-available/2, i, f"{available}초", ha="center", va="center", size=9)
    ax.set_yticks(range(3), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1800)
    ax.set_xlabel("설치 + 실행 / 연두색은 1,800초까지 남은 시간")
    audit = load("sub78-reject-runtime-audit-20261128")
    approved_additional = audit["spend_crosscheck"]["additional_approved_ceiling_krw"]
    cost_values = [spend["using_known_transfer_scenario"],
                   spend["including_1500_krw_total_network_and_miscellaneous_allowance"],
                   approved_additional, spend["approved_budget_krw"]]
    for i, number in enumerate(cost_values):
        cost.barh(i, number, color=[GREEN, BLUE, RUST, "#dce5df"][i])
        cost.text(number/2, i, f"{number:,}원", ha="center", va="center", color="white" if i < 3 else "#1b3831")
    cost.set_yticks(range(4), ["사용 추정", "보수 추정", "이번 추가 승인 상한", "총 승인 한도"])
    cost.invert_yaxis()
    cost.set_xlim(0, 15000)
    cost.set_xlabel("이번 상한은 총 한도에 포함 · 청구서 미확인")
    fig.suptitle("서울 L4 실제 200장 실행과 비용 원장", x=.12, ha="left", weight="bold", fontsize=17)
    export(fig, "seoul-timing-budget-20261128")


def reject_decode():
    audit = load("sub78-reject-runtime-audit-20261128")
    replay = audit["replays"]["fold4_fixed_epoch2_bf16_actual_batches"]
    tokens, dense = replay["reencoded_token_estimate"], replay["dense_batch_estimate"]
    original = [tokens["baseline_nonpad"], dense["baseline_dense_token_slots"]]
    early_stop = [tokens["early_stop_nonpad"], dense["early_stop_dense_token_slots"]]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    fig.subplots_adjust(left=.22, right=.95, top=.77, bottom=.22)
    for i, (a, b) in enumerate(zip(original, early_stop)):
        ax.barh(i+.16, a, height=.29, color="#baccc3", label="기존" if not i else None)
        ax.barh(i-.16, b, height=.29, color=GREEN, label="조기 종료 재생" if not i else None)
        ax.text(a, i+.16, f"  {a:,}", va="center")
        ax.text(b, i-.16, f"  {b:,}", va="center")
    ax.set_yticks(range(2), ["개별 생성 토큰", "배치의 처리 토큰 슬롯"])
    ax.set_xlim(0, 1.18*max(original))
    ax.set_xlabel("디코딩된 원문을 실제 토크나이저로 다시 토큰화한 추정치")
    ax.legend(frameon=False)
    fig.suptitle("기각 응답을 끊어도 같은 배치의 다른 응답은 계속된다", fontsize=17, weight="bold")
    fig.text(.04, .037, "Fold 4 보존 응답의 CPU 재생. 공식 서울 L4 200장의 단축 시간이나 바이트 동등은 미측정.", size=10)
    export(fig, "reject-token-batches-20261128")


if __name__ == "__main__":
    quality()
    timing()
    reject_decode()
