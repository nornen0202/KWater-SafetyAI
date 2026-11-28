"""Draw Sub 7/8 figures from committed aggregate JSON only."""

import argparse
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
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.unicode_minus": False,
        "text.color": "#1b3831",
        "axes.labelcolor": "#1b3831",
        "figure.facecolor": "#f5f8f5",
        "axes.facecolor": "#f5f8f5",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
GREEN, BLUE, RUST = "#216653", "#3979a4", "#aa623d"


def load(name):
    return json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))


def export(fig, name):
    fig.savefig(
        FIG / f"sub78-{name}.png",
        dpi=190,
        bbox_inches="tight",
        pad_inches=0.22,
        metadata={"Software": "tools/plot_sub78_audit.py"},
    )
    plt.close(fig)


def quality():
    mixed = load("sub78-8b-verifier-audit-20261128")["panels"]
    fixed = load("fold4-v3-fixed-verifier-paired-code-evaluation")["panels"]
    stock = load("stock-27b-fold4-20261128")["panels"]
    receipt = all("measured_missing_merged" in mixed[key] for key in ("primary_175", "sensitivity_107"))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.1 if receipt else 5.4))
    fig.subplots_adjust(wspace=0.19, left=0.055, right=0.98, top=0.84, bottom=0.21)
    for ax, key in zip(axes, ("primary_175", "sensitivity_107"), strict=True):
        panel = mixed[key]
        entries = [
            ("6차 4B 1에폭", panel["sixth"], False),
            ("4B 1에폭 + 고정 검증", fixed[key]["arm_metrics"]["epoch1_fixed2"], False),
            ("8B 1에폭 단독", panel["eight_b_standalone"], False),
        ]
        if receipt:
            entries.append(("8B + 4B 검증 · 실측¹", panel["measured_missing_merged"], False))
        entries.extend(
            [
                ("8B + 4B 검증 · 사전 범위", panel["unmeasured_omitted"], True),
                ("27B Q4 단독", stock[key]["arm_metrics"]["stock_27b_q4"], False),
            ]
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.4, len(entries) + 0.9)
        ax.axis("off")
        ax.text(
            0.0, len(entries) + 0.65, f"{panel['photos']}장 · {panel['groups']}그룹", size=14, weight="bold"
        )
        for x, label in ((0.49, "Macro-F1"), (0.73, "정상 오경보"), (0.9, "심각도 4 미탐")):
            ax.text(x, len(entries) + 0.1, label, ha="center", size=9.5, weight="bold")
        for i, (name, metrics, interval) in enumerate(entries):
            y = len(entries) - 0.65 - i * 0.96
            observed = name.endswith("실측¹")
            if observed or i % 2 == 0:
                ax.axhspan(y - 0.36, y + 0.36, facecolor="#d9eae2" if observed else "#e7f0ea")
            ax.text(0.01, y, name, va="center", size=10, weight="bold" if observed or interval else "normal")
            sev = metrics["severity4"]
            f1_text = f"{metrics['macro_f1']:.4f}"
            sev_text = f"{sev['complete_misses']}/{sev['photos']}"
            if interval:
                lo, hi = panel["exact_assignment_ranges"]["macro_f1"]
                low_miss, high_miss = panel["exact_assignment_ranges"]["severity4_complete_misses"]
                f1_text = f"{lo:.4f}–{hi:.4f}"
                sev_text = f"{low_miss}–{high_miss}/{sev['photos']}"
            ax.text(0.49, y, f1_text, ha="center", va="center", size=10, color=GREEN)
            ax.text(
                0.73,
                y,
                f"{metrics['no_hazard_false_alarms']}/{metrics['no_hazard_photos']}",
                ha="center",
                va="center",
            )
            ax.text(0.90, y, sev_text, ha="center", va="center")
    fig.suptitle("Fold 4 위험 코드 점수와 오류", fontsize=19, weight="bold", x=0.055, ha="left")
    note = (
        "¹ 고정 4B 검증기. 기존 Windows 캐시와 서울 L4 추가 5건을 결합한 실측이다."
        if receipt
        else "8B 초벌과 고정 4B 검증기. 추가 5건은 아직 미측정이다."
    )
    fig.text(0.055, 0.105, note, size=10)
    fig.text(0.055, 0.061, "사전 범위는 미측정 5건을 승인·기각한 32개 조합의 최솟값과 최댓값이다.", size=10)
    fig.text(
        0.055,
        0.020,
        "오른쪽은 교차 현장 그룹을 제외한 패널이다. 정상 오경보와 중대 미탐의 분모는 서로 다르다.",
        size=9,
    )
    export(fig, "fold4-quality-20261128")


def timing(eighth_evidence=None):
    source = load("seoul-l4-roadmap-experiments-20261128")
    sixth = source["accepted_sixth_public200"]
    seventh = source["full1045_seventh_public200"]["kit_clock"]
    big = next(
        p["official_clock"]
        for p in source["gguf_27b_official"]["panels"].values()
        if p.get("official_clock", {}).get("total_seconds") == 1591
    )
    spend = source["spend_estimate_krw"]
    if eighth_evidence is None:
        default_path = ROOT / "results/sub78-seoul-final-run-20261128.json"
        eighth_evidence = default_path if default_path.is_file() else None
    final = json.loads(eighth_evidence.read_text(encoding="utf-8")) if eighth_evidence else {}
    clock8 = final.get("eighth", {}).get("kit_clock")
    cost8 = final.get("cost")
    budget = spend["approved_budget_krw"]
    prior = [
        spend["using_known_transfer_scenario"],
        spend["including_1500_krw_total_network_and_miscellaneous_allowance"],
    ]
    fig, (ax, cost) = plt.subplots(1, 2, figsize=(13.0, 5.1), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.13, right=0.97, top=0.78, bottom=0.20, wspace=0.57)
    clocks = (sixth, seventh, clock8, big)
    totals = [1800]
    for i, clock in enumerate(clocks):
        if clock is None:
            ax.text(45, i, "공식 설치·실행 미측정", va="center", color=RUST)
            continue
        install, infer = clock["install_seconds"], clock["execution_seconds"]
        total = clock.get("total_seconds", clock.get("kit_total_seconds"))
        if install < 0 or infer < 0 or install + infer != total:
            raise ValueError("Inconsistent official installation/execution times")
        totals.append(total)
        ax.barh(i, install, color=RUST)
        ax.barh(i, infer, left=install, color=GREEN)
        ax.text(total / 2, i, f"{install}+{infer:,} = {total:,}초", color="white", ha="center", va="center")
    ax.set_yticks(range(4), ["6차 4B", "7차 전량 4B", "8차 후보", "27B Q4"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals) * 1.09)
    ax.axvline(1800, color="#81a799", linestyle="--", linewidth=1)
    ax.set_xlabel("설치 + 실행(초) · 제한 1,800초")
    ax.set_title("같은 200장 공식 키트", loc="left", fontsize=13, weight="bold", pad=16)

    incremental = None
    if cost8 is not None:
        incremental = [cost8["incremental_known_krw"], cost8["incremental_allowance_krw"]]
        cumulative = [cost8["cumulative_known_krw"], cost8["cumulative_allowance_krw"]]
        if (
            cost8["approved_total_krw"] != budget
            or min(incremental) < 0
            or [old + new for old, new in zip(prior, incremental, strict=True)] != cumulative
        ):
            raise ValueError("Inconsistent incremental/cumulative Seoul cost scenarios")
    totals_cost = [budget]
    for i, amount in enumerate(prior):
        color = GREEN if i == 0 else BLUE
        cost.barh(i, amount, height=0.52, color=color)
        cost.text(amount / 2, i, f"이전 {amount:,}원", va="center", ha="center", color="white", size=9)
        if incremental is not None:
            new = incremental[i]
            totals_cost.append(amount + new)
            cost.barh(i, new, left=amount, height=0.52, color="#a7c6b9")
            cost.text(amount + new + 220, i, f"{amount + new:,}원", va="center", ha="left", size=9)
            cost.text(
                amount + new / 2, i + 0.39, f"+{new:,}원", va="center", ha="center", size=9, color=GREEN
            )
    cost.axvline(budget, linestyle="--", color=RUST, linewidth=1.2)
    cost.set_yticks(range(2), ["실제 반출량 기준", "부대비 여유 포함"])
    cost.set_ylim(1.75, -0.68)
    cost.set_xlim(0, max(totals_cost) * 1.16)
    cost.set_xlabel(f"총 승인 한도 {budget:,}원 · 청구서 미확인")
    cost.set_title("서울 L4 누적 비용 추정", loc="left", fontsize=13, weight="bold", pad=16)
    fig.suptitle("서울 L4 200장 실행 시간과 비용", x=0.105, ha="left", weight="bold", fontsize=18)
    message = (
        "연한 막대는 이번 증분 추정이다. 시나리오별 이전 비용과 이번 비용을 합해 점선의 승인 한도와 비교했다."
        if incremental is not None
        else "8차 공식 200장 실행 및 추가 지출 산정이 끝나면 이 표에 반영한다."
    )
    fig.text(0.105, 0.043, message, fontsize=9.5)
    export(fig, "seoul-timing-budget-20261128")


def fold2_masks():
    source = load("sub78-fold2-prospective-readiness-20261128")
    masked = source["sensitivity_mask"]
    panels = (
        source["primary_197"],
        masked["retained"],
        masked["optional_stricter_visual_exposure_sensitivity_roi_and_curation"],
    )
    if tuple(panel["photos"] for panel in panels) != (197, 55, 19):
        raise ValueError("Fold 2 mask sizes differ from the documented panels")
    labels = ("Fold 2 전체 개발", "교차·보류 관련 그룹 제외", "ROI·큐레이션 노출도 제외")
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    fig.subplots_adjust(left=0.25, right=0.90, top=0.77, bottom=0.21)
    for i, panel in enumerate(panels):
        total, normal = panel["photos"], panel["normal"]
        hazard = panel["hazard_photos"]
        cap = panel["max_normal_false_alarms_at_18pct"]
        if normal + hazard != total or cap != normal * 18 // 100:
            raise ValueError("Fold 2 normal/hazard/false alarm denominators disagree")
        ax.barh(i, normal, height=0.49, color=GREEN, label="정상 사진" if i == 0 else None)
        ax.barh(i, hazard, left=normal, height=0.49, color="#acc6ba", label="위험 사진" if i == 0 else None)
        ax.text(total + 4, i, f"{total}장 · {panel['groups']}그룹", va="center", size=10)
        ax.text(0, i + 0.35, f"정상 {normal}장 · 18% 기준 최대 {cap}건", va="center", size=9)
    ax.set_yticks(range(3), labels)
    ax.set_ylim(2.67, -0.68)
    ax.set_xlim(0, 240)
    ax.legend(frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.19), fontsize=9)
    ax.set_xlabel("사진 수")
    fig.suptitle("Fold 2 노출 이력에 따른 검증 분모", x=0.10, ha="left", fontsize=18, weight="bold")
    fig.text(
        0.10,
        0.038,
        "문서화된 노출을 제외한 민감도다. 모델 점수는 아직 없고, 남은 사진의 완전 미노출도 확인되지 않았다.",
        size=9,
    )
    export(fig, "fold2-exposure-masks-20261128")


def package_options():
    result = load("sub78-single-8b-verifier-design-audit-20261128")
    budget = load("sub78-package-readiness-20261128")["eight_a"]
    archive = result["archive"]
    limit = budget["fold4_mixed_model_replay"]["unquantized_archive_limit_bytes"] / 1e9
    bars = [
        ("7차 전량 4B 하이브리드", archive["seventh_bytes"], "실물 ZIP", GREEN),
        ("8B 단독", budget["fold4_8b_epoch1"]["standalone_zip_estimate_bytes"], "치환 계산", BLUE),
        (
            "8B + 전문가 · LoRA 공유",
            archive["one_adapter_substitution_complete_base_files"],
            "치환 계산",
            BLUE,
        ),
        (
            "8B + 전문가 · LoRA 전환",
            archive["two_adapters_substitution_complete_base_files"],
            "치환 계산",
            BLUE,
        ),
        (
            "8B + 4B BF16",
            budget["fold4_mixed_model_replay"]["archive_bf16_bases_only_bytes"],
            "기반모델 두 개만의 하한",
            RUST,
        ),
    ]
    fig, ax = plt.subplots(figsize=(12.0, 5.1))
    fig.subplots_adjust(left=0.28, right=0.95, top=0.78, bottom=0.23)
    ax.axvline(limit, linestyle="--", color=RUST, linewidth=1.2)
    for i, (name, size, evidence, color) in enumerate(bars):
        gb = size / 1e9
        ax.barh(i, gb, height=0.58, color=color, alpha=0.92, hatch="////" if i == len(bars) - 1 else None)
        ax.text(0.5, i, evidence, va="center", color="white", weight="bold", size=9.5)
        if 0 < limit - gb < 1.1:
            ax.text(gb - 0.35, i, f"{gb:.2f}", va="center", ha="right", size=11, color="white")
        else:
            ax.text(gb + 0.28, i, f"{gb:.2f}", va="center", ha="left", size=11)
    ax.set_yticks(range(len(bars)), [name for name, *_ in bars])
    ax.set_ylim(len(bars) - 0.43, -0.78)
    ax.set_xlim(0, 29)
    ax.set_xlabel("십진 GB (1 GB = 10억 바이트)")
    ax.text(limit + 0.23, -0.61, f"{limit:.0f} GB 제출 한도", color=RUST, size=9.5, weight="bold")
    fig.suptitle("단일 8B 패키지 크기와 제출 한도", x=0.10, ha="left", weight="bold", fontsize=18)
    safe = result["calibration"]["c1_by_kind"]["verify_neg_safe"]
    fig.text(
        0.10,
        0.10,
        f"8B 검증기 실행 시간·최대 VRAM·어댑터 전환은 미측정이다. C1 보정의 정상 음성은 {safe}건이다.",
        fontsize=10,
    )
    fig.text(
        0.10,
        0.053,
        "치환 계산에는 기존 ZIP의 구성과 여유를 적용했다. 실제 8B ZIP 크기는 제작 후 확인한다.",
        fontsize=9,
    )
    export(fig, "single-8b-package-options-20261128")


def reject_decode():
    audit = load("sub78-reject-runtime-audit-20261128")
    replay = audit["replays"]["fold4_fixed_epoch2_bf16_actual_batches"]
    tokens, dense = replay["reencoded_token_estimate"], replay["dense_batch_estimate"]
    original = [tokens["baseline_nonpad"], dense["baseline_dense_token_slots"]]
    early_stop = [tokens["early_stop_nonpad"], dense["early_stop_dense_token_slots"]]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    fig.subplots_adjust(left=0.22, right=0.95, top=0.77, bottom=0.22)
    for i, (a, b) in enumerate(zip(original, early_stop)):
        ax.barh(i + 0.16, a, height=0.29, color="#baccc3", label="기존" if not i else None)
        ax.barh(i - 0.16, b, height=0.29, color=GREEN, label="조기 종료 재생" if not i else None)
        ax.text(a, i + 0.16, f"  {a:,}", va="center")
        ax.text(b, i - 0.16, f"  {b:,}", va="center")
    ax.set_yticks(range(2), ["개별 생성 토큰", "배치의 처리 토큰 슬롯"])
    ax.set_xlim(0, 1.18 * max(original))
    ax.set_xlabel("디코딩된 원문을 실제 토크나이저로 다시 토큰화한 추정치")
    ax.legend(frameon=False)
    fig.suptitle("기각 응답을 끊어도 같은 배치의 다른 응답은 계속된다", fontsize=17, weight="bold")
    fig.text(
        0.04,
        0.037,
        "Fold 4 보존 응답의 CPU 재생. 공식 서울 L4 200장의 단축 시간이나 바이트 동등은 미측정.",
        size=10,
    )
    export(fig, "reject-token-batches-20261128")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eighth-evidence", type=Path)
    parser.add_argument("--only", choices=("quality", "timing", "fold2", "package", "reject"))
    args = parser.parse_args()
    if args.only in (None, "quality"):
        quality()
    if args.only in (None, "timing"):
        timing(args.eighth_evidence)
    if args.only in (None, "fold2"):
        fold2_masks()
    if args.only in (None, "package"):
        package_options()
    if args.only in (None, "reject"):
        reject_decode()
