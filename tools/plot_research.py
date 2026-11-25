"""Render report figures from the shareable aggregate JSON, without loading photos."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


TEAL, BLUE, ORANGE, GRAY, INK = "#007c76", "#35669c", "#ba552c", "#697889", "#182d40"
LABELS = {"slip_trip":"같은 높이 걸림·넘어짐", "fall_height":"높이차 추락", "other":"기타",
          "struck_by_lifting":"인양물 낙하·충돌", "caught_in_machine":"기계 끼임", "electrical_energy":"감전",
          "fire_explosion":"화재·폭발", "chemical_exposure":"화학물질 노출", "confined_space":"밀폐공간",
          "drowning_inundation":"익수·침수", "thermal_weather":"온도·기상"}


def read(name):
    return json.loads((ROOT / "results" / (name + ".json")).read_text(encoding="utf-8"))


def axes(height):
    fig, ax = plt.subplots(figsize=(8.3,height), layout="constrained")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#b6c4cb")
    ax.tick_params(axis="both", length=0, pad=5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e1e8ea", linewidth=.8)
    return fig, ax


def save(fig, out, name):
    fig.savefig(out / (name + ".png"), dpi=175, facecolor="white",
                metadata={"Software": "K-water SafetyAI research"})
    plt.close(fig)


def draw(out):
    counts = read("train_audit_summary")["per_code_photo_count"]
    items = sorted(counts, key=counts.get)
    fig, ax = axes(3.45)
    values = [counts[k] for k in items]
    bars = ax.barh(range(len(items)), values, color=[ORANGE if n==1 else TEAL for n in values], height=.66)
    ax.set_yticks(range(len(items)), [LABELS[k] for k in items])
    ax.bar_label(bars, padding=4, fontsize=9, color=INK)
    ax.set(xlim=(0,345), xlabel="해당 코드가 표시된 사진 수 (장)")
    save(fig,out,"training-support")

    ref = read("prefix_pilot_fpr20")["pooled_out_of_prefix"]["linear_head"]
    w = read("weight_pilot")
    chosen = [ref["inner_fpr20"],ref["fixed_0.5"],ref["inner_calibrated"],w["pooled"]["inner_calibrated"]["blend"]]
    names = ["내측 오경보 20% 제약","임계값 0.5","내측 F1 최대화","분류기 + 사례 검색"]
    fig, ax = axes(2.85)
    ys = np.arange(4)
    for offset, key, denom, color, name in (
            (-.18,"no_hazard_false_alarms",66,ORANGE,"무위험 사진에 오경보 (66장 중)"),
            (.18,"hazard_completely_missed",104,BLUE,"위험 사진의 모든 위험 누락 (104장 중)")):
        vals = [s[key] for s in chosen]
        bars = ax.barh(ys + offset, [100*n/denom for n in vals],color=color,height=.29,label=name)
        ax.bar_label(bars,labels=[f"{n}장" for n in vals],padding=4,fontsize=9,color=INK)
    ax.set_yticks(ys,names); ax.invert_yaxis(); ax.set(xlim=(0,70),xlabel="각 집단에서 차지하는 비율 (%)")
    ax.legend(loc="lower center",bbox_to_anchor=(.43,1.01),ncol=2,frameon=False,fontsize=8.5)
    save(fig,out,"error-tradeoff")

    fig, ax = axes(2.2)
    names = ["공유 방향 제거","같은 크기 무작위 편집","사례 검색 단독","분류기 + 사례 검색"]
    for i,key in enumerate(("shared_axis_edit","random_control","neighbors","blend")):
        r = w["paired_prefix_resampling"][key]
        d = 100*r["macro_f1_delta_b_minus_a"]
        lo,hi = [100*x for x in r["percentile_95_ci"]]
        ax.hlines(i,lo,hi,color=BLUE,lw=2.6)
        ax.plot(d,i,"o",color=TEAL if key=="blend" else BLUE,ms=7,zorder=3)
        ax.text(7.5,i,f"{d:+.2f}",va="center",ha="right",fontsize=10,color=INK)
    ax.axvline(0,color="#7e8f99",ls="--",lw=1)
    ax.set_yticks(range(4),names); ax.invert_yaxis()
    ax.set(xlim=(-8,8),ylim=(3.5,-.5),xlabel="기존 분류기 대비 Macro F1 차이 (퍼센트포인트)")
    save(fig,out,"paired-changes")

    fig, ax = axes(2.35)
    law = read("law_retrieval_pilot")["methods"]
    ex = read("law_examples_pilot")["methods"]
    rows = [law["word_evidence_action"],law["official_rag_no_examples"],ex["code_frequency"],ex["official_train_examples"]]
    yn = ["어절 검색","공식 검색기 · 법령만","코드별 최빈 조문","공식 검색기 · 다른 묶음 사례"]
    ys = np.arange(4)
    for j,(field,color,title) in enumerate((("top1_hits",TEAL,"첫 번째 후보"),("top3_hits",BLUE,"상위 3개 이내"))):
        vals=[x[field] for x in rows]
        bars=ax.barh(ys+(j-.5)*.32,vals,height=.29,color=color,label=title)
        ax.bar_label(bars,labels=[str(n) for n in vals],padding=3,fontsize=9)
    ax.set_yticks(ys,yn);ax.invert_yaxis()
    ax.set(xlim=(0,155),xlabel="라벨에 기재된 조문을 찾은 건수 (147건 중)")
    ax.legend(loc="lower center",bbox_to_anchor=(.44,1),ncol=2,frameon=False)
    save(fig,out,"law-retrieval")

    fig, ax = axes(2.2)
    s = read("adapter_spectrum")
    ranks = [1,4,8,s["rank"]]
    e = s["overall"]["retained_energy_at_rank"]
    energy = [100*e[str(k)] for k in ranks[:-1]]+[100.]
    ax.grid(False);ax.grid(axis="y",color="#e1e8ea")
    ax.plot(ranks,energy,marker="o",markersize=7,linewidth=2,color=TEAL)
    for x,y in zip(ranks,energy):
        ax.annotate(f"{y:.1f}%",(x,y),xytext=(0,8),textcoords="offset points",ha="center",color=INK,fontsize=10)
    ax.set(xlim=(0,17),ylim=(50,112),xticks=ranks,yticks=[60,80,100],
           xlabel="각 업데이트 행렬에 남긴 특이값 수 (원래 랭크 16)",ylabel="보존 에너지 (%)")
    save(fig,out,"adapter-energy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font",type=Path,default=Path("C:/Windows/Fonts/malgun.ttf"))
    parser.add_argument("--out",type=Path,default=ROOT/"reports/figures")
    args=parser.parse_args()
    if not args.font.is_file():
        parser.error("Supply --font with a Korean-capable font file")
    font_manager.fontManager.addfont(str(args.font))
    name = font_manager.FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({"font.family":name,"font.size":10.2,"text.color":INK,"axes.labelcolor":INK,
                         "xtick.color":GRAY,"ytick.color":INK,"axes.unicode_minus":False})
    args.out.mkdir(parents=True,exist_ok=True)
    draw(args.out)
