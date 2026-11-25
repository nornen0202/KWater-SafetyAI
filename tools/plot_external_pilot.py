"""Plot measured external and competition-pilot metrics from aggregate JSON only."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fonts
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parents[1]
result = json.loads((root/"results/external_flood.json").read_text(encoding="utf-8"))
font = Path("C:/Windows/Fonts/malgun.ttf")
if font.exists():
    fonts.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = fonts.FontProperties(fname=str(font)).get_name()
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1,2,figsize=(12,4.8))
ext = result["external"]["metrics"]
official = result["official_pilot"]
panels = [
    ("FloodNet 시험 사진의 침수 F1",
     ["모두 음성","고정 문장","침수 자료 학습","라벨 섞기"],
     [ext[k]["f1"] for k in ("always_negative","fixed_text","floodnet_centroid","shuffled_train_labels")]),
    ("대회 탐색 사진 170장의 macro-F1",
     ["대회 자료","건설 객체 추가","침수 판정 추가","두 자료 추가"],
     [official[k]["macro_f1"] for k in ("official_only","kalis","floodnet","kalis_and_floodnet")])]
for ax,(title,labels,values) in zip(axes,panels):
    positions=range(len(labels))
    ax.barh(positions,values,height=.56,color=["#64748b","#358175","#006ba8","#695787"])
    ax.set_yticks(positions,labels)
    ax.invert_yaxis()
    ax.set_xlim(0,1.06)
    ax.set_xticks([0,.25,.5,.75,1])
    ax.set_title(title,fontsize=14,loc="left",pad=18)
    ax.grid(axis="x",alpha=.18)
    ax.set_axisbelow(True)
    for y,v in enumerate(values):
        ax.text(v+.015,y,f"{v:.3f}",va="center",fontsize=11,color="#182635")
    for edge in ("top","left","right"):
        ax.spines[edge].set_visible(False)
    ax.tick_params(axis="both",length=0,pad=6)
fig.subplots_adjust(left=.13,right=.99,wspace=.45,top=.84,bottom=.22)
fig.text(.03,.06,"왼쪽은 항공사진의 침수 건물·도로 유무, 오른쪽은 산업현장 위험 코드 조합을 평가했다. 두 패널은 서로 다른 시험이다.",fontsize=10)
dest=root/"reports/figures/external-pilot.png"
dest.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(dest,dpi=170,facecolor="white",bbox_inches="tight",pad_inches=.18)
plt.close(fig)
print(dest)
