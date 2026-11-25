"""Plot the aggregate depth comparison on two distinct evaluation sets."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fonts
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parents[1]
def results(name):
    return json.loads((root / "results" / name).read_text(encoding="utf-8"))


official = results("depth_pilot.json")["metrics"]
external = results("depth_state_pairs.json")["metrics"]
font = Path("C:/Windows/Fonts/malgun.ttf")
if font.exists():
    fonts.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = fonts.FontProperties(fname=str(font)).get_name()
plt.rcParams["axes.unicode_minus"] = False
methods = ("rgb", "brightness", "edge", "depth", "wrong_depth")
names = ("사진 RGB", "+ 밝기", "+ 2D 경계", "+ 같은 사진 깊이", "+ 다른 사진 깊이")
colors = ("#405266", "#8898a7", "#8898a7", "#126856", "#ab5f24")
fig, (left, right) = plt.subplots(1, 2, figsize=(11.4, 4.0))
left.set_title("대회 탐색 사진 170장", loc="left", fontsize=15, pad=14)
left.set_xlabel("위험 코드 macro-F1 (관측 7종)")
right.set_title("외부 현장의 추락 상태 7쌍", loc="left", fontsize=15, pad=14)
right.set_xlabel("비정상 사진의 점수가 더 높았던 쌍 수")
for y, (method, color) in enumerate(zip(methods, colors)):
    score = official[method]["macro_f1"]
    right_count = external[method]["fall_height"]["abnormal_scored_higher"]
    left.barh(y, score, height=.54, color=color)
    left.text(score+.004, y, f"{score:.3f}", va="center", fontsize=10)
    right.barh(y, right_count, height=.54, color=color)
    right.text(right_count+.1, y, f"{right_count} / 7", va="center", fontsize=10)
for axis in (left, right):
    axis.set_yticks(range(len(methods)), names)
    axis.invert_yaxis()
    axis.grid(axis="x", color="#dce3e8", linewidth=.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "left", "right"]].set_visible(False)
    axis.spines["bottom"].set_color("#a7b3ba")
    axis.tick_params(axis="both", length=0, pad=6)
left.set_xlim(0, .265)
left.set_xticks([0, .05, .10, .15, .20, .25])
right.set_xlim(0, 7.25)
right.set_xticks(range(8))
fig.subplots_adjust(left=.13, right=.98, bottom=.18, top=.85, wspace=.44)
fig.savefig(root / "reports/figures/depth-and-state-pairs.png", dpi=160, facecolor="white", bbox_inches="tight", pad_inches=.17)
plt.close(fig)
