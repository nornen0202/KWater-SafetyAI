# 현장 위험 판독에서 검출·분할·캡셔닝의 추가 가치를 검증하는 실험

## 후속 실측 — 익명 48장 직접 판독과 Qwen 비교 입력

Codex가 단독으로 익명 사진 48장을 눈으로 판독했다. 사진 순서와 대상 계열은 알았고 공식 양성·대조 배정, 정답표, 모델 예측은 열지 않았다. 사진 네 장씩을 배열한 화면에서 각 장의 표시 영역(500×400)을 기준으로 사각형을 표시한 뒤 EXIF 방향이 반영된 원본 픽셀 좌표로 환산했다. 이 좌표는 확대용 근사 사각형이다. 가느다란 선의 픽셀 마스크나 독립된 사람 검토자 간 합의는 측정하지 않았다.

| 대상 계열 | 사진 | 보이는 관계를 지지 | 보이는 범위에서 반대 근거 | 관계 불명확 |
|---|---:|---:|---:|---:|
| 걸림·넘어짐 | 16 | 5 | 1 | 10 |
| 추락 | 16 | 4 | 0 | 12 |
| 낙하·인양 | 16 | 2 | 0 | 14 |
| 합계 | 48 | 11 | 1 | 36 |

통로·작업 바닥을 가로지르는 선이나 가로 부재 사이의 개방 공간은 지목할 수 있었다. 반면 후크와 사슬을 잡은 손이 보여도 하중은 화면 밖일 수 있고, 단부가 보여도 아래 깊이와 작업자의 발 위치가 가려진 장면이 있었다. 불명확 36장도 관찰 내용과 이유를 한 장씩 기록하고 비교 분모에 남겼다. 위 관계는 현장 안전 정답과 별개인 2차원 시각 관찰이다. 잔재와 사람이 한 화면에 있다는 사실에서 낙하 궤적을 추정하지 않았다.

**비교 입력.** 로컬 `data/vision-roi-48/reviewed-v1/` 아래에 검수한 LabelMe 파일 48개, `observations.csv`, 좌표 원장, `roi.json`과 `fixed.json`을 보관한다. 두 조건 모두 1024×384 PNG에 왼쪽은 전체 사진, 오른쪽은 원본 폭과 높이의 60%를 잘라 넣었다. 고정 타일은 익명 순서로 네 모서리를 균등 순환한다. ROI는 표시된 사각형 전체를 감싼 직사각형의 중심에 같은 크기의 창을 둔다. 사각형이 없는 4장은 양쪽에 같은 타일을 넣었다. 창보다 넓은 영역은 일부가 잘릴 수 있으므로 이 건수와 두 입력의 실제 동일 건수를 집계에 포함했다. 이미지·프롬프트·추론 명단에는 좌표선, 대상 계열, 관찰 메모를 넣지 않았다.

**영역 잘림에 대한 보완.** 44장에 사각형 총 69개(대상 41, 상대 대상 23, 차단물 5)를 그렸다. 동일 배율 비교에서 5쌍은 실제 이미지 바이트까지 동일했다. 24장은 고정 폭 창이 표시 영역 일부를 잘라냈다. 이를 해결하기 위해 세 번째 명단 `roi_full.json`과 이미지 48장을 만들었다. 해당 24장만 창 경계를 필요한 만큼 넓혀 모든 사각형을 담았다. **관심영역 전체를 보여주는 실용 비교는 fixed 대 roi_full, 배율을 통제한 위치 효과는 fixed 대 roi**로 보고한다. 전자는 일부 사진의 확대 배율·여백도 달라진다. 공통 판독 48장 전체를 분모로 삼고, 세 조건 중 성적이 좋은 조건만 골라 보고하는 것을 피한다.

후속 실행은 기존 `kwater_lab.vlm_11code.infer_batched_vlm`에 각 JSON을 각각 읽어 `eval_rows`로 주고 `image_root`에는 위 패킷 디렉터리를 준다. 같은 공식 Qwen3-VL-4B와 Fold 0 v2 118스텝 어댑터, 같은 `system_prompt_11code.txt`, BF16, 393,216화소 상한, batch 1, greedy·620 생성 토큰으로 고정한다. 캐시를 쓰면 조건마다 경로와 입력 지문을 분리한다. 모델은 한 번 적재해 재사용한다. **이 단계에서 Qwen 실생성은 0건**이다. 대상 계열을 알고 고른 영역의 효과를 보는 개발 진단이며 실제 배포 라우팅이나 독립 외측 성능 추정으로 쓰지 않는다. 수동 판독 원문과 사진은 로컬에 보존한다. [입력 생성 코드](../tools/build_blind_roi_inputs.py) · [검수·비교 입력 집계](../results/vision-roi-review-20261127.json)

2026년 11월 27일. 메인 8종 Fold 4 학습은 14:05 KST에 68/100스텝이었다. 메인 세션은 8종·v3 단독·하이브리드의 175장 및 107장 채점을 담당한다. 이번 실험은 별도 개발 사진에서 **사람이 정확한 영역을 알려주면 현재 판독기가 오류를 고칠 수 있는가**부터 묻는다. 이 효과가 있어야 검출기·분할기·캡션 모델의 추론 비용을 지불할 이유가 생긴다. [메인 로드맵](safeground-v3-review-parallel-roadmap-20261127.md)

## 기존 실측에서 남은 질문

Grounding DINO Tiny는 64장 중 사람을 20장에서 찾았고 나머지 다섯 대상은 한 번도 찾지 못했다. 16장에서 사람과 위험 대상 동시 검출은 Tiny 0장, Base 1장이었다. 이때 사람이 확인한 가시 대상 분모는 없다. 같은 16장의 Tiny 경로에서 14장은 확대할 대상이 없어 원경으로 복귀했다. Base 박스에서 SAM 2.1 Tiny·Large는 각각 실제 마스크 6개를 만들었다. SAM 3.0은 공식 가중치 지문 검증까지 완료했고, 3.1은 첫 사진에서 tracker의 인자 불일치로 멈췄다. 3.1의 개선은 주로 영상 다중 객체 추적이다. 단일 사진은 3.0 정지영상 경로부터 비교한다. [객체·마스크 실측](../docs/phase2-item38-measured-20261126.md) · [3.1 실행 원장](../docs/phase2-large-model-execution-20261126.md) · [Meta 발표](https://ai.meta.com/blog/segment-anything-model-3/)

SigLIP2와 DINOv2-L은 이미 공간·전역 후보 생성에 사용했다. 과거 920장 실험에서 구성 변경으로 F1이 0.2658→0.2926이 됐지만 중대 완전 미탐은 96→109장이었다. Depth Anything V2의 170장 탐색에서 낙하·인양 TP는 0→2, FP는 13→6이었다. 추락 TP는 7/34 그대로였고 외부 추락 위험 상태 순위는 2/7쌍만 맞았다. DINOv3는 같은 헤드에서의 특징 교체, 깊이는 사람·부재가 겹쳐 보이는 장면에서의 추가 단서로 시험할 가치가 있다. [기존 공간 특징 실험](../docs/phase01-development-oof-and-full-fit-20261126-2155e8fa.md) · [깊이 실측](depth-and-state-pairs.md)

## 모델 선택

| 역할 | 우선 비교와 채택 이유 |
|---|---|
| 개방어휘 직접 검출·분할 | SAM 3.0을 품질 기준으로, EfficientSAM3 EV-M(89.2M)을 보조 PC의 첫 경량 후보로 둔다. SAM3-LiteText는 주로 텍스트 인코더를 줄인다. Falcon Perception 0.6B는 독립 구조의 후보이며 제작사가 지적한 부재 장면 오검출을 정상 사진에서 따로 측정한다. [SAM 3](https://github.com/facebookresearch/sam3) · [EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3) · [Falcon](https://huggingface.co/tiiuae/Falcon-Perception) |
| 박스와 실행 비용 | 기존 Grounding DINO Base와 고정 타일이 대조군이다. YOLOE-26은 텍스트·이미지 예시 프롬프트 속도 후보, OWLv2는 독립 검출기, RF-DETR은 사람 등 기존 클래스의 저비용 기준이다. YOLOE의 AGPL·Enterprise 및 RF-DETR의 모델별 조건을 확인한다. [YOLOE](https://docs.ultralytics.com/models/yoloe) · [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) · [RF-DETR](https://github.com/roboflow/rf-detr) |
| 관계 기반 위치 지목 | LocateAnything-3B와 Rex-Omni 3B는 복합 질의가 단일 명사로 놓친 대상을 회수하는지 보는 후보이다. NVIDIA 비상업 연구 조건 및 IDEA·Qwen 약관을 사용 목적·재배포와 대조한 뒤 선택한다. [LocateAnything](https://huggingface.co/nvidia/LocateAnything-3B) · [Rex-Omni](https://github.com/IDEA-Research/Rex-Omni) |
| 영역 설명 | MIT의 Florence-2-large-ft(0.77B)를 첫 캡션 모델로 둔다. 같은 영역에서 PAM 1.5B/3B, DAM-3B를 비교할 수 있다. DAM 가중치는 비상업 조건이다. Molmo2/MolmoPoint·Sa2VA는 설명 속 대상을 실제로 지목하는지 확인할 연구 후보이다. [Florence](https://huggingface.co/microsoft/Florence-2-large-ft) · [PAM](https://github.com/Perceive-Anything/PAM) · [DAM](https://huggingface.co/nvidia/DAM-3B) · [Molmo2](https://github.com/allenai/molmo2) · [Sa2VA](https://github.com/bytedance/Sa2VA) |
| 공간 특징·깊이 | DINOv3-B/16과 기존 DINOv2를 같은 ROI·헤드·학습 사진에서 비교한다. DA3-Small과 V2-Small은 낙하·인양에서 같은 사진 깊이와 교환 깊이를 비교한다. 단안 깊이로 실제 높이·거리를 단정하지 않는다. [DINOv3](https://github.com/facebookresearch/dinov3) · [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) |

Grounded-SAM 2는 검출과 분할을 연결하는 재현 기준이다. HQ-SAM·Semantic-SAM은 가는 경계와 부품 분할, RAM++는 어휘 발굴에 맞는다. Caption Anything의 외부 언어모델 단계는 채점 서버에서 사용할 수 없다. CoTracker·SAM 3D의 시간·입체 정보는 현재 단일 사진에서 독립적으로 검증하기 어려워 후순위다. [Grounded-SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) · [HQ-SAM](https://github.com/SysCV/SAM-HQ) · [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM) · [RAM++](https://github.com/xinyu1205/recognize-anything) · [Caption Anything](https://github.com/ttengwang/Caption-Anything)

## 48장 최초 동결 시점의 검수 절차와 준비 상태

**동결 결과.** 제외 조건을 통과한 308장에서 24쌍·48장·38그룹을 선정했다. 한 그룹은 최대 4장을 차지한다. 세 코드가 8쌍씩이고 대조 24장 중 공식 정상과 다른 위험 사진은 각각 12장이다. 48개 원본의 SHA-256을 전부 대조했다. 검수 폴더에는 익명 PNG 48장, 빈 LabelMe 양식 48개, 관찰 기록 CSV가 있다. 최초 동결 당시 ROI 판정은 0/48장이었다. 개발 사진 중 시각적으로 가까운 쌍을 의도적으로 골랐으므로 여기서 얻은 오류율을 리더보드 분포로 환산할 수 없다.

재현은 프로젝트 루트에서 CPU BLAS 스레드를 1로 지정한 뒤 `python -B tools/prepare_vision_roi_panel.py`로 한다. 로컬 패킷이 이미 있는 경우 기존 검토자 작업을 보존하고 종료한다.

Fold 1·2·3의 개발 사진 575장에서 공식 양성은 걸림 170장·38그룹, 추락 130장·26그룹, 낙하·인양 40장·13그룹이다. 사진의 사전 노출 표시, Fold 0·4의 확인된 교차 장면 훈련측 그룹, 기존 공식 대비쌍 32장의 SHA를 제외한다. 같은 출처·서로 다른 추정 그룹에서 각 코드 8쌍을 고른다. 음성 후보는 공식 정상 4장과 다른 위험 표시 4장이다. 선택 시 그룹 반복을 먼저 줄이고 동률에서 저장된 SigLIP2·DINOv2 전역 특징 코사인 유사도를 높인다. 독립 시험이 아닌 개발용 난례 표본이다. 기존 모델의 학습 노출과 미확인 현장 중복 가능성이 있다. [선정 코드](../tools/prepare_vision_roi_panel.py) · [동결 집계](../results/vision-roi-panel-20261127.json)

1. 로컬 `data/vision-roi-48/review/`만 검토자에게 연다. R01~R48 이미지와 같은 이름의 LabelMe JSON이 대응한다. `observations.csv`에는 대상 위험 *계열*만 표시한다. 정답, 짝, 출처, 기존 모델 점수는 가린다. 정답표 `key-local.json`은 1차 판정 뒤 대조한다.
2. EXIF 방향을 반영한 전체 해상도 PNG에서 실제 보이는 대상에만 LabelMe 사각형을 그린다. 라벨은 `object`(케이블·단부·상부 불안정 부재 등), `counterpart`(통로·작업자·노출 공간), `barrier`(난간·덮개 등)로 쓰며 같은 사진의 연관된 사각형은 같은 group_id를 준다. 두 좌표는 원점 좌상단의 픽셀이다. 추후 두 역할을 포함하는 하나의 확대 영역을 구성한다.
3. CSV의 두 가시성 열은 `visible / not_visible / unclear`, 관계는 `supported / contradicted / unclear` 중 하나로 기록한다. 가려짐·저화질·화면 밖을 구별하고 관찰 한 문장과 검토자 이니셜을 남긴다. 미검출이나 무주석만으로 관계를 부정하지 않는다. 미확인 영역을 임의로 그리지 않는다.
4. 걸림은 물체가 통로를 가로지르는지, 추락은 작업자·단부·난간 배치, 낙하·인양은 상부 잔재·불안정 부재·매달린 하중과 노출 공간을 관찰한다. 각 계열에서 양성·음성 후보 모두 같은 지침을 쓴다. 불일치·판독 불가를 보존하고 가능하면 두 사람이 독립 판정한다. 48장 전체와 관계 판독 가능 하위집단의 분모를 모두 기록한다.

## 보조 PC 실험과 판정

| 단계 | 바꾸는 한 가지 | 판단 |
|---|---|---|
| 1. 사람이 고른 영역의 가치 | 같은 Qwen에 원경, 원경+고정 타일, 원경+사람 ROI를 준다. 재판독 횟수와 총 시각 토큰을 맞춘다. | 사람이 고른 ROI에서도 미탐 회복·오경보 기각이 없으면 해당 하위 유형의 자동 검출기 확대를 중단한다. |
| 2. 자동 위치 찾기 | 고정 프롬프트로 SAM 3.0·EV-M·기존 Base의 검출을 비교한다. 객체 가시성 골드에서 개별 회수율과 두 역할 동시 회수율을 센다. | 대상 부재 오검출, 가려짐, 원경 복귀율을 포함한다. 자동 ROI를 동일 Qwen에 넣어 실제 코드 변경의 정오를 확인한다. |
| 3. 마스크의 순기여 | 같은 박스에서 확대만, 실제 마스크 윤곽, 면적 일치 사각형·교환 마스크를 비교한다. | 마스크 처리 비용을 상쇄하는 개선이 없으면 박스만 채택한다. |
| 4. 캡션의 순기여 | 같은 ROI에서 캡션 없음, Florence 캡션, 비슷한 사진의 길이 맞춘 교환 캡션을 비교한다. | 위치·대상·관계별 사실성을 모델명을 가리고 확인한다. 근거 문장의 허위 주장, 기존 정답 손실과 개선을 센다. 법령과 예방수칙은 원래 경로를 사용한다. |
| 5. 확증과 비용 | 정책·모델·프롬프트를 내측에서 고정하고 하나의 후보만 별도 그룹으로 평가한다. 메인 175/107장을 본 뒤 고른 정책이면 그 두 집단은 재생 결과로 명시한다. | 활성 Macro-F1, 코드별 TP/FP/FN, 전체 정상 오경보, 중대성 4 완전 미탐, 그룹 쌍체 95% 구간을 낸다. 팀 기준 +0.03 F1·정상 오경보 ≤20%·중대 미탐 비증가를 적용한다. |

보조 PC의 VRAM은 실기에서 확인하고 batch 1부터 시작한다. 사람 ROI는 연구 상한으로만 쓰고 최종 배포에서는 자동 입력만 허용한다. 시간 예산이 부족하면 내측에서 호출 대상을 동결하고, 호출하지 않은 사진 중 자신 있게 틀린 사례도 집계한다. 실제 ZIP·빈 캐시에서 설치, 가중치 로드, ROI 추출, 추가 판독까지 모두 잰다. 공식 규정 revision 93은 오프라인 L4 24GB, ZIP 20GB, Public 200장 1,800초·Private 300장 3,600초다. 모델 가중치별 조건 및 생성 감독의 사용 방법을 기록하고 외부 데이터가 들어갈 때는 신고 요건을 적용한다. 이번 후속 검수로 수동 ROI와 비교 입력을 준비했다. 위험 판정 개선과 Public 점수는 미측정이다. [공식 채점·제출 안내](https://hackathon.imaginationgroup.co.kr/kwater/api/report)
