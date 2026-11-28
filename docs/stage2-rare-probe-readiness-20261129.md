# Stage 2 B1 희귀 3종 준비 감사 · 2026-11-29

기존 이미지 전용 OOF 점수는 개발 사진 966장 모두에서 확인했다. 고정 0.5와 원점수 top-1·top-3에서는 수몰·밀폐공간·고열/한랭의 공식 양성 3건을 모두 놓쳤다. 8B 판독 원문의 `scene_note`는 372장에 남아 있고 모두 어댑터가 켜진 판독이다. 같은 원본 8B를 NF4로 적재해 어댑터를 끈 상태에서 966장 전체를 확인하는 B1 실행을 준비했다. [집계와 입력 지문](../results/stage2-rare-probe-readiness-20261129.json) · [Stage 1 판정 기준](../reports/safeground-v6-stage1-20261128.md)

## 기존 후보 신호

고정 개발 폴드의 사진 수는 216/177/197/201/175장, 그룹 수는 합계 192개다. 밀폐공간 양성은 Fold 0, 수몰은 Fold 1, 고열/한랭은 Fold 2에 각 1장 있다. 세 코드가 모두 음성인 963장에는 일반 위험 장면이 포함된다. 정상 사진은 290장이다. 이 분할에서 사진·그룹을 훈련에서 제외했더라도 후속 육안 검토에서 확인된 동일 현장 교차는 남아 있다. [개발 분할](../results/clean_split_v1_summary.json) · [현장 교차 점검](fold0-crosssplit-visual-audit-20261127.md) · [신규 OOF 분할 설계](../results/stage2-oof-design-20261128.json)

| 모델과 신호 | 실제 확인한 개발 범위 | 학습 노출과 B1 적용 범위 |
| --- | --- | --- |
| 8B BF16, 활성 어댑터, `scene_note` | Fold 2 혼합 첫 에폭 197장과 Fold 4의 175장, 합계 **372/966장**. 공식 열노출 양성 1장 포함. Fold 4 두 에폭은 같은 범위다. | 두 판독은 해당 평가 사진을 뺀 769장·791장 적합본이다. Fold 0·1·3의 594장이 남는다. Fold 4 두 에폭은 각각 구조 오류 1건이 있어도 관찰 문자열 175개가 유효하게 저장돼 있다. [Fold 2](../results/seoul-fold2-matched-8b-epoch1-20261129.json) · [Fold 4](../results/seoul-l4-roadmap-experiments-20261128.json) |
| 4B BF16, 활성 어댑터, `scene_note` | Fold 0·2·4 **588/966장**, 밀폐공간·열노출 양성 각 1장 포함. | 폴드별 학습 어댑터의 해당 외측 생성이다. 8B와 합쳐도 Fold 1의 177장과 Fold 3의 201장이 남으며 수몰 양성은 그 범위에 있다. [Fold 0](../results/fold0_qlora_v2_118_bf16_vlm.json) · [Fold 2](../results/fold2-v3-4b-epoch1-20261128.json) · [Fold 4](../results/stock-27b-fold4-20261128.json) |
| 4B BF16, 어댑터 비활성, `scene_note` | Fold 0 **216/966장**, 밀폐공간 양성 1장 포함. | 원래 가중치 판독을 위한 참고 대조다. 8B의 무어댑터 생성 캐시는 이 분할에서 **0/966장**으로 확인했다. [4B 실행 기록](../results/fold0_qlora_v2_118_bf16_verified_followup.json) · [비활성 코드 경로](../kwater_lab/hybrid_pipeline.py) |
| 27B stock Q4 GGUF | Fold 4 형식화 응답 175장. 원생성 내부 검사 통과 174장. | 대회 어댑터 없이 평가했다. 현지 보존본은 공식 형식화 응답이고 Fold 4에 공식 희귀 양성이 없다. [27B 평가](../results/stock-27b-fold4-20261128.json) · [저장 경로](../kwater_lab/gguf_timing_entrypoint.py) |
| SigLIP2+DINOv2 전문가의 원점수 | **966/966장**, `966×11` 유한 코드 점수 및 장면 점수 966개. | 평가 순간 입력은 RGB 7-view 특징뿐이다. 외측 사진의 주석·notes·심각도·경로·ID를 후보 판단에 넣지 않는다. 외측마다 다른 폴드의 정답과 notes로 헤드를 훈련하므로 원본 8B 검증과 별개다. [OOF 실행 결과](../results/clean_specialist_v1_oof.json) · [특징과 외측 적합 코드](../kwater_lab/clean_specialist.py) |

원문 캐시 9개의 입력 순서·사진 지문·출력 대응은 보관된 개발 원장과 대조했다. 966장 OOF 파일은 원장의 순서와 폴드를 모두 일치시킨 뒤, `y_codes` 배열을 읽지 않고 원 `code_probs`의 후보를 고정했다. 공식 양성을 포함한 폴드의 외측 전문가 헤드는 그 클래스의 유일한 공식 양성을 훈련에서 제외했다. 아래 후보 수는 비교용 기술 통계다. 외측 주석이 직간접 관여하는 기존 OOF 보정점수나 전량 적합 임곗값은 적용하지 않았다. [집계와 검사 범위](../results/stage2-rare-probe-readiness-20261129.json)

| 원점수 후보 규칙 | 공식 희귀 코드 회수 | 공식 비희귀 963장 중 희귀 후보를 1개 이상 낸 사진 |
| --- | ---: | ---: |
| 희귀 원확률 ≥ 0.5 | 0/3 | 0/963 |
| 전체 11종 원확률 top-1 | 0/3 | 3/963 |
| 전체 11종 원확률 top-3 | 0/3 | 144/963 |

제공 어댑터와 전량 1,045장 어댑터는 개발 사진을 사용한 계보다. Fold 2 적합 어댑터에도 Fold 0 밀폐공간과 Fold 1 수몰 양성이 들어 있으므로, B1의 목표인 세 양성의 원본 판독은 거주 어댑터를 끈 동일 원본 가중치에서 진행한다. 원 제작사의 사전학습 데이터에 해당 사진이 포함됐는지는 확인할 수 없다. [제공·전량 이력](../reports/turnaround-handoff-20261127.md) · [769장 Fold 2 적합 원장](../results/fold2-matched-4b-8b-epoch1-20261128.json)

기존 `kwater_lab.rare_probe`는 170장 노출 이력과 희귀 양성 0개를 검사한 뒤 이전에 추출한 512차원 CLIP 전체화면 특징을 사용한다. `--features --diagnostic --exposure --old-prompts --model-dir --output`은 그 표본을 고정하는 계약이다. 11개 코드와 정상 문구를 함께 코사인 순위화했을 때 top-1은 38/170, top-3은 88/170에서 희귀 후보를 냈다. 최종 오탐과 희귀 양성 회수는 그 CLI로 측정할 수 없다. [170장 실행기](../kwater_lab/rare_probe.py) · [170장 결과](../results/rare_candidate_pilot.json)

## 원본 8B 실행과 판정

1. B2에서 쓰는 8B NF4, BF16 연산, 393,216px의 단일 RTX 3060 모델을 재사용한다. 두 단계의 모든 순전파는 `model.disable_adapter()` 안에서 실행한다. 진입 뒤 LoRA가 병합되지 않았고 각 층에서 비활성임을 확인한다. 모든 질의는 `use_cache=False`, 마지막 로짓 위치 한 개로 제한한다. 활성 어댑터의 KV·표현·관찰 기록을 사용하지 않는다.
2. 1차는 사진 한 장과 수변, 제한 공간 입구, 고열·한랭원에 대한 고정 환경 문구를 사용한다. 각 조합·보이는 환경 없음·판단 불가를 포함한 9개 선택지의 다음 토큰 로짓만 비교한다. 동률이면 판단 불가로 기록한다. 사진 순서는 주석과 별개로 고정하며, 모델 입력에는 사진의 픽셀과 고정 질의만 넣는다.
3. 2차는 1차의 후보 코드마다 해당 환경이 실제 가시적인지와 관련 위치의 방호 결여가 명백한지를 두 개의 독립 질의로 확인한다. 각각 예/아니요/판단 불가를 한 토큰으로 비교한다. 두 질의 모두 `logit(예) - max(logit(아니요), logit(판단 불가)) ≥ ln(9)`를 만족해야 수락한다. 사진 밖 방호, 기체·온도·작업시간처럼 확인할 수 없는 사실은 추정하지 않는다. 공식 양성이나 963장 음성으로 문턱을 맞추지 않는다.
4. 완료 후에만 공식 정답과 합친다. 코드마다 양성 1장·음성 965장에서 후보와 최종 TP/FN/FP를 낸다. 비희귀 963장의 코드별·합집합 최종 FP, 정상 290장의 합집합 FP, 공식 양성 3장의 후보 회수와 최종 회수를 별도로 기록한다. `0/963`을 관측하더라도 독립 표본 가정하의 단측 95% 상한은 약 0.31%다. 동일 현장 상관, 개발 표본의 기존 노출, 외부 검증 부족을 함께 해석한다.

처음 처리하는 20장은 사진 순번에 고정 난수를 적용해 뽑는다. 앞선 RTX 3060의 다른 8B NF4 검증은 콜드 적재 351.6초, 52개 단일토큰 마진 질의 87.93–111.73초, CUDA 최대 예약량 10.14–10.30 GiB였다. B1 실행기는 대표 20장의 이미지 확인·1차·2차 시간을 따로 기록하고, 그 합에서 966장 시간의 첫 추정치를 계산한다. 후보 코드마다 2차 질의가 두 개씩 추가된다. 이 준비 감사 시점에는 실제 B1 시간과 CUDA 피크가 미측정이다. [기존 현지 하드웨어 실측](sub78-single8-verifier-replay-20261128.md) · [B1 실행기](../kwater_lab/stage2_rare_base_probe.py)

실제 AutoProcessor에서 일곱 질의의 완성 접두사와 선택지의 단일 토큰을 확인했다. 합성 RGB 이미지로 질의마다 사진 그리드 한 개, 확장된 시각 토큰 361개와 동일한 답변 직전 토큰을 검산했다. [동결 토큰 및 입력 지문](../results/stage2-rare-probe-readiness-20261129.json)

아래 준비 및 채점은 GPU를 사용하지 않는다. 거주 모델 실행의 `after_extract` 콜백은 B2 외측 특징까지 저장된 직후 호출한다. 재개 시 `data/stage2-rare-base-nf4-20261129/`의 동결 조합과 사진 지문을 검사하고 이미 완료된 행을 재사용한다. 사진별 캐시는 행 번호, 사진 지문, 로짓과 측정값만 로컬에 보관한다.

```powershell
$runner = 'C:/Users/JY/.antigravity_kwater_venv/Scripts/python.exe'
& $runner -B -m kwater_lab.stage2_rare_base_probe freeze
& $runner -B -m unittest discover -s tests -p test_stage2_rare_base_probe.py -v

# Run only in the local GPU slot assigned to B2 followed by B1.
@'
from kwater_lab import stage2_rare_base_probe as b1
from kwater_lab import stage2_representation_head as b2

def after_extract(model, processor, _):
    b1.run_with_model(model, processor, base=b2.BASE, adapter=b2.ADAPTER,
                      manifest=b1.MANIFEST, image_root=b2.IMAGE_ROOT, out=b1.WORK)

b2.extract(b2.WORK, b2.BASE, 'all', None, after_extract=after_extract)
'@ | & $runner -u -B -

& $runner -B -m kwater_lab.stage2_rare_base_probe score
```

## 외부 대조 확보 상태

| 현지 원장 | 현재 적용 범위 |
| --- | --- |
| FloodNet Supervised v1.0, CDLA-Permissive-1.0 | 공급자와 원본 지문, 기존 CLIP 적합 192장과 별도 시험 114장(침수 50장, 음성 64장)이 추적된다. 해당 시험은 현행 8B 팀 어댑터 적합에 쓰지 않았다. 미국 허리케인 이후 항공 영상이므로 수몰 분야 시점 전이용 대조로 남긴다. 산업현장 수변 작업·방호의 외부 검증에는 해당 장면의 별도 표본이 필요하다. [특징 원장](../public-data/floodnet-pilot/README.md) · [외부 표본 원장](../results/external_inventory.json) |
| AI-Hub 71793 부산 침수 CCTV | 출처와 원본 샘플 지문은 확보됐다. 기존 CCTV 1,000장의 공급자 위험값은 내수/외수 구분과 완전히 겹친다. 실제 장면 정답과 평가 전용 이용권을 확인하기 전에는 B1의 독립 대조에 편입하지 않는다. [기존 집계](../results/external_busan.json) · [이용 범위 원장](safeground-v6-external-assets-declaration-draft.md) |
| 밀폐공간·고열/한랭 참고 자료 | 고용노동부·안전보건공단 PDF는 지침 검토용으로 추적된다. 개별 사진의 사용 권한, 실제 출처, 현재 시스템 개발·학습에 들어가지 않았음을 확인한 현장 양성 및 정상 대조 표본은 원장에서 찾지 못했다. [이용 범위 원장](safeground-v6-external-assets-declaration-draft.md) |

외부 검증에는 각 코드의 실제 양성과 같은 시각 환경의 정상 대조가 필요하다. 수변의 방호된 작업 위치, 통제된 제한 공간 입구, 방호된 고열/한랭 환경처럼 1차 후보는 되지만 2차에서 제외되어야 하는 장면을 포함한다. 출처·권리·수집 시점이 확인되고 팀 학습·문턱 선정과 겹치지 않는 장소에서 양성과 정상 대조를 동결한다. 외부 대조를 확보하기 전 B1은 개발 진단 상태로 유지한다.
