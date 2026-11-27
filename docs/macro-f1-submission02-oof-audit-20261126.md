# K-water 현장 안전 AI · 두 번째 접수본의 Macro F1과 다음 세 실험

2026년 11월 26일 KST · 개발팀 검토용 · 공식 학습 자료, 로컬 키트, 접수 영수증 및 공개 운영 규정 기준

**0.25083은 두 번째 접수본과 같은 판정 방식을 이미 탐색한 170장에 적용한 로컬 점수다.** 접두사 한 묶음씩 빼고 만든 예측에서 정답 위험 149건 중 121건이 상위 3위에 들었고, 그중 89건이 최종 경보로 남았다. 놓친 60건은 **후보 순위 밖 28건**, **후보 안이지만 임계값 미달 32건**으로 나뉜다. 추락은 후자가 14건이고 인양·낙하물은 전자가 9건이다. 따라서 추락에는 먼저 보정을, 인양에는 사람·하중·작업 구역이 함께 보이는 영상 증거를 시험할 이유가 있다. 어느 쪽의 순증도 독립 사건에서 측정되지 않았다. [접수본 헤드와 기준 점수](../results/submission2_head.json) · [판정 함수](../kwater_lab/train_head.py#L136)

**실행 순서**는 ① 사건을 확인해 분할하고 같은 분류기로 공식 학습 자료 확대 효과부터 측정 ② 장면 위험 확률과 코드별 경보를 내부 검증에서 공동 보정 ③ 같은 인코더에서 작은 위험 단서와 전체 장면을 함께 판독하는 실험이다. ②를 먼저 수행할 수 있는 범위는 기존 170장의 탐색 진단까지다. Public·Private 성능이나 공식 정성 점수에 대한 향상폭은 미측정이다.

## 1. 원본에서 확인한 버전과 점수

| 관측 시점(KST) | 근거 | 해석 |
|---|---|---|
| 11/26 04:58:06 | G 드라이브 원본 `main`과 로컬 원격 추적 참조 `origin/main`, 리뷰 작업 트리 `nornen0202/kwater-macro-f1-review`가 모두 `12f5fc6c789f9045e2f9a3ee3f28020619a4c75b`. 최초 상태에서 추적 변경·일반 미추적 항목 0 | `git worktree list --porcelain`에 원본 및 C 드라이브 리뷰 트리 2개. 다른 앱이 원본을 함께 편집할 수 있다. |
| 11/26 05:02:27 → 05:04:34 | 원본에서 README 수정과 미추적 `docs/f1-breakthrough-strategy-2026-11-26.md` 관측. 05:03:35 커밋 `e4af204e6ad19a348108c5d03b211c36fb342d13`에 두 파일이 반영됨 | 두 SHA 사이의 변경은 문서 2개다. 42,979바이트 분석 문서의 SHA-256은 `cac156f834667d37cb22a7fcc56b53a6d8fa35071aae083e09a78349c3937a62`. 0.58~0.66+ 등은 그 문서의 전망이다. 평가·학습 산출물의 새 점수를 뒷받침하는 로그는 이 커밋에 없다. [동시 작성 문서](f1-breakthrough-strategy-2026-11-26.md) |
| 로컬 실험과 로그 | `.runtime/prompt-pilot-8192-status.json` 01:11:53 종료, `.runtime/adapter-scale-pread-status.json` 03:21:06 종료, 모두 exit 0. `results/generated_law_comparison.json` 04:26:58, `results/synthetic_basin_probe.json` 04:33:13 수정 | 기존 생성문 법령 대조와 합성 방호 유무 한 쌍 실험까지 반영했다. 완료 로그와 제출 영수증을 별도로 대조했다. [법령 비교](../results/generated_law_comparison.json) · [합성 쌍](../results/synthetic_basin_probe.json) |
| 다른 작업의 확인 범위 | Antigravity의 해당 프로젝트 작업 로그 `9ed2ba00…/.system_generated/tasks/task-102.log` 05:04:02, SHA-256 `ecdc85f46ac8128cc6772be5526d4f35df1f93d8587f6441536f06f8363cd034` | 63개 테스트 OK와 위 README·문서의 변경 기록을 확인했다. 다른 Codex 계정의 결과는 프로젝트에 남은 파일과 Git 이력 범위에서만 반영했다. 작업 중인 앱의 미저장 내용, 신규 모델 점수 및 작업자 개인의 성과는 확인할 수 없다. |

| 구분 | 무엇을 확인했나 | 점수의 범위 |
|---|---|---|
| 1차 접수 · 11/25 20:42 | 운영사 `kwater-baseline.zip` 원본, Qwen3-VL-4B + 공식 LoRA 3에폭. 9,032,392,642바이트, SHA-256 `be91eb3a745dc6e1f305237b44842106707164f4174940439b246e4abb385aab`(기존 원본 대조 원장) | 로컬 키트 10장 모델 원출력 파싱 10/10, 출력 형식 10/10, 위험 예측 2장. 정답 성능과 별개다. [1차 영수증](../results/public_submission_001.json) · [실행](../results/official_baseline_10.json) |
| 2차 접수 · 11/25 23:57 | `kwater-fusion-20261125.zip`. 203,513,187바이트, 항목 17개, 실파일 SHA-256 `22e32c10f56c4dec50a9365ee59c16b79439fa28787132d6f2fcfa500070eaa0`. ZIP 속 헤드 `e09b2e442bd9017dd53576941da1eed1b09c0111fa1826d7eb4ab36c75805e52`. 두 값 모두 접수 기록과 일치 | 170장으로 최종 적합한 헤드. 공식 키트 예시 10장 중 위험 7·미관측 3, 기존 출력 재검사 10/10. `external_research_datasets_used=[]`. [2차 영수증](../results/public_submission_002.json) · [실제 진입점](../submission/fusion/main.py) |
| 2차 방식의 로컬 검증 | 6개 접두사 외측 OOF(out-of-fold). 내부에서 매번 선택한 임계값 `[.2,.3,.3,.2,.2,.2]`이면 0.24705. 그 여섯 값의 중앙값 0.20을 모든 외측 묶음에 재적용하면 0.25083, 12/170장의 코드 집합 변경 | 후자 ×70 = **17.5582/70**, 활성 7코드. 고정 0.20 선정에 다른 외측 반복의 정보가 간접 반영됐다. 두 값 모두 사건 독립 검증이나 배포 헤드 자체의 신규 표본 성능과 구별한다. [헤드 재생](../kwater_lab/submission2_fit.py#L55) · [중첩 검증](../kwater_lab/encoder_pilot.py#L84) |
| 공식 Public / Private / 정성 | 두 접수 기록의 `public_score`와 순위는 null. 11/26 04:59:16 KST에 공개 API에서 `teams=[]`, `sample=false`, `privateRevealed=false`, `updatedAt=null` 확인 | 0점으로 기록하지 않는다. Public F1·LLM·총점, Private 점수, 최종 2차 정성 심사 모두 확인 전이다. 18장 공식 LoRA의 0.5244와 6장 프롬프트 중 공통 완성 4장의 0.250도 별도 탐색치다. [공식 리더보드 API](https://hackathon.imaginationgroup.co.kr/kwater/api/leaderboard) · [18장](../reports/official-lora-scale-pilot.md) · [6장](../reports/full-image-prompt-pilot.md) |

11/26 05:12 KST에 원본 `main`은 `e4af204e6ad19a348108c5d03b211c36fb342d13`, 리뷰 트리는 `12f5fc6c789f9045e2f9a3ee3f28020619a4c75b`였다. 원본의 일반 미커밋 변경은 이 시점에 0개였다. 같은 시각 공개 API를 재확인한 결과 규정 revision 85, 리더보드 0팀이었다.

## 2. 어떤 결과가 공식 점수에 들어가는가

11/26 한국시간에 [운영사 홈페이지](https://hackathon.imaginationgroup.co.kr/kwater/)와 홈페이지가 실제로 읽는 [공개 원문 API](https://hackathon.imaginationgroup.co.kr/kwater/api/report)를 HTTPS로 직접 확인했다. API `revision=85`, `updatedAt=2026-11-25`, 응답 바이트 SHA-256 `f07168654efd427871bd9a580a1f314365c8560507c8c534686f608687842f34`다. 참가자 규칙 변경표에는 **v1.0, 11/25 최초 공개** 한 행이 있다. API 개정 번호와 규칙 판 번호는 서로 다르다. FAQ는 공지·답변과 다르면 최신 규칙 페이지를 적용하도록 안내한다. [규칙 및 변경 이력](https://hackathon.imaginationgroup.co.kr/kwater/#rules) · [FAQ](https://hackathon.imaginationgroup.co.kr/kwater/#faq)

| 항목 | 공식 기준과 이 리뷰에서의 적용 |
|---|---|
| 정량 70점 | 코드마다 전체 평가 사진의 `2TP/(2TP+FP+FN)`를 구한다. 정답 **또는** 예측에 한 번이라도 등장한 코드의 산술평균 ×70이다. 양성도 예측도 없는 코드는 제외, 음성 사진에만 잘못 등장한 코드는 0으로 포함한다. `no_visible_hazard`는 제12코드가 아니다. 그림마다의 F1이나 폴드 F1 평균으로 대체하면 분모가 달라진다. [평가 기준](https://hackathon.imaginationgroup.co.kr/kwater/#evaluation) |
| 리더보드 정성 30점과 결선 | LLM 3종이 사진과 출력 항목 1·2·4·5를 보고 근거 12, 예방수칙 12, 출처 6점을 평가한다. 위험성 수준은 출력 필수이고 Public·Private 리더보드에서는 배점이 없다. Public 200장(모든 제출), Private 300장(Public 상위 20팀의 최종본), 합산 상위 10팀 2차 정성 100점(현장 활용성 40, 방법론 30, 재현성 30). 최종 300점. 공개 총점 하나에서 F1·클래스 분포·정성 점수를 역산할 수 없다. [평가](https://hackathon.imaginationgroup.co.kr/kwater/#evaluation) · [FAQ](https://hackathon.imaginationgroup.co.kr/kwater/#faq) |
| 출력과 무효 | 최상위 `/output/submission.jsonl` 단일 파일, BOM 없는 UTF-8, 한 줄 ≤1 MiB. manifest의 소문자 16진수 32자리 ID마다 한 줄. `hazard`이면 유효 코드 1~3개, 4개 이상은 해당 사진 0점. 미관측이면 빈 객체와 `scene_evidence`. 근거·미관측 근거 240자, 예방 480자, 참조 0~3개. 가능성(1~5)×중대성(1~4)의 구간 등급을 맞춘다. 중복 ID는 해당 사진 모두 무효, 누락 사진 0점, 유효행 0개·출력 파일 전체 오류 등은 전체 0점·순위 제외다. [출력 규격](https://hackathon.imaginationgroup.co.kr/kwater/#output-spec) |
| 실행·시간·용량 | Ubuntu 24.04, Python 3.12, L4 24 GB 1장, CPU 6개, RAM 56 GiB. 오프라인 설치 시작부터 결과 저장까지 Public 1,800초, Private 3,600초. ZIP ≤20 GB. `main.py`·`assets.json`·README를 루트에 두고, 추가 휠은 Linux x86-64 및 버전·해시 고정. 읽기 전용 입력과 제출 폴더, 외부 API·다운로드 및 localhost 연결 차단. `assets.json`에 시드와 사용 자산의 이름·출처·리비전·라이선스·용도 기재. [제출 방법](https://hackathon.imaginationgroup.co.kr/kwater/#submission) · [키트 README 7절](../data/local-kit/README.txt) |
| 시각·횟수 | 팀당 하루 2회, 00:00 KST 초기화. ZIP 검사 통과 접수 시 차감. 11/26 13:00 이전 제출은 그 시각 일괄 반영, 이후에는 LLM 등 채점 완료 시 반영. 12/13 10:00 KST 업로드·최종 1건 지정 마감, 미지정 시 최고 Public 자동 선택. [일정](https://hackathon.imaginationgroup.co.kr/kwater/#schedule) · [채점 반영](https://hackathon.imaginationgroup.co.kr/kwater/#evaluation) |
| 모델·릴리스·자료 권리 | 최신 규칙은 모델 선택을 허용하며 모델별 이용·재배포 권리와 오프라인 실행을 요구한다. 확인한 규칙·FAQ에서 모든 모델에 일괄 적용하는 특정 공개일 컷오프는 찾지 못했다. 2차의 CLIP 수출본 고정 커밋은 2026-10-28 UTC, SigLIP 2 수출본은 2025-02-21 UTC. 자산 신고에는 원모델 기준 MIT, Apache-2.0 및 배포 고지가 포함된다. 다음 후보도 원모델·변환 가중치·런타임별 조건과 파일 해시를 함께 남긴다. [규칙](https://hackathon.imaginationgroup.co.kr/kwater/#rules) · [접수 자산](../submission/fusion/assets.json) · [CLIP 리비전](https://huggingface.co/Xenova/clip-vit-base-patch32/commit/6b7c523703ae26e39be4b03774966f8571c7f523) · [SigLIP 2 리비전](https://huggingface.co/onnx-community/siglip2-base-patch16-384-ONNX/commit/99f5d40afb5331b81eecee4bb02c360fbb9cdf8a) · [원모델 카드](https://huggingface.co/google/siglip2-base-patch16-384) |
| 학습·신고·보안 | 적법한 공개 외부 데이터는 **12/6까지** 운영사 `cs@illusionists.co.kr`로 이름·출처 신고. 데이터별 실제 이용 권한과 신고 여부는 별도이며 확보만으로 신고 완료를 뜻하지 않는다. 외부 AI로 학습 데이터·라벨·설명을 만들면 사용 모델·방법 공개, 후보팀 결과보고서에 개발 AI 도구와 활용 범위 기재. FAQ는 본인/팀 전용 비공개 클라우드 학습을 허용하고 종료 후 삭제를 안내한다. 제공 사진·라벨·베이스라인 가중치는 GitHub 업로드 금지, 자체 코드는 팀 비공개 공유 허용. 이번 검토는 로컬 학습 자료와 기존 출력 집계로 한정했다. 비공개 평가 사진 검색·추정·라벨링·반출, 제공 자료 외부 AI 전송, 대회 제출은 수행하지 않았다. [규칙](https://hackathon.imaginationgroup.co.kr/kwater/#rules) · [데이터 이용 조건](https://hackathon.imaginationgroup.co.kr/kwater/#core-data) · [FAQ](https://hackathon.imaginationgroup.co.kr/kwater/#faq) |

## 3. 사진에서 제출 행까지 추적한 코드 리뷰

아래 순서는 현재 수치 해석과 다음 후보 선택에 미치는 영향을 기준으로 정했다. **관측**은 확인한 코드·파일·재계산, **가설**은 대조 실험에서 반증할 수 있는 설명이다. 줄 링크는 해당 스냅샷 파일의 시작 위치다.

| 순위·신뢰도 | 호출 경로와 관측 | 영향과 확인 방법 |
|---|---|---|
| 1 · 높음 | `train.jsonl`의 `sample_id/image/target/notes` → 감사기가 유효성·화소·그룹을 대조 → `prefix_cohort`가 `S숫자C숫자` 중 12~40장 접두사만 골라 여섯 묶음을 해시 순으로 선택 → `submission2_fit.main`이 170개 ID만 허용. 모수 1,045장에 확인된 사건 매핑 **0**, 탐색 170장·키트/공개 예시 포함 보류 제외 182장. [감사 181행](../kwater_lab/audit.py#L181) · [선택 26행](../kwater_lab/prefix_pilot.py#L26) · [학습 제한 28행](../kwater_lab/submission2_fit.py#L28) · [노출 목록](../data/holdout-exclusions.json) | 전체 730위험/315미관측(무위험률 30.1%)에 비해 탐색은 104/66(38.8%). 나머지 875장을 그대로 모두 훈련에 넣으면 독립 비교의 기준이 사라진다. 원 사건·장소를 확인해 보류를 먼저 잠그고 개발측의 공급량만 늘리는 것이 첫 비교다. |
| 2 · 높음 | 1차 공식 `prepare_dataset.convert`가 11종을 내부 8종으로 변환한다. `struck_by_lifting→struck_crushed`, `other→collision_contact`, `fire_explosion→fire_thermal`, `chemical_exposure→hazardous_exposure`, 나머지 네 종은 같은 이름. 익수·밀폐·온도는 건너뛰고 위험 코드가 다 빠진 행은 제외한다. 1,045장 감사에서 세 코드의 단독 1행·타 코드와 혼합 2행으로 대조됨. `main.convert`가 역매핑한다. [훈련 변환 21·35행](../data/baseline/scripts/prepare_dataset.py#L21) · [8종 검증기](../data/baseline/label_schema.py#L3) · [출력 역매핑](../data/baseline/main.py#L23) | 1차는 세 코드를 출력할 수 없고, `other`의 내부 의미도 더 좁다. 2차는 11열이 있으나 학습 양성이 0인 익수·밀폐·화학·온도 확률은 항상 0. 화학은 전체 19장이라 자료 분할로 개선 가능성을 검증할 수 있다. 익수·밀폐·온도는 전체 각 1장으로 감독학습과 독립 양성 재현율을 동시에 검증할 수 없다. [고정 prior 127행](../kwater_lab/train_head.py#L127) · [전체 분포](../results/train_audit_labels.json) |
| 3 · 높음 | 2차는 원영상을 EXIF 방향 보정·RGB 변환 → 전체 화면을 종횡비 유지 패딩 → CLIP 224와 SigLIP 2 384의 INT8 ONNX CPU 특징(512·768차원) → L2 정규화 → 학습 폴드 평균·표준편차로 표준화 → 각 11개 sigmoid 확률을 1:1 평균한다. 학습은 BCE, L2 0.01, 250회. `argsort` 안정 정렬 상위 3개 중 `p≥0.20`만 통과한다. 별도로 학습한 scene gate는 없고 반환 집합이 비면 `no_visible_hazard`다. [영상 변환](../kwater_lab/clip_probe.py#L104) · [헤드](../kwater_lab/train_head.py#L92) · [배포 점수 34행](../submission/fusion/main.py#L34) · [판정·저장 49행](../submission/fusion/main.py#L49) | 보정이 잘못되면 사진 경보 수와 코드 누락이 함께 변한다. 반대로 순위 밖 정답은 같은 순위에서 공통 임계값을 낮춰도 되살릴 수 없다. 4절에서 두 오류를 실제 건수로 나눴다. `.20`을 11개 코드에 무조건 낮추거나 상위 3개를 항상 채우는 전략은 무위험 66장에 과경보를 만든다. |
| 4 · 높음(구현), 정성 효과 미측정 | 2차 `make_row`는 코드로 항상 같은 한국어 근거·예방·조문 한 개와 가능성·중대성 기본값을 조회한다. 미관측도 단일 문장이다. 1차는 사진 → LoRA greedy 생성 → 파싱/내부 검증 → 유효 행만 동봉 `LawRetriever`로 포맷 → 공식 코드로 변환한다. 생성 실패 행도 최종 변환에서 미관측 기본행이 될 수 있다. [2차 서술·곱셈](../submission/fusion/safety_output.py#L64) · [1차 생성 및 검증](../data/baseline/main.py#L71) · [1차 법령 검색](../data/baseline/submission.py#L11) | 2차의 코드 적중과 문장의 사진 근거는 다른 문제다. 1차의 형식 통과율과 모델 생성 성공률도 별도 계수한다. 배포 LoRA 및 동봉 사례집은 제공 학습자료를 썼다. 새 독립 실험에서 VLM은 공개 기반 모델에서 다시 시작하고 검색 인덱스도 각 학습 폴드만 사용한다. [공식 제공 내용](https://hackathon.imaginationgroup.co.kr/kwater/#core-data) |
| 5 · 높음 | 2차 `run`은 manifest ID 순서로 `submission.tmp`에 쓰고 마지막에 `submission.jsonl`로 교체한다. ZIP 빌더가 접수본의 바이트·해시·항목 CRC를 고정하므로 새 후보의 변경된 소스를 그대로 넣으면 중단된다. 키트는 ZIP → 오프라인 설치 → 제한 실행 → JSON 스키마·위험성 곱 검사. `check_output.py`는 1장이라도 유효하면 OK가 가능하고 BOM을 관대하게 제거한다. 로컬 `metrics` CLI는 ID 불일치나 무효행에서 비교를 중단한다. [ZIP 30행](../tools/build_fusion_submission.py#L30) · [키트 50행 이후](../data/local-kit/run_local.sh#L50) · [행별 검사 92행 이후](../data/local-kit/kit/check_output.py#L92) · [지표 입력 검증](../kwater_lab/metrics.py#L174) | 새 후보에는 별도 버전·출력 경로가 필요하다. `RESULT OK`와 공식 규격 준수 여부를 나누어 확인한다. BOM 0, 출력 파일 1개, 유효행 N/N, 누락·중복·4코드 행 0을 통과 조건으로 둔다. 형식에서 탈락한 행을 임의로 제외하여 F1을 높이지 않는다. |
| 6 · 높음 | `audit.assign_groups`는 묶음 크기와 목표 사진 수로 분할하고 `audit`는 미매핑이나 미판정이 있으면 `needs_group_review`를 낸다. CLI는 `invalid_data`가 아니면 종료코드 0이 가능하다. [그룹 할당 132행](../kwater_lab/audit.py#L132) · [상태 258행](../kwater_lab/audit.py#L258) · [종료코드 325행](../kwater_lab/audit.py#L325) | 종료코드만 확인하면 미완료 사건 지도를 승인할 수 있다. `status=ready`, 교차 그룹 위반 0, 보류-노출 교집합 0, 매핑·분할 파일 해시를 확인한다. 현재 6개 접두사는 사건 또는 장소 인증을 대신하지 않는다. |

## 4. 170장 재점수 결과와 오류가 생긴 위치

원본의 허용된 특징 캐시 두 개를 해시 대조하고, 기존 `submission2_fit.py`의 여섯 외측 계산을 CPU 메모리에서 그대로 재생했다. 각 검증 사진의 헤드는 그 사진의 접두사를 뺀 특징으로만 복원했다. 동일 고정 사양을 재생하는 동안 새로운 후보·가중치 파일이나 행별 결과를 저장하지 않았다. 기본 평가 사전 전체가 `oof_fixed_deployment_threshold`와 같음을 검증했다. 표의 **상위3**은 임계값 적용 전 정답이 후보에 든 횟수, **예측**은 임계값·개수 제한 적용 뒤 해당 코드의 발현 횟수다. P는 TP/예측 수, R은 TP/정답 수다. `—`는 그 분모 0을 뜻한다. [저장 집계](../results/submission2_head.json) · [코드별 지표식](../kwater_lab/metrics.py#L188)

| 위험 코드 | 전체 1,045장 양성 | 탐색 양성(양성 접두사 수) | 상위3 | 예측 | TP/FP/FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fall_height` | 224 | 34 (6) | 28 | 52 | 14/38/20 | .269 | .412 | .326 |
| `slip_trip` | 317 | 56 (6) | 51 | 84 | 45/39/11 | .536 | .804 | .643 |
| `caught_in_machine` | 48 | 3 (2) | 1 | 1 | 0/1/3 | .000 | .000 | .000 |
| `struck_by_lifting` | 72 | 19 (6) | 10 | 22 | 5/17/14 | .227 | .263 | .244 |
| `electrical_energy` | 47 | 1 (1) | 0 | 0 | 0/0/1 | — | .000 | .000 |
| `drowning_inundation` | 1 | 0 (0) | 0 | 0 | 0/0/0 | — | — | — |
| `fire_explosion` | 44 | 1 (1) | 0 | 0 | 0/0/1 | — | .000 | .000 |
| `confined_space` | 1 | 0 (0) | 0 | 0 | 0/0/0 | — | — | — |
| `chemical_exposure` | 19 | 0 (0) | 0 | 0 | 0/0/0 | — | — | — |
| `thermal_weather` | 1 | 0 (0) | 0 | 0 | 0/0/0 | — | — | — |
| `other` | 168 | 35 (6) | 31 | 57 | 25/32/10 | .439 | .714 | .543 |

**빈 경보와 틀린 경보.** 전체 170장의 예측 코드 개수 0/1/2/3은 **51/50/41/28장**이다. 정답보다 많은 사진 74, 적은 사진 19, 같은 개수 77이고 집합까지 완전히 같은 사진은 53장이다. 위험 104장 중 빈 예측 15장, 코드를 냈어도 정답과 겹치지 않은 사진 26장, 합계 **41/104장**에서 모든 정답을 놓쳤다. 미관측 66장 중 30장(45.5%)에 경보를 냈다. 24장은 하나, 6장은 둘을 냈다. 임계값 이상 코드가 4개 이상인 사진은 **절단 전 7장**, 출력 뒤 0장이다. 이 7장은 형식 오류가 아니다.

**오류의 위치.** 상위 3개 안에서 임계값이 제거한 정답은 추락 14, 걸림 6, 기타 6, 인양 5, 끼임 1건이다. 상위 3개 밖에서 놓친 28건 중 인양 9, 추락 6, 걸림 5, 기타 4, 끼임 2, 감전·화재 각 1건이다. 전체 216개 출력 중 TP 89, FP 127이다. 이 중 무위험 사진에서 난 FP는 36개(걸림 13, 추락 9, 기타 9, 인양 5)이고 **위험 사진에서 잘못 추가한 코드가 91개**다. 따라서 장면 판별을 완벽하게 해도 코드 혼동이 남는다. 실제 정답 장면으로 무위험 66장의 경보만 강제로 끄고 위험 사진의 출력을 그대로 두는 *진단용 오라클*에서 F1은 0.27893이다. 이 값은 해당 고정 출력의 완벽한 음성사진 차단 효과를 계산한 것으로 새 모델의 기대값이 아니다.

**사건·자료 출처와 희소성.** 전체 1,045장의 위험 인스턴스는 942개, 탐색 170장은 149개다. 접두사 `ADD` 67장에 화재 37/44, 화학 16/19가 몰려 있고, `SBE` 29장 모두에 감전이 있다(전체 47장 중 29). 두 묶음에는 미관측 사진이 없다. 위 170장에는 둘 다 들어가지 않았다. 감전·화재의 유일한 탐색 양성이 검증에 들어간 외측 폴드에서 해당 코드 점수는 정확히 0이다. 끼임은 양성 3개 중 1개가 이미 후보에 들어와 있어 미탐 원인을 하나로 단정하기 어렵다. `ADD`·`SBE`는 파일 식별자 유형이고 사건·장소 라벨이 아니다. 출처 편중을 드러내는 보조표에만 쓰며 예측기에 입력하지 않는다. [원본 감사](../results/train_audit_summary.json) · [접두사 선택](../kwater_lab/prefix_pilot.py#L26)

**오경보와 중대한 미탐.** 같은 고정 0.20에서는 중대성 4인 위험 65건 중 33건을 맞혔고 해당 59장 중 28장에서 모든 중대성 4 코드를 놓쳤다. 위험성 곱 ≥13인 8건 중 5건 적중·3장 완전 미탐이다. 내부 가변 임계값의 해당 중대성 4 결과는 32/65와 29/59로 다르다. 별도 저장된 단일 CLIP의 사진별 예측도 다시 채점했다. `0.5 고정 / 내부 F1 선택 / 내부 무위험 오경보 ≤20%`에서 F1은 각각 **0.19503 / 0.22291 / 0.18899**, 무위험 오경보는 **13 / 31 / 9**장, 위험 빈 예측은 **37 / 19 / 41**장이었다. 접두사를 통제해도 경보 비용에 따라 정책 선택은 바뀐다. [중대 위험 정의](../kwater_lab/encoder_fusion.py#L21) · [가변 임계값 집계](../results/encoder_fusion.json) · [단일 CLIP 기록](../results/prefix_pilot_fpr20.json)

## 5. 데이터 분할과 선정 기준을 먼저 고정한다

1. **사건 지도.** 1,045장에 중복 화소 0쌍, dHash 거리 ≤4인 쌍 0, ≤12인 쌍 47이다. 최근접 후보를 합친 검토 대기열은 1,233쌍이며 접두사 간 후보가 1,184쌍이다. PNG 1,045장 모두 EXIF·원 사건 필드가 없다. 비슷하지 않은 사진도 같은 작업·장소의 전후 사진일 수 있다. 승인된 **학습 자료** 안에서 원사고-사진 대응표나 신뢰할 수 있는 사례 근거를 확인하고, 라벨을 가린 채 동일 장면·구조물·크롭·연속사진을 검토한다. `sample_id → incident_id → site_id(확인될 때만)`와 근거·미확정 상태를 기록한다. 미확정 관계는 하나의 보수적 묶음으로 격리하거나 독립 비교 대상에서 제외한다. 같은 장소를 재방문한 사례는 별도 장소 통째 제외 민감도 분석을 한다. [그룹 대기열](../results/train_audit_review_queue.json) · [사건 식별자 조사](../docs/official-incident-provenance.md)
2. **보류 봉인.** 탐색·공개·키트와 겹치는 등록 182장 및 그 사진과 연결된 사건 모두를 최종 보류에서 제외한다. 1,045−182=863은 제외 전파 전 후보 수다. 새로 검토·튜닝에 노출된 행도 등록한다. 사건 배정은 모델 출력과 분리한 담당자가 수행하고, 양성 수를 맞추기 위해 사건을 쪼개지 않는다. 기존 `audit.assign_groups`는 사진 수를 맞추므로 그룹 할당 이후 개발측의 코드별 양성 사건 수·무위험·출처 구성을 점검한다. 최종 보류 목록은 해시와 접근 기록을 고정하고, 모델·보정·텍스트·포맷 후보와 분석 계획을 동결한 뒤 한 번만 개봉한다. [노출 제외 빌더](../kwater_lab/holdout_exclusions.py#L16) · [감사 분할](../kwater_lab/audit.py#L132)
3. **개발 OOF.** 검증된 개발 사건에 외측 5겹을 우선 시도하되 독립 사건·양성 수가 부족하면 3겹 또는 leave-one-group-out을 사전 지정한다. 그룹 단위의 코드별 양성 벡터, 자료 계열, 미관측 수를 써서 편중을 줄이고 분포와 남는 불균형을 공개한다. `StratifiedGroupKFold`는 단일 클래스 라벨용이므로 다중라벨 행렬을 그대로 넘기지 않는다. 안쪽 그룹 OOF에서만 보정기·임계값·이웃·가중치·텍스트 인덱스를 정한다. 외측 평가 행에 입력하는 것은 사진뿐이며 사건·파일 식별자는 결합과 검증용이다. 각 사진은 한 번만 외측 예측을 갖는다. 사건이 검증되지 않으면 170장의 접두사 결과와 사후 가설만 보고한다. [그룹 분할 문서](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) · [모델 선택 편향 연구](https://jmlr.org/papers/v11/cawley10a.html)
4. **점수와 불확실성.** 전체 외측 예측을 이어 붙여 공식 합집합 Macro F1을 한 번 계산한다. 예측마다 활성 코드 집합·수와 TP/FP/FN/P/R, 양성 사진·사건 수, 미관측 경보율, 위험 빈 예측률, 중대성 4 완전 미탐, 코드 개수 분포, 원점수 상위3 재현과 최종 TP를 쓴다. 그룹별 B−A를 단순 평균하지 않고 **같은 사건을 양 후보에 쌍으로 복원추출**(시드 42, 2,000회), 각 재표집에서 합집합·F1을 다시 계산한다. 95% 백분위 구간과 정의된 반복 수, 희소 코드가 사라진 반복 비율을 병기한다. 사건 몇 개만 효과를 좌우하면 한 사건 제외 민감도와 양성 사건별 원계수를 남긴다. 기존 6개 **미확정** 접두사의 결합−단일 CLIP 차이 +0.02414, 참고 구간 [−0.01309, +0.06270]은 이 독립 검증을 대신하지 못한다. [부트스트랩 구현](../kwater_lab/metrics.py#L243) · [이전 결합 비교](../results/encoder_fusion.json)
5. **실험 선택 기준(팀 제안).** 이전 프로젝트 기준과 맞춰 새 사건 개발 OOF에서 직전 대조군 대비 Macro F1 **+0.02 이상**, 무위험 오경보율 증가 **3%p 이내**, 중대한 위험 완전 미탐 비증가를 1차 관문으로 둔다. 크기와 방향이 불안정해 쌍체 95% 구간이 0을 가로지르면 탐색으로 보류한다. 표본이 부족해 기준을 판단할 수 없으면 해당 코드/전체를 ‘미확정’으로 기록한다. 분모를 맞추려는 목적으로 코드 출력을 끄지 않는다. 동결된 단일 후보만 최종 보류의 같은 지표·형식·시간을 확인하며 보류 결과로 재튜닝하지 않는다. 이 수치들은 대회 규정이나 목표 향상폭 예측이 아니다. [기존 승격 기준](coordinated-safety-experiments.md)

## 6. 우선순위가 높은 세 가지 비교

### 1순위 · 동일한 모델로 개발 학습 자료 확대 효과 분리

**질문.** 화학은 0/19, 감전은 1/47, 화재는 1/44, 끼임은 3/48의 양성만 2차 헤드의 170장에 들어갔다. 접근 가능한 공식 학습 사건을 늘리면 같은 영상 특징으로도 이 네 코드의 유효 후보를 만들 수 있는가? `ADD`·`SBE`가 소수 양성 코드를 사실상 대표하므로 파일 형식·출처 배경을 외우는 것이 주요 실패 모드다. 실제 사례 수가 한 개인 3종은 재현율 승격을 유보하고 지원 수·실제 오경보만 별도로 본다.

**대조와 실행.** (1) 감사·배정이 끝난 *동일한 외측 검증 사건*을 정한다. 각 외측 학습측에서 A는 `기존 170 ∩ 학습 사건`, B는 `검토된 개발 학습 사건 전체`를 사용한다. 둘 다 외측과 같은 사건인 170장 사진을 제외한다. A와 B의 CLIP·SigLIP 특징, 1:1 평균, BCE/L2/250회, 0.20·top3를 고정한다. 자료량 비교 이후에만 내부 공통 임계값을 두 팔에 같은 방식으로 붙인다. (2) 출처별 무위험·양성 사건·예측률 및 자료 계열을 통째 뺀 개발 결과를 보조 분석한다. 출처만으로 코드를 내는 빈도 대조군과 영상 모델을 같은 외측 행에서 평가해 배경 편향을 확인한다. (3) 영점 클래스가 부활했는지와 함께 오경보로 활성 분모만 늘었는지 본다. [자료 선택 코드](../kwater_lab/submission2_fit.py#L28) · [현재 영상 특징 추출과 보류 차단](../kwater_lab/clip_probe.py#L162) · [고정 특징 검증 연구](https://proceedings.mlr.press/v139/radford21a.html)

**변경 지점과 비용.** `audit.audit/assign_groups`의 노출 제외 후 `encoder_pilot.cross_validate`가 검토된 사건 맵과 두 캐시를 받게 한다. `submission2_fit.main`의 170 고정·저장 경로는 접수 재현용으로 두고, 새 후보는 그 함수의 코드 순서·특징 검사를 재사용하는 별도 출력 경로에서 적합한다. 개발 학습/검증 ID만 추출하고 같은 이미지 특징을 폴드 간 캐시한다. 캐시 생성 1회·소형 선형 헤드 반복비용이 들며 배포 추론 횟수와 ZIP 모델 수는 기존과 같다. **최소 산출물**은 사건/노출 맵 해시, 사진·사건별 fold 분포, 공통 검증 ID, 양 팔 OOF 확률·예측과 11행 지표표다. 자료의 출처 분리가 무너지거나 준비 상태가 `ready`가 아니면 독립 성능 비교를 중단한다. 희소 4종 중 어떤 코드에서도 독립된 양성 TP가 늘지 않으면 ‘학습 자료 확대가 해당 코드를 복구했다’는 가설 채택을 보류한다.

### 2순위 · 무위험 판정과 코드별 보정을 분리해서 시험

**질문.** `fall_height`의 상위3 28건 중 절반만 출력을 통과한다. `slip_trip`은 재현율 .804이나 84개 경보 중 39개가 오경보다. 모든 코드에 같은 값을 적용하는 현재 정책은 이 두 오류에 같은 비용을 준다. 기존 무위험 FPR 20% 제한은 미탐을 늘렸고, 자유 개수 한계 F1 기여도 방식은 미관측 60/66장에 경보했다. 개별 임계값 11개를 작은 표본에서 자유롭게 찾으면 희소 양성 하나가 임계값을 결정할 수 있다. [기존 보정 교차실험](../results/evidence_factorial.json) · [F1 최적 임계값의 확률·빈도 의존성](https://arxiv.org/abs/1402.1892) · [정확히 k개를 다루는 선행 연구](https://arxiv.org/abs/2401.16594)

**최소 요인 실험.** 1순위에서 택한 *동일 훈련 풀·동일 전역 특징*으로 기준 **A**(원래의 비조건부 코드 점수, 내부 공통 임계값)와 기준 **A0**(접수 사양 .20)을 둔다. **B**는 A에 학습측 `status`로 만든 이진 장면 헤드의 중단 기준만 추가한다. **C**는 장면 헤드 없이 코드별 여유도 `logit(p_c)−logit(t_c)`를 보정해 양수 중 상위 3개를 낸다. **D**는 B와 C를 결합한다. B−A가 장면 억제, C−A가 코드 보정, D−B와 D−C가 각 요소의 조건부 기여다. 네 팔 모두 0~3개, 동률은 공식 코드 순서다. 동일 점수에서 임계값만 조정한 결과와, 조정된 여유도로 순위까지 바뀐 결과도 같은 입력에서 기록한다.

- 장면·코드 확률 보정은 각 외측 학습측의 **내부 OOF 점수**로만 적합한다. 내부 분할에서 공통 임계값을 먼저 고른 뒤 빈발 코드의 개별 임계값을 공통값 쪽으로 수축한다. 예를 들어 양성 **사건** 수 `g_c`에 따라 `w_c=g_c/(g_c+κ)`로 공통/개별 logit 임계값을 보간하고 `κ`는 소수의 사전 고정 후보에서 선택한다. 양성 0개는 공통 대체, 1개는 코드별 최적화 생략. 확률이 항상 0인 헤드에는 임계값 조정만으로 새로운 증거가 생기지 않는다. 보정 전후 Brier 점수·빈별 관측률·양성 사건 수를 함께 남긴다. [온도 보정 원 연구](https://proceedings.mlr.press/v70/guo17a.html) · [희소 이항 관측의 부분 풀링](https://mc-stan.org/learn-stan/case-studies/pool-binary-trials.html)
- D가 남을 때만 연속형 대조 **D2**를 추가한다. 별도로 학습한 `q=P(위험 장면|사진)`와 위험 사진 안에서 학습한 `r_c=P(코드 c|위험 장면,사진)`를 곱해 코드 확률을 만든다. 현행 무조건부 확률에 q를 또 곱하면 장면 정보를 두 번 적용할 수 있으므로 D2는 조건부 헤드를 별도 적합한다. D2의 대조는 같은 조건부 헤드의 `r_c` 단독과 D다. 임계값·장면 컷·풀링 강도는 외측 점수를 보고 바꾸지 않는다.
- `encoder_pilot.cross_validate` 내측 점수와 `train_head.calibration/decisions`를 변경 대상으로 삼는다. 채택된 정책만 `RiskEncoder.scores/run`에 이식한다. OOF 산출물이 있으면 ②의 경보 규칙은 CPU에서 재생 가능하고 장면 헤드는 기존 특징을 공유하므로 새 영상 패스가 필요 없다. **최소 산출물**은 팔별 같은 ID의 확률·경보 수, 끈 경보의 TP/FP, 장면 단위 FN, 구간별 보정표, 중대성 4 완전 미탐·쌍체 구간이다. 무위험 오경보만 줄고 중대 미탐이 늘면 장면 차단을 중단한다. [내측 학습](../kwater_lab/encoder_pilot.py#L84) · [반환 함수](../kwater_lab/train_head.py#L136)

### 3순위 · 지역 시각 증거가 위험과 방호 상태를 구별하는가

**질문.** 인양 양성 19건 중 9건은 2차 방식의 상위 3위 밖이다. 추락의 방호 난간, 인양물과 사람의 상대 위치는 작은 특징을 확대한 화면과 전체 맥락이 동시에 필요할 수 있다. 전체 영상만으로 병목의 원인을 확정할 수는 없다. 기존 36개 압축 공간 토큰의 단일 패치 블록은 상위3 125/149로 빈도만 쓰는 대조와 같았고, F1은 .1331/.1336(시드 2개)에 머물렀다. 합성 난간 유무 한 쌍에서 추락 점수 .7066→.6594였지만 양쪽 모두 경보였다. 검토된 공식 탐색 대조 26쌍 중 상태 대비로 쓸 수 있었던 쌍은 3개, 사람 검토는 0개다. [패치 대조](loop-transformer.md) · [방호 상태 쌍](official-state-contrasts.md) · [합성 민감도](../results/synthetic_basin_probe.json)

**실험과 반증.** 하나의 고정 SigLIP 2 384, 동일 그룹·헤드·보정·시드에서 (a) 전체 1뷰, (b) 같은 전체 화면 반복으로 입력 횟수를 맞춘 대조, (c) 전체+사전 지정 중첩 2×2 타일, (d) 다른 학습/검증 **같은 분할 안에서** 관련 없는 사진의 타일을 고정 순열로 교환하는 음성 대조를 둔다. 교환은 검증 시 모델이 실제 지역을 쓰는지 보는 개발 진단이며 배포 입력은 한 사진의 자른 영역뿐이다. `fit`은 사진 단위 weak label로 수행하고 각 타일에 사진 위험을 정답으로 복제하지 않는다. 초기에는 전역 점수와 지역의 상위 풀링을 소형 결합 헤드로 연결하고, 검증 개선이 있을 때만 학습형 pooling을 분리 비교한다. 동일 개수/순위·AP 대조와 별도로 원래의 **0~3 코드** 공식 지표를 확인한다. 인양·추락의 상위3/양성, 클래스 정밀도, 안전하게 방호된 구조물·정지 장비에서의 FP, 다른 부위로 바꿨을 때의 변화가 핵심이다. 교환 대조와 같거나 코드 빈도만으로 같은 효과를 얻으면 해당 영상 가설을 중단한다. [작은 물체를 위한 슬라이싱 원 연구](https://arxiv.org/abs/2202.06934) · [영상 단위 약한 감독의 다중 인스턴스 학습](https://proceedings.mlr.press/v80/ilse18a.html)

**변경 지점과 비용.** `ImageEncoder.encode`, `loop_pilot.patch_grid`의 패치/캐시 검증과 `encoder_fusion.average_scores`를 재사용한다. 최소 산출물은 뷰 좌표·전처리 해시, 폴드별 표, 관측 가능한 상태 대조의 판정·불확실 수, 실제 L4 설치·메모리·화질별 지연 측정이다. CLIP 전역 1회는 유지하고 SigLIP을 전체+4타일로 돌리면 후자의 영상 호출 수는 1→5회다. 전체 지연 배수는 미측정이다. 전체 다중뷰가 부족한 개선에 비해 느리면 안쪽 검증에서 고정한 불확실 구간에만 추가 뷰를 쓰는 단일 후속 대조를 검토하고, 최악의 전 사진 호출도 8절 시간 기준을 만족해야 한다. [입력 변환](../kwater_lab/clip_probe.py#L104) · [패치 추출](../kwater_lab/loop_pilot.py#L33) · [결합](../kwater_lab/encoder_fusion.py#L16)

## 7. 정성 서술·희소 위험에서 이미 해 본 비교

| 대안 | 선행 관측 | 다음에 허용할 비교와 중단 기준 |
|---|---|---|
| 학습 주석을 이용한 영상-근거 정렬 | SigLIP 단일 계열에서 진짜 학습 근거/같은 코드 안 문장 교환/이미지 단독의 상위3은 121/117/116건(분모 149). 정합 근거와 이미지 단독의 공통 임계값 F1은 .2442 대 .2438, 미관측 오경보 34 대 29. 4절의 2인코더 121건은 다른 모델이다. [요인 실험](../results/evidence_factorial.json) | 3순위에서 실제 지역 단서의 기여가 먼저 나타날 때만 학습측 `target.hazards[c].evidence`와 타일 표현을 정렬한다. 같은 코드·같은 출처 안에서 문장을 바꾼 대조, 손실 제거 대조, 사진만 쓰는 추론을 비교한다. 사진별 근거를 섞어도 성능이 유지되면 텍스트 감독을 채택하지 않는다. 훈련 시만 보조 신호라 배포 텍스트 인코더 비용을 없앨 수 있다. [기존 함수](../kwater_lab/evidence_factorial.py#L44) · [특권정보·증류 연구](https://arxiv.org/abs/1511.03643) |
| 희소 양성 가중치·외부 자료 | 기존 비대칭 손실은 170장 내부 F1 선택에서 약 .240이나 미관측 경보 53/66. 희소 CLIP 문장 순위에서는 양성이 없는 170장 중 밀폐공간 72장이 상위3에 들었다. 개별 인양 장비·물·관로의 존재와 위험 노출은 다르다. [손실 비교](../results/loss_pilot.json) · [희소 후보](../results/rare_candidate_pilot.json) | 48/47/44/19장의 끼임·감전·화재·화학은 1·2순위 후 여전히 누락되면 제한된 비용 가중치를 BCE와 비교한다. 1장 세 코드에는 코드별 비용 확대보다 다빈도 코드를 0·1·5개 양성 사건으로 가린 모의 희소 대조를 쓴다. 모의 결과는 실제 희소 재현율과 별도다. 외부 프레임 사용 시 권리·신고, 동영상/현장 분리, 실제 공식 개발 사건으로의 전이가 모두 필요하다. 배경만 같은 음성에서 경보가 늘면 중단한다. [모의 희소 선행 결과](../results/compositional_pilot.json) · [비대칭 손실 원 연구](https://arxiv.org/abs/2009.14119) |
| 이미지에 맞춘 서술과 오프라인 법령 | 2차 코드는 정해진 텍스트를 반환한다. 1차 배율 1.0의 실제 18장 생성에서 정답 인용 위험 23건 중 코드와 첫 조문 동시 적중은 코드별 최빈 10, 조문 본문 검색 1, 폴드 사례 검색 10, 조치 재정렬 2건. 최빈·사례 모두 틀린 위험에도 5건 인용했다. 사진별 예측을 완벽한 정답 코드·근거로 바꾼 147건의 조회 실험과 구별한다. [생성문 재검색](../results/generated_law_comparison.json) · [정답 질의 파일럿](../results/law_examples_pilot.json) | 동일한 **예측 코드와 사진**을 고정하고 (a) 접수 템플릿 (b) 사진에서 확인한 위치·대상·상태만 서술 (c) 동일 b에 학습측 사례+법령 본문 검색을 덧붙인다. `attach_law.prediction_with_article`에서 예측 조치를 입력하고 인용이 지지되지 않으면 비워 둔다. 합법 ID·조치 지지·잘못된 코드에 붙인 인용·형식·생성 실패를 별도 집계한다. 블라인드로 근거·조치·출처와 과잉 단정을 평가하고, 순서 무작위/동점 허용 및 평가자 간 합의도를 남긴다. 18장 블라인드 문구 26쌍에 기존 사람 평정 0건으로, 공식 30점을 예측할 근거가 없다. 이미지 기반 b의 독립 우위가 확인될 때만 생성 비용을 부담한다. [오프라인 후처리](../kwater_lab/attach_law.py#L58) · [기존 평정 현황](../results/preference_pilot.json) · [검색 결합 연구](https://arxiv.org/abs/2005.11401) |

생성 모델을 비교한다면 단지 코드 이름을 11개 열거하는 프롬프트와 학습·내부 검증·최종 출력까지 11종을 보존하는 경로를 구분한다. 고정된 6장 프롬프트 실험은 30번 생성 중 28행 유효로 2건이 출력 한도에서 잘렸다. 공식 LoRA 배율 1.00→0.75는 같은 18장 36행 모두 유효했지만 완전 일치가 9→6장으로 줄었다. 두 실험에서 확인한 것은 원출력·형식·탐색 라벨 일치이고, 독립 VLM 우위나 공식 LLM 선호는 미측정이다. [프롬프트 결과](../reports/full-image-prompt-pilot.md) · [어댑터 결과](../reports/official-lora-scale-pilot.md)

## 8. 일정, 제출 계약, 중단 시점

| KST · 제안 일정 | 선행조건과 수행 순서 | 다음 단계로 넘길 최소 증거 |
|---|---|---|
| 11/26~11/28 | 사건/장소 후보 검토, 노출 확장, 분할과 기준 코드 고정. 근거 불충분한 기간에는 기존 170장 재생만 사용 | `audit.status`, 그룹 해시·보류 봉인 시점, 지원 사건 수, 전처리·OOF 사양. 출처를 확인할 수 없다면 170장 결과를 독립 검증으로 승격하지 않는다. |
| 11/29~12/02 | 검토된 개발 풀에서 ① 자료량 비교 → ② A~D 보정 비교. 캐시·입력 ID·라벨 순서가 같은지 비교 | 11개 클래스 원계수, 활성 합집합, F1 차이와 사건별 구간, 오경보·중대 미탐·단계별 누락. 보정 후보는 선정 기준 통과 시만 동결한다. |
| 12/03~12/06 | 누락이 순위에서 지속되는 코드에만 ③ 다중뷰를 시험. 정성 서술은 코드 고정 짝으로 별도 비교. 외부자료 선택은 공식 신고 마감 전에 판정 | 전체·지역·교환 대조, 코드별 부위 교란 민감도, 새 자료 이용/신고 근거. 외부자료가 없어도 ①~③은 공식 학습자료만으로 설계 가능하다. |
| 12/07~12/10 | 검증 후보 1개와 대조군 고정. 실제 배포 경로에서 오프라인 리허설, 정성 눈가림, 입력 크기·경보 수에 따른 지연 확인 | 새 ZIP 해시, assets·라이선스, 모든 행 유효, 동일 seed 재실행 동일 출력, 설치 포함 200·300장 전체 벽시계·메모리. |
| 12/11~12/12 | 고정 후보로 봉인한 독립 학습자료 보류셋 1회 평가. 미확정 희소 재현율·지원 그룹을 그대로 기록 | 기준을 통과한 후보 또는 기존 대조군 유지 결정, 사용 버전과 원장. 대회 공식 지정 마감은 12/13 10:00이다. |

**시간 판단.** 2차의 로컬 CPU 12장·36회 측정은 예열 후 중앙값 0.456초, p90 0.663초, 가중치 검사·세션 로드 3.52초였다. 압축 사진 메모리 입력부터 코드 반환까지였고 디스크 I/O·설치·VLM은 제외했다. 공식 1차 제공 설명은 L4에서 Public 약 15분이다. 프로젝트의 RTX 3060/Windows 첫 실행 2,035.4초는 다른 환경이다. 새 패키지는 실제와 같은 L4 오프라인 환경에서 `설치+기동+N장 추론+저장`을 재고 200/300장을 각각 검증한다. 허용된 개발 입력을 익명 manifest 구조로 반복한 부하 검사는 시간 측정에만 쓴다. 전체 실측 또는 환경 여건이 없으면 ‘L4 미검증’으로 남긴다. [결합 모델 시간 원장](../results/fusion_timing.json) · [공식 모델 실행 환경](official-baseline-windows.md) · [공식 모델 제공 설명](https://hackathon.imaginationgroup.co.kr/kwater/#core-data)

**키트 명령.** 이미 제공된 키트 SHA-256 `28acc73fab6a7605361866f87825bd00740b1aacd9bfe35996bdec542dbc37c2`, 2,713,157바이트를 실제 파일에서 대조했다. 아래는 Linux/WSL2의 *후속 리허설 예시*다. `candidate.zip`과 `train-load-*`은 이후 준비할 개발용 경로다. 키트 pinned 이미지의 기본 제한은 1,800초이므로 300장 시험에서는 명시적으로 바꾼다. 각 결과의 출력 폴더에 JSONL 한 파일, 유효 N/N, 타임아웃 없음, L4 최대 메모리 ≤24 GB를 확인한다. 초과하면 후보에서 제외한다. [키트 실행기](../data/local-kit/run_local.sh#L9) · [공식 크기·시간](https://hackathon.imaginationgroup.co.kr/kwater/#submission)

```bash
cd /mnt/g/Projects/KWater-SafetyAI/data/local-kit
KWATER_TIME_LIMIT=1800 bash run_local.sh /path/to/candidate.zip /path/to/train-load-200
KWATER_TIME_LIMIT=3600 bash run_local.sh /path/to/candidate.zip /path/to/train-load-300
```

## 부록 · 실행한 확인 명령과 재현

원본 G 드라이브 루트에서 실행한다. 저장 응답 없이 공식 규정과 현황을 확인할 수 있다. 마지막 줄은 입력·출력을 바꾸지 않고 이미 생성된 예시 10행을 재검사한다. 시스템 Python 3.14에는 `jsonschema`가 없어서 프로젝트에 이미 설치된 baseline 가상환경의 검사기를 사용했다. 결과는 `유효 결과 10/10`, exit 0이었다.

```powershell
Set-Location 'G:/Projects/KWater-SafetyAI'
git --no-optional-locks status --short --branch
git worktree list --porcelain
git diff --stat 12f5fc6c e4af204e
(Invoke-RestMethod https://hackathon.imaginationgroup.co.kr/kwater/api/report) | Select-Object revision,updatedAt
(Invoke-RestMethod https://hackathon.imaginationgroup.co.kr/kwater/api/leaderboard) | Select-Object generatedAt,phase,sample,privateRevealed,@{n='teams';e={$_.teams.Count}}
Get-FileHash data/submissions/kwater-fusion-20261125.zip,data/packages/kwater-local-kit.zip -Algorithm SHA256
& ./.runtime/baseline-venv/Scripts/python.exe -B -X utf8 data/local-kit/kit/check_output.py data/local-kit/sample-input .runtime/submission2-smoke
```

다음 PowerShell 블록은 4절을 재현한다. Python 3.14.3·NumPy 2.4.4, CPU 1스레드, 170장 특징 두 개와 기존 정답·사양만 사용한다. ZIP이나 OOF 행을 덮어쓰는 학습 스크립트의 `main`은 호출하지 않는다. 출력 첫 줄의 순서는 *상위3 적중 / 최종 TP / 정답*이며 **121 / 89 / 149**가 기대값이다. `assert`는 재생한 11종별 집계까지 기존 원장과 다르면 중단한다.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
@'
import json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from kwater_lab.metrics import CODES, evaluate
from kwater_lab.train_head import fit, probabilities, decisions

def read(path):
    return json.loads(Path(path).read_bytes())

ref = read("results/submission2_head.json")
cohort = read("data/prefix-pilot/exposure.json")
audit = read("results/train_audit_full.json")
assert cohort["labels_sha256"] == audit["labels_sha256"]
assert not audit["exact_pixel_duplicates"] and not audit["cross_group_near_pairs"]
ids = sorted(cohort["exposed_ids"])
assert len(ids) == len(set(ids)) == 170
group = {k:g for g, sub in cohort["by_prefix"].items() for k in sub}
gold = {k:set(audit["inventory"][k]["codes"]) for k in ids}
y = np.array([[c in gold[k] for c in CODES] for k in ids], dtype=float)
x = []
for name, fn in [("clip","features.npz"),("sig","siglip2-base16-384-features.npz")]:
    path = Path("data/prefix-pilot") / fn
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ref["feature_sha256"][name]
    with np.load(path, allow_pickle=False) as z:
        pos = {k:i for i,k in enumerate(z["ids"].tolist())}
        assert set(pos) == set(ids)
        x.append(z["features"][[pos[k] for k in ids]])
scores = np.zeros(y.shape)
for g in cohort["by_prefix"]:
    v = np.array([group[k] == g for k in ids])
    scores[v] = sum(probabilities(fit(a[~v], y[~v], .01, 250), a[v]) for a in x) / 2
pred = decisions(ids, scores, .20)
ev = evaluate(gold, pred, group)
assert ev == ref["oof_fixed_deployment_threshold"]
rank = np.argsort(-scores, axis=1, kind="stable")[:, :3]
in3 = np.zeros_like(y, dtype=bool)
np.put_along_axis(in3, rank, True, axis=1)
tp = in3 & (scores >= .20) & (y == 1)
print("top3 / emitted TP / gold", int((in3*y).sum()), int(tp.sum()), int(y.sum()))
for i,c in enumerate(CODES):
    m=ev["per_code"][c]
    print(c,m["support"],m["predicted"],m["tp"],int((in3[:,i]*y[:,i]).sum()),
          int(((scores[:,i]==0)&(y[:,i]==1)).sum()))
print("threshold>=4", int(((scores>=.20).sum(1)>3).sum()))
print("no-correct / empty on hazard", sum(bool(gold[k]) and gold[k].isdisjoint(pred[k]) for k in ids), ev["hazard_completely_missed"])
print("over / under / equal-card", Counter(int(np.sign(len(pred[k])-len(gold[k]))) for k in ids))
for k in sorted(cohort["by_prefix"]):
    sub=cohort["by_prefix"][k]
    e=evaluate({i:gold[i] for i in sub},{i:pred[i] for i in sub})
    print(k,len(sub),round(e["macro_f1"],4),e["no_hazard_false_alarms"],e["no_hazard_photos"])

oracle={k:(pred[k] if gold[k] else set()) for k in ids}
print('oracle scene',evaluate(gold,oracle,group)['macro_f1'])
print('rows before/after/gold')
for m in (0,1,2,3):
 print(m, [sum(len(gold[k])==m and len(pred[k])==i for k in ids) for i in (0,1,2,3)])
zeros=Counter(c for k in ids if not gold[k] for c in pred[k])
print('FP on no-hazard',zeros)
levels = [json.loads(line) for line in Path("data/train/train.jsonl").read_text(encoding="utf8").splitlines()]
for name in ('sev4','risk13'):
 targets={r['sample_id']:{c for c,h in r['target']['hazards'].items() if (h['risk_level']['severity']['score']==4 if name=='sev4' else h['risk_level']['risk']['score']>=13)} for r in levels if r['sample_id'] in group}
 targets={k:v for k,v in targets.items() if v}
 print(name,len(targets),sum(map(len,targets.values())),sum(len(pred[k]&v) for k,v in targets.items()),sum(pred[k].isdisjoint(v) for k,v in targets.items()))

archive = read("data/prefix-pilot/diagnostic_fpr20.json")["paired_out_of_prefix_predictions"]
for arm in ("fixed_0.5", "inner_calibrated", "inner_fpr20"):
    archived = {row["sample_id"]: set(row["linear_head_"+arm]) for row in archive}
    e = evaluate(gold, archived, group)
    print("CLIP", arm, e["macro_f1"], e["no_hazard_false_alarms"], e["hazard_completely_missed"])
'@ | python -B -X utf8 -
```

새로운 사건 지도, 양성이 충분한 희소 코드의 분리 검증, L4 설치 포함 실행, 사람·운영사 정성점수는 이번 리뷰에서 확보되지 않았다. 이 네 항목이 채워져야 후보를 독립 성능과 배포 조건으로 비교할 수 있다.
