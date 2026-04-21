# Terminal Agent 평가 계획서

## 1. 평가 개요

NVIDIA Nemotron-Terminal-Corpus 데이터셋으로 SFT한 모델을 **Terminal-Bench 2.0** 기준으로 평가한다.
Terminal-Bench 2.0은 터미널 환경에서 에이전트가 실제로 명령을 실행해 과제를 해결하는 벤치마크로,
모델의 터미널 task 해결 능력을 정량 평가하는 업계 표준이다.

- **최종 평가**: https://www.tbench.ai/leaderboard/terminal-bench/2.0
- **실행 방법**: `harbor run -d terminal-bench@2.0 -a "agent" -m "model" -k 5`
- **평가 지표**: Accuracy (task 성공률), 5-repeat 평균 ± 표준편차

## 2. Terminal-Bench 2.0 구조

### 2.1 태스크 구성
총 **89개 태스크**, 16개 카테고리 분포:

| 카테고리 | 태스크 수 | 난이도 | 비고 |
|----------|----------|--------|------|
| software-engineering | ~15 | easy~hard | 빌드, 컴파일, 구현 등 |
| security | ~7 | medium~hard | XSS, 암호분석, 포렌식 |
| scientific-computing | ~7 | medium~hard | 수치해석, 생물학, 물리 |
| system-administration | ~5 | medium~hard | 서버설정, 이메일, VM |
| debugging | ~5 | easy~medium | 버그수정, 호환성, 복구 |
| data-processing | ~4 | medium | 로그분석, ETL, 정규식 |
| data-science | ~4 | medium~hard | ML, 검색, 통계모델링 |
| file-operations | ~4 | medium~hard | 복구, 압축, 데이터추출 |
| machine-learning | ~4 | medium~hard | 학습, 추론, 스케줄링 |
| model-training | ~3 | medium~hard | PyTorch, 모델복구, FastText |
| mathematics | ~3 | medium~hard | 암호학, 고유값, 분포 |
| optimization | ~1 | medium | 포트폴리오 최적화 |
| data-querying | ~1 | hard | SPARQL 쿼리 |
| video-processing | ~1 | hard | 비디오 분석 |
| games | ~1 | medium | 체스, CoreWars |
| personal-assistant | ~1 | medium | 일정관리 |

### 2.2 평가 특징
- 에이전트가 실제 Docker 컨테이너에서 명령 실행
- 파일 생성/수정 결과를 자동 검증 (test script 기반)
- 멀티턴 대화 + 실시간 터미널 출력 피드백
- thinking/reasoning 능력 필수 (복잡한 디버깅, 알고리즘 구현)

### 2.3 리더보드 참고치 (2026-04 기준)
| 모델 | 정확도 |
|------|--------|
| GPT-5.4 (ForgeCode) | 81.8% |
| Claude Opus 4.6 (ForgeCode) | 81.8% |
| Gemini 3.1 Pro (TongAgents) | 80.2% |
| Qwen3-Coder 480B | 23.9% |
| Nemotron-Terminal-32B (논문) | 27.4% |

## 3. 학습 데이터셋

### 3.1 원본: NVIDIA Nemotron-Terminal-Corpus
- **총량**: ~366,154 샘플 (7.7 GB)
- **위치**: `/dataset/`
- **구성**:
  - `dataset_adapters/` (~226k): Math, Code, SWE 데이터셋을 터미널 형식으로 변환
  - `synthetic_tasks/skill_based/` (~140k): 10개 도메인 × 3개 난이도 (easy/medium/mixed)

### 3.2 학습용 데이터 (평가 샘플 제거 후)
- **총량**: ~366,104 샘플
- 평가 샘플 50개가 각 parquet 파일에서 제거됨

### 3.3 스키마
```
conversations: list[{role: str, content: str}]  # 멀티턴 대화
agent: str           # 에이전트 이름
model: str           # 사용된 모델
task: str            # 태스크 식별자
episode: str         # 에피소드 ID
enable_thinking: bool # thinking 토큰 사용 여부
source: str          # (dataset_adapters만) 원본 출처
```

## 4. 내부 평가 데이터셋

### 4.1 개요
Terminal-Bench 2.0 최종 평가 전, 학습 과정에서 모델 성능을 추적하기 위한 **내부 평가셋**.

- **샘플 수**: 50개
- **위치**: `/eval/eval_dataset.jsonl`
- **형식**: JSONL (각 행 = 1개 trajectory)

### 4.2 선정 기준

Terminal-Bench 2.0의 카테고리 분포에 **비례하게** 샘플을 선정했다.
medium 난이도를 주축으로, mixed(복합) 난이도를 보조로 사용했다.
easy 난이도는 제외했다 — Terminal-Bench 2.0 태스크가 대부분 medium~hard이기 때문이다.

| 카테고리 | 샘플 수 | 출처 (난이도) | 선정 이유 |
|----------|---------|---------------|-----------|
| software_engineering | 6 | medium | TB2.0 최다 카테고리. 빌드, 구현, 컴파일 등 핵심 |
| debugging | 4+2=6 | medium + mixed | 버그 수정은 터미널 에이전트 핵심 역량 |
| data_science | 3+1=4 | medium + mixed | ML/통계 관련 실무 task |
| data_processing | 3+2=5 | medium + mixed | ETL, 로그분석 등 실사용 빈도 높음 |
| scientific_computing | 3+1=4 | medium + mixed | 수치해석, 생물학 등 복합 domain |
| security | 3+1=4 | medium + mixed | XSS, 암호학, 포렌식 |
| file_operations | 3+1=4 | medium + mixed | 복구, 압축, 데이터 추출 |
| system_administration | 3 | medium | 서버 설정, git, 서비스 구성 |
| data_querying | 2 | medium | DB 쿼리, 데이터 검색 |
| dependency_management | 2 | medium | 패키지 설치, 의존성 해결 |
| model_training | 2 | medium | 모델 학습/파인튜닝 task |
| code (adapters) | 3 | - | 실제 코드 문제 → 터미널 풀이 |
| swe (adapters) | 3 | - | 소프트웨어 엔지니어링 실과제 |
| math (adapters) | 2 | - | 수학 → 터미널 명령으로 해결 |

### 4.3 왜 이 샘플들인가
1. **비례 분포**: TB2.0의 실제 카테고리 비율을 반영 — 소프트웨어 엔지니어링이 가장 많고, 보안/과학계산/디버깅이 그 다음
2. **난이도 매칭**: easy 제외, medium~mixed 중심 — TB2.0 태스크 난이도와 정렬
3. **다양성**: 20개 파일에서 각각 독립적으로 무작위 추출 — 특정 패턴에 과적합 방지
4. **독립성**: 학습 데이터에서 완전 제거 — 평가 누수(leakage) 방지

### 4.4 내부 평가 방법

학습 중 체크포인트별로 아래 metric을 측정:

```
metric = (정확히 완료된 태스크 수) / (전체 평가 태스크 수) × 100
```

평가는 각 trajectory의 마지막 assistant 응답을 기준으로:
- **자동 평가**: 최종 응답에 명령어 실행 결과가 포함되어 있는지, task 목표 달성 여부
- **정성 평가**: 대화 품질, 명령어 적절성, 에러 복구 능력

## 5. 최종 평가 프로세스

### 5.1 1단계: 내부 평가 (현재)
- 50개 내부 평가셋으로 학습 과정 모니터링
- 체크포인트별 성능 추적

### 5.2 2단계: Terminal-Bench 2.0 제출
- 내부 평가에서 성능이 수렴하면 TB2.0에 제출
- `harbor run -d terminal-bench@2.0` 으로 89개 태스크 평가
- k=5 반복으로 안정적인 accuracy 측정

### 5.3 타겟 성능
| 모델 크기 | Base → 타겟 | 참고 (Nemotron 논문) |
|-----------|-------------|---------------------|
| 8B | 타겟 > 10% | 논문: 2.5% → 13.0% |
| 14B | 타겟 > 15% | 논문: 4.0% → 20.2% |
| 32B | 타겟 > 25% | 논문: 3.4% → 27.4% |

## 6. 참고 자료

- Terminal-Bench 리더보드: https://www.tbench.ai/leaderboard/terminal-bench/2.0
- Nemotron-Terminal-Corpus: https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus
- 논문: Pi et al., "On Data Engineering for Scaling LLM Terminal Capabilities" (arXiv:2602.21193)
