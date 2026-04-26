# DeepSeek V4 Full Eval 중단 기록 (2026-04-25)

## 결론

이번 DeepSeek V4 평가는 **모델 변환과 smoke test까지만 성공**했고,  
`tb2_lite` full replay 평가는 **중단**했습니다.

중단 이유는 간단합니다.

- 8x H200을 붙여도 속도가 너무 느림
- VRAM을 거의 못 채우는 구조라 메모리 증설로 해결되지 않음
- 배치 크기를 크게 올리면 오히려 첫 배치 완료가 늦어짐
- `Flash`도 이 정도면 `Pro`는 더 비실용적임

즉, **현재 공식 DeepSeek inference code path로는 full replay eval이 시간 대비 가치가 낮다**고 판단했습니다.

## 완료된 항목

- `Flash` convert 완료
  - output: `/home/work/deepseek_models/DeepSeek-V4-Flash-mp4`
- `Pro` convert 완료
  - output: `/home/work/deepseek_models/DeepSeek-V4-Pro-mp8`
- DeepSeek 전용 evaluator / progress / shard merge 코드 작성 완료
- `Flash` smoke run 1개 성공

## smoke 결과

완료된 유일한 정식 결과물은 아래 smoke run 입니다.

- 결과 파일:
  - [DeepSeek-V4-Flash.json](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/smoke_deepseek_flash_20260425/DeepSeek-V4-Flash.json)

핵심 수치:

- `Load`: `13.1s`
- `Gen`: `255.3s`
- `Avg sec/step`: `255.251`
- `Cmd F1`: `0.3767`
- `First Cmd Exact`: `0.0%`
- `Score`: `26.37`

중요:

- 이 값은 **1개 샘플 smoke test**라서, 정상 실행 확인용이지 본평가 점수로 쓰면 안 됩니다.

## full replay 시도 기록

시도한 run 경로:

- [20260425T202900Z_deepseek_flash_full](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260425T202900Z_deepseek_flash_full)
- [20260425T212100Z_deepseek_flash_full_bs8_len16384](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260425T212100Z_deepseek_flash_full_bs8_len16384)
- [20260425T212350Z_deepseek_flash_full_bs24_len16384](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260425T212350Z_deepseek_flash_full_bs24_len16384)
- [20260425T212920Z_deepseek_flash_full_bs128_len16384](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260425T212920Z_deepseek_flash_full_bs128_len16384)
- [20260425T215000Z_deepseek_flash_full_bs8_progress](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260425T215000Z_deepseek_flash_full_bs8_progress)
- [20260425T220000Z_deepseek_flash_full_bs8_sorted](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260425T220000Z_deepseek_flash_full_bs8_sorted)
- [probe_flash_bs8_len16384](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/probe_flash_bs8_len16384)

실제 마지막까지 가장 의미 있게 간 run은 아래입니다.

- [20260425T220000Z_deepseek_flash_full_bs8_sorted](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260425T220000Z_deepseek_flash_full_bs8_sorted)

이 run의 중단 직전 상태:

- `completed_steps`: `56 / 386`
- `progress_pct`: `14.5%`
- `approx_completed_steps`: `65.7 / 386`
- `approx_progress_pct`: `17.0%`
- `elapsed_gen_sec`: `3315.8s`
- `eta_sec`: `16165.2s` (`약 4시간 29분 추가`)

shard별 상태:

- shard 0:
  - `24 / 193` 완료
  - `approx 30.95 / 193`
- shard 1:
  - `32 / 193` 완료
  - `approx 34.75 / 193`

## 속도/자원 사용 해석

중단 직전 대략적인 자원 사용은 이랬습니다.

- GPU util: `약 87~90%`
- VRAM: `약 44GB / 143.8GB` (`약 31%`)

즉 문제는:

- GPU는 돌고 있었음
- 하지만 VRAM은 많이 남았음
- 그럼에도 속도는 여전히 느렸음

이건 단순히 배치를 더 키우면 해결되는 문제가 아니었습니다.

## 왜 VRAM을 다 못 썼는가

현재 경로는 `vLLM`이 아니라 DeepSeek 공식 inference code path 입니다.

또한 모델 구조 자체가:

- `token-by-token decode`
- `sliding window`
- `KV cache compression`

을 사용해서, 작은 vLLM 모델처럼 VRAM을 밀어넣는 방식이 잘 안 나옵니다.

실제로 확인한 문제:

- `batch=24`, `batch=128`까지 올려도 근본 해결이 안 됨
- 오히려 **첫 배치 완료 시점이 더 늦어짐**
- `%`가 오랫동안 `0`으로 보이는 비효율이 발생

## 추가로 한 최적화

다음 패치까지 넣고 다시 돌려봤습니다.

- shard progress/status 파일 기록
- 배치 내부 token 진행률 기록
- prompt 길이 정렬 후 batching
- batch / max_model_len 조절
- 8 GPU 전체 사용

그 뒤에는 실제로 진행률이 잡히기 시작했고, `sorted bs=8`이 가장 현실적이었습니다.  
그래도 여전히 full replay 기준으로는 느렸습니다.

## 최종 판단

현재 기준 추천은 아래입니다.

1. DeepSeek V4 `Flash` / `Pro`의 **full replay eval은 중단 유지**
2. 필요하면 `20~40`개 규모 subset만 sanity check
3. full 평가는 `vLLM` 또는 `sglang` 등 더 최적화된 경로가 가능해질 때 재개

한 줄로 요약하면:

**DeepSeek V4는 지금 경로로 full eval을 계속 밀기보다, 변환 결과만 보존하고 평가 자체는 보류하는 게 맞습니다.**
