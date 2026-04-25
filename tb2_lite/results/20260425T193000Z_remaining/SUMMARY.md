# TB2-lite Replay Results (2026-04-25)

Primary ranking uses `next_action_score = 0.7 * avg_command_f1 + 0.3 * first_cmd_exact`.

| Rank | Model | Score | Cmd F1 | First Cmd Exact | Valid JSON | Complete Recall | Premature Complete | Sec/Step | Load (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Qwen3.5-9B | 27.04 | 0.2808 | 24.6% | 19.9% | 31.0% | 0.6% | 0.072 | 90.6 |
| 2 | Qwen3.5-2B | 26.52 | 0.2743 | 24.4% | 18.1% | 10.3% | 0.0% | 0.024 | 84.7 |
| 3 | Qwen3.5-4B | 26.36 | 0.2745 | 23.8% | 17.4% | 17.2% | 0.0% | 0.055 | 77.2 |
| 4 | LFM2-2.6B | 24.12 | 0.2379 | 24.9% | 15.8% | 44.8% | 2.8% | 0.032 | 36.4 |
| 5 | gemma-4-E4B-it | 23.43 | 0.2391 | 22.3% | 12.7% | 44.8% | 0.8% | 0.051 | 129.7 |
| 6 | LFM2.5-1.2B-Instruct | 23.36 | 0.2381 | 22.3% | 18.7% | 34.5% | 9.5% | 0.021 | 31.7 |

Results directory: `tb2_lite/results/20260425T193000Z_remaining`
