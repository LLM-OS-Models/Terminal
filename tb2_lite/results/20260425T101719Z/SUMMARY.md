# TB2-lite Replay Results (2026-04-25)

Primary ranking uses `next_action_score = 0.7 * avg_command_f1 + 0.3 * first_cmd_exact`.

| Rank | Model | Score | Cmd F1 | First Cmd Exact | Valid JSON | Complete Recall | Premature Complete | Sec/Step | Load (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LFM2-8B-Terminal-SFT-Unsloth | 30.14 | 0.2965 | 31.3% | 18.1% | 3.4% | 0.0% | 0.029 | 46.6 |
| 2 | Nemotron-Terminal-14B | 27.72 | 0.2751 | 28.2% | 35.0% | 6.9% | 0.0% | 0.108 | 68.9 |
| 3 | Qwen3.6-35B-A3B-FP8 | 25.21 | 0.2658 | 22.0% | 13.7% | 17.2% | 0.6% | 0.082 | 120.5 |
| 4 | gemma-4-31B-it | 24.70 | 0.2594 | 21.8% | 21.8% | 37.9% | 0.8% | 0.404 | 101.4 |
| 5 | LFM2-8B-A1B | 23.19 | 0.2336 | 22.8% | 14.2% | 41.4% | 1.1% | 0.025 | 58.2 |
| 6 | gemma-4-E2B-it | 23.05 | 0.2359 | 21.8% | 10.1% | 10.3% | 0.0% | 0.032 | 116.0 |
| 7 | LFM2-24B-A2B | 22.80 | 0.2323 | 21.8% | 20.2% | 27.6% | 10.6% | 0.050 | 81.6 |

Results directory: `tb2_lite/results/20260425T101719Z`
