# TB2-lite Replay Results (2026-04-25)

Primary ranking uses `next_action_score = 0.7 * avg_command_f1 + 0.3 * first_cmd_exact`.

| Rank | Model | Score | Cmd F1 | First Cmd Exact | Valid JSON | Complete Recall | Premature Complete | Sec/Step | Load (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Nemotron-Terminal-8B | 30.02 | 0.2969 | 30.8% | 15.0% | 10.3% | 0.0% | 0.078 | 49.8 |
| 2 | Nemotron-Terminal-32B | 29.13 | 0.2872 | 30.1% | 21.5% | 10.3% | 0.0% | 0.281 | 99.1 |
| 3 | Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled | 26.35 | 0.2744 | 23.8% | 19.2% | 10.3% | 0.0% | 0.282 | 113.4 |
| 4 | gemma-4-26B-A4B-it | 25.95 | 0.2631 | 25.1% | 20.7% | 31.0% | 0.6% | 0.094 | 123.9 |
| 5 | Qwen3.6-27B | 25.60 | 0.2702 | 22.3% | 17.9% | 13.8% | 0.3% | 0.282 | 118.5 |

Results directory: `tb2_lite/results/20260425T102110Z_bonus`
