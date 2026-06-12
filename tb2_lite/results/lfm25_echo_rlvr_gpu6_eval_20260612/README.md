# LFM2.5 ECHO RLVR GPU6 TB2-lite Evaluation

Updated: `2026-06-12 00:43:03 UTC`

This directory tracks TB2-lite full replay evaluations for ECHO-style RLVR LoRA checkpoints.

Score 기준: `100 * avg_command_f1`.

Current best in this comparison: `lfm25-echo-rlvr-continue-checkpoint-250` Score `52.88`.

| Rank | Model / checkpoint | Score | Cmd F1 | First Cmd | Valid JSON | Result |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `lfm25-echo-rlvr-continue-checkpoint-250` | 52.88 | 0.5305 | 52.5% | 74.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-250.json` |
| 2 | `LFM2.5-8B-A1B Terminal ToolBench Full SFT 1Epoch` | 52.30 | 0.5230 | 49.5% | 76.9% | `tb2_lite/results/20260605T_live_hrm_lora_lfm_epoch1/LFM2.5-8B-A1B-terminal-toolbench-full-1epoch-checkpoint-1542.json` |
| 3 | `lfm25-echo-rlvr-continue-checkpoint-100` | 52.02 | 0.5156 | 53.1% | 77.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-100.json` |
| 4 | `lfm25-echo-rlvr-continue-checkpoint-50` | 51.59 | 0.5163 | 51.5% | 76.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-50.json` |
| 5 | `lfm25-echo-rlvr-continue-checkpoint-400` | 51.47 | 0.5218 | 49.8% | 77.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-400.json` |
| 6 | `lfm25-echo-rlvr-continue-checkpoint-500` | 51.41 | 0.5150 | 51.2% | 75.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-500.json` |
| 7 | `lfm25-echo-rlvr-continue-checkpoint-150` | 51.16 | 0.5187 | 49.5% | 76.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-150.json` |
| 8 | `lfm25-echo-rlvr-continue-checkpoint-450` | 51.13 | 0.5072 | 52.1% | 76.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-450.json` |
| 9 | `lfm25-echo-rlvr-continue-checkpoint-200` | 50.73 | 0.5169 | 48.5% | 75.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-200.json` |
| 10 | `lfm25-echo-rlvr-continue-checkpoint-350` | 50.60 | 0.5094 | 49.8% | 76.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-350.json` |
| 11 | `LFM2.5-8B-A1B Terminal ToolBench Full SFT 2Epoch` | 50.48 | 0.5048 | 49.2% | 74.9% | `tb2_lite/results/20260605T_all_idle_eval/LFM2.5-8B-A1B-terminal-toolbench-full-2epoch-final.json` |
| 12 | `lfm25-echo-rlvr-continue-checkpoint-570` | 50.44 | 0.5097 | 49.2% | 75.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-570.json` |
| 13 | `lfm25-echo-rlvr-parent-checkpoint-1880` | 50.05 | 0.5114 | 47.5% | 74.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parent-checkpoint-1880.json` |
| 14 | `lfm25-echo-rlvr-continue-checkpoint-300` | 49.87 | 0.5058 | 48.2% | 76.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-300.json` |
| 15 | `LiquidAI/LFM2.5-8B-A1B Base` | 36.53 | 0.3653 | 39.9% | 59.1% | `/home/work/.data/tb2_lite_eval/corrected_readme_models_vllm/liquid_lfm25_8b_a1b_base.json` |

Note: the GPU6 watcher evaluates stride checkpoints plus the most recent checkpoints as they appear.
