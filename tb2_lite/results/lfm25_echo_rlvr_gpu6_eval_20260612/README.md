# LFM2.5 ECHO RLVR GPU6 TB2-lite Evaluation

Updated: `2026-06-12 02:39:05 UTC`

This directory tracks TB2-lite full replay evaluations for ECHO-style RLVR LoRA checkpoints.

Score 기준: `100 * avg_command_f1`.

Current best in this comparison: `lfm25-echo-rlvr-continue-checkpoint-250` Score `52.88`.

| Rank | Model / checkpoint | Score | Cmd F1 | First Cmd | Valid JSON | Result |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `lfm25-echo-rlvr-continue-checkpoint-250` | 52.88 | 0.5305 | 52.5% | 74.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-250.json` |
| 2 | `lfm25-echo-rlvr-continue-checkpoint-220` | 52.34 | 0.5326 | 50.2% | 77.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-220.json` |
| 3 | `LFM2.5-8B-A1B Terminal ToolBench Full SFT 1Epoch` | 52.30 | 0.5230 | 49.5% | 76.9% | `tb2_lite/results/20260605T_live_hrm_lora_lfm_epoch1/LFM2.5-8B-A1B-terminal-toolbench-full-1epoch-checkpoint-1542.json` |
| 4 | `lfm25-echo-rlvr-continue-checkpoint-100` | 52.02 | 0.5156 | 53.1% | 77.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-100.json` |
| 5 | `lfm25-echo-rlvr-parentrun-checkpoint-1830` | 51.94 | 0.5269 | 50.2% | 76.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-1830.json` |
| 6 | `lfm25-echo-rlvr-continue-checkpoint-180` | 51.86 | 0.5176 | 52.1% | 76.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-180.json` |
| 7 | `lfm25-echo-rlvr-continue-checkpoint-190` | 51.86 | 0.5231 | 50.8% | 77.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-190.json` |
| 8 | `lfm25-echo-rlvr-parentrun-checkpoint-1880` | 51.86 | 0.5215 | 51.2% | 76.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-1880.json` |
| 9 | `lfm25-echo-rlvr-continue-checkpoint-210` | 51.85 | 0.5175 | 52.1% | 76.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-210.json` |
| 10 | `lfm25-echo-rlvr-continue-checkpoint-20` | 51.78 | 0.5276 | 49.5% | 77.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-20.json` |
| 11 | `lfm25-echo-rlvr-continue-checkpoint-80` | 51.78 | 0.5276 | 49.5% | 76.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-80.json` |
| 12 | `lfm25-echo-rlvr-parentrun-checkpoint-20` | 51.70 | 0.5234 | 50.2% | 75.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-20.json` |
| 13 | `lfm25-echo-rlvr-continue-checkpoint-40` | 51.61 | 0.5209 | 50.5% | 77.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-40.json` |
| 14 | `lfm25-echo-rlvr-continue-checkpoint-50` | 51.59 | 0.5163 | 51.5% | 76.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-50.json` |
| 15 | `lfm25-echo-rlvr-continue-checkpoint-560` | 51.55 | 0.5243 | 49.5% | 76.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-560.json` |
| 16 | `lfm25-echo-rlvr-continue-checkpoint-110` | 51.52 | 0.5238 | 49.5% | 75.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-110.json` |
| 17 | `lfm25-echo-rlvr-continue-checkpoint-90` | 51.49 | 0.5162 | 51.2% | 74.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-90.json` |
| 18 | `lfm25-echo-rlvr-continue-checkpoint-140` | 51.48 | 0.5177 | 50.8% | 76.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-140.json` |
| 19 | `lfm25-echo-rlvr-continue-checkpoint-400` | 51.47 | 0.5218 | 49.8% | 77.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-400.json` |
| 20 | `lfm25-echo-rlvr-continue-checkpoint-60` | 51.44 | 0.5214 | 49.8% | 76.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-60.json` |
| 21 | `lfm25-echo-rlvr-continue-checkpoint-500` | 51.41 | 0.5150 | 51.2% | 75.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-500.json` |
| 22 | `lfm25-echo-rlvr-continue-checkpoint-30` | 51.36 | 0.5203 | 49.8% | 76.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-30.json` |
| 23 | `lfm25-echo-rlvr-continue-checkpoint-160` | 51.30 | 0.5237 | 48.8% | 76.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-160.json` |
| 24 | `lfm25-echo-rlvr-continue-checkpoint-120` | 51.21 | 0.5109 | 51.5% | 75.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-120.json` |
| 25 | `lfm25-echo-rlvr-continue-checkpoint-150` | 51.16 | 0.5187 | 49.5% | 76.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-150.json` |
| 26 | `lfm25-echo-rlvr-parentrun-checkpoint-10` | 51.14 | 0.5154 | 50.2% | 76.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-10.json` |
| 27 | `lfm25-echo-rlvr-parentrun-checkpoint-30` | 51.14 | 0.5155 | 50.2% | 75.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-30.json` |
| 28 | `lfm25-echo-rlvr-continue-checkpoint-450` | 51.13 | 0.5072 | 52.1% | 76.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-450.json` |
| 29 | `lfm25-echo-rlvr-continue-checkpoint-540` | 51.12 | 0.5152 | 50.2% | 76.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-540.json` |
| 30 | `lfm25-echo-rlvr-continue-checkpoint-580` | 51.12 | 0.5182 | 49.5% | 77.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-580.json` |
| 31 | `lfm25-echo-rlvr-continue-checkpoint-600` | 51.04 | 0.5085 | 51.5% | 74.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-600.json` |
| 32 | `lfm25-echo-rlvr-continue-checkpoint-70` | 50.95 | 0.5187 | 48.8% | 77.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-70.json` |
| 33 | `lfm25-echo-rlvr-continue-checkpoint-170` | 50.84 | 0.5111 | 50.2% | 75.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-170.json` |
| 34 | `lfm25-echo-rlvr-parentrun-checkpoint-1810` | 50.84 | 0.5155 | 49.2% | 73.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-1810.json` |
| 35 | `lfm25-echo-rlvr-continue-checkpoint-130` | 50.77 | 0.5187 | 48.2% | 78.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-130.json` |
| 36 | `lfm25-echo-rlvr-parentrun-checkpoint-1850` | 50.74 | 0.5157 | 48.8% | 75.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-1850.json` |
| 37 | `lfm25-echo-rlvr-continue-checkpoint-200` | 50.73 | 0.5169 | 48.5% | 75.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-200.json` |
| 38 | `lfm25-echo-rlvr-continue-checkpoint-620` | 50.68 | 0.5162 | 48.5% | 75.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-620.json` |
| 39 | `lfm25-echo-rlvr-parentrun-checkpoint-1870` | 50.67 | 0.5130 | 49.2% | 76.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-1870.json` |
| 40 | `lfm25-echo-rlvr-parentrun-checkpoint-1790` | 50.64 | 0.5168 | 48.2% | 75.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-1790.json` |
| 41 | `lfm25-echo-rlvr-continue-checkpoint-350` | 50.60 | 0.5094 | 49.8% | 76.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-350.json` |
| 42 | `LFM2.5-8B-A1B Terminal ToolBench Full SFT 2Epoch` | 50.48 | 0.5048 | 49.2% | 74.9% | `tb2_lite/results/20260605T_all_idle_eval/LFM2.5-8B-A1B-terminal-toolbench-full-2epoch-final.json` |
| 43 | `lfm25-echo-rlvr-continue-checkpoint-10` | 50.46 | 0.5075 | 49.8% | 76.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-10.json` |
| 44 | `lfm25-echo-rlvr-continue-checkpoint-570` | 50.44 | 0.5097 | 49.2% | 75.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-570.json` |
| 45 | `lfm25-echo-rlvr-parentrun-checkpoint-1800` | 50.37 | 0.5087 | 49.2% | 78.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-1800.json` |
| 46 | `lfm25-echo-rlvr-parentrun-checkpoint-1860` | 50.24 | 0.5068 | 49.2% | 74.3% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-1860.json` |
| 47 | `lfm25-echo-rlvr-parentrun-checkpoint-1820` | 50.22 | 0.5053 | 49.5% | 77.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-1820.json` |
| 48 | `lfm25-echo-rlvr-parent-checkpoint-1880` | 50.05 | 0.5114 | 47.5% | 74.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parent-checkpoint-1880.json` |
| 49 | `lfm25-echo-rlvr-parentrun-checkpoint-1840` | 49.99 | 0.5020 | 49.5% | 74.9% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-1840.json` |
| 50 | `lfm25-echo-rlvr-continue-checkpoint-590` | 49.95 | 0.5015 | 49.5% | 74.3% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-590.json` |
| 51 | `lfm25-echo-rlvr-continue-checkpoint-610` | 49.93 | 0.5012 | 49.5% | 75.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-610.json` |
| 52 | `lfm25-echo-rlvr-continue-checkpoint-300` | 49.87 | 0.5058 | 48.2% | 76.2% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-300.json` |
| 53 | `lfm25-echo-rlvr-continue-checkpoint-630` | 49.85 | 0.5086 | 47.5% | 76.6% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-630.json` |
| 54 | `lfm25-echo-rlvr-continue-checkpoint-550` | 49.38 | 0.4946 | 49.2% | 74.3% | `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-continue-checkpoint-550.json` |
| 55 | `LiquidAI/LFM2.5-8B-A1B Base` | 36.53 | 0.3653 | 39.9% | 59.1% | `/home/work/.data/tb2_lite_eval/corrected_readme_models_vllm/liquid_lfm25_8b_a1b_base.json` |

Note: the GPU6 watcher evaluates stride checkpoints plus the most recent checkpoints as they appear.
