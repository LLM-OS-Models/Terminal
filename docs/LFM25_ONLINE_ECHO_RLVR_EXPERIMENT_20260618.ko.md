# LFM2.5 Online ECHO RLVR Experiment Record (2026-06-18)

Generated from git history and local TB2-lite JSON results on 2026-06-18.

## Current Stop State

- User requested stopping training/evaluation and preserving the experiment record.
- GPU 0-5 online training/vLLM rollout processes were stopped by matching the online RLVR launch, train, vLLM, adapter-sync, rollout-sync, eval-sync, and GPU6 watcher command patterns.
- GPU6 automatic evaluation watcher was stopped. The final saved online checkpoint was preserved on disk before shutdown.
- Latest saved online checkpoint found on disk: `checkpoint-6225`. Saved checkpoint count: `249`.
- Latest completed GPU6 evaluation JSON: `checkpoint-6200` with official Score `48.77` and Next Action `49.29`.
- Saved checkpoints without completed eval JSON: `1` total; tail: `checkpoint-6225`.

## Score Interpretation

- Official README/ranking score is `100 * avg_command_f1`.
- `next_action_score` is recorded only as a secondary diagnostic and is not the official leaderboard score.
- README/main leaderboard keeps only three representative RLVR rows: static/offline best, online best, and no-SFT raw best.
- Full RLVR checkpoint sweep, 6000+ rows, and the score/checkpoint graph are consolidated in one document: [`docs/LFM25_ECHO_RLVR_SWEEP_ANALYSIS_20260618.ko.md`](LFM25_ECHO_RLVR_SWEEP_ANALYSIS_20260618.ko.md).

## Online Result Summary

| Item | Checkpoint | Score | Next Action | Note |
| --- | ---: | ---: | ---: | --- |
| Online best | 425 | 53.58 | 52.27 | Main leaderboard representative for online RLVR |
| Online late 6000 first | 6000 | 51.17 | 50.88 | Below SFT1 baseline 52.30 |
| Online latest completed eval | 6200 | 48.77 | 49.29 | `-4.81` versus online best |
| Final saved checkpoint | 6225 | N/A | N/A | Saved on disk but no completed eval JSON |

## Conclusion

- The online RLVR pipeline did train and produced many checkpoint/eval pairs, but the best online checkpoint did not beat the older static ECHO RLVR best.
- Online best is Score `53.58` at `checkpoint-425`, which is `-0.47` versus static/offline best Score `54.05` at `checkpoint-610`.
- The late 6000+ region regressed: latest completed `checkpoint-6200` is Score `48.77`, far below both online best and SFT1 baseline `52.30`.
- The detailed reason and all checkpoint rows are intentionally kept in the single sweep document linked above, so README and other docs do not duplicate hundreds of RLVR checkpoint rows.
