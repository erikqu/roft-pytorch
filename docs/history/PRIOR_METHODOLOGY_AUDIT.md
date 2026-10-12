# Current ROFT methodology audit

The v4 training run performed one real 256,721-token update. It did not complete a final model comparison. All64source rewards werezero; one-byte corrupt output was incorrectly labelled executed_and_saved by the execution summary.

| Item | Paper | Observed v4 | Correction |
|---|---|---|---|
| Online update schedule | Many fresh generate/train cycles; 64 source attempts and four reflections per nominal update | One 256,721-token update, no source attempts from updated checkpoint | Multiple fresh updates; batch/token choice must be explicit |
| All accepted target tokens supervised | Every retained thinking/answer/EOT token participates | 169,442 of 426,163 retained tokens omitted via random masks | Use whole accepted targets; exact token cap only boundary target may be partial; disclose budget adaptation |
| Native tool interface | Functional file/shell agent interface | Custom JSON actions without model-native tool schema;84.3% malformed observations | Model-native schemas/tool serialization; real task qualification |
| Execution/verdict evidence | Observed tests/patch, raw outcome distinct from shaped reward and termination | One-byte file labelled executed_and_saved; verifier reason omitted from reflector | File existence != valid Blender scene; pass verifier reasons and stdout/stderr truthfully |
| Turn cutoff | 8192 per-turn cap,131072 total,100 turns | A single per-turn length cutoff ends entire episode | Record truncated turn and continue within remaining whole-episode limits |
| Context exhaustion | 131072 context without compaction | Reserve full8192 tokens; stop with up to8191 tokens still usable | Compute actual remaining context before next request |
| Asynchronous batching | Completion-order sources, serving refresh, bounded <=1-step stale | Wait for entire source batch, then entire reflection batch | Serial on-policy phase barrier acceptable singleH100 adaptation; do not claim latency equivalence |
| Gradient precision | BF16 training,FP32 gradient accumulation | BF16 native per-backward gradients copied to FP32 between microbatches | Qualify/correct accumulation precision; separate native BF16 contributions from across-microbatch FP32 |
| Task distribution/rewards/hardware | 767 mixed-success coding tasks,eight B200s,coding test success | 72 Blender train tasks,singleH100/CPUAdam,continuous geometry/visual reward | Retain user-required adaptations, report explicitly |
| Evaluation/robustness | Fresh heldout evaluation of completed checkpoints | GRPO Docker cleanup timeout stopped workflow before any final evaluation | Preserve episode on cleanup error; evaluate ROFT independently of comparator completion; snapshot/backup actual saved states |
