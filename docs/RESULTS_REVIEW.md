# Corrected ROFT on Blender: completed and audited

Qwen3.5-4B received seven full-parameter online updates using exactly 256,721 backward tokens. It generated 112 fresh source attempts and 448 fresh reflections; 438 selected targets were trained once. There was no reflection replay. All seven saved full FP32 master/Adam/RNG states passed numerical audits. The final 13 answer/EOT tokens of one boundary target were omitted to meet the user’s exact token budget.

| Arm | Evaluation provenance | Backward tokens | Transfer mean reward (24 episodes) | Within-task mean reward (10 episodes) |
|---|---|---:|---:|---:|
| Base Qwen3.5-4B | Fresh inference, interrupted histories resumed | 0 | 0.0067806 | 0.0075516 |
| ROFT Qwen3.5-4B | Fresh inference | 256,721 | 0.0116385 | 0.0117857 |
| Earlier GRPO 27B run | Historical/reused, incompatible model and setup; not measured here | Not compared | Not measured | Not measured |

| Paired difference (ROFT minus base) | Mean reward delta | 95% task-cluster bootstrap interval | Higher / lower / unchanged reward pairs |
|---|---:|---|---|
| Transfer | +0.0048579 | [-0.0009334, +0.0113663] | 6 / 5 / 13 |
| Within-task | +0.0042341 | [-0.0088239, +0.0150513] | 5 / 1 / 4 |

These are continuous rewards from the actual frozen Blender verifiers, not coding success rates or percentage-point accuracy. Both intervals include zero. The run does not establish a reliable improvement, and absolute rewards are low. Within-task evaluation uses new fixed episode seeds on task families used during training, so it is not transfer to new tasks. No equal-token fresh GRPO arm was measured in this experiment.

The paper’s exact prompts and recipe were pinned, but no official ROFT trainer was found. This is a local implementation on Blender tasks with one H100 and CPU Adam, rather than the paper’s coding dataset and eight B200s. It cannot establish the paper’s wall-time advantage.

A qualified uniform malformed-topology guard was admitted before both final evaluation arms. It uses the released verifier’s existing InvalidSurface handling and preserves valid scores in qualification. Original failures and candidate bytes are retained.

The baseline’s deadline interruption and later backup-sync interruption were recovered from original task containers and pre-grading raw turns. Completed candidates were preserved unchanged; next seeds and context budgets were checked. One White House tool observation lost its exit status/stdout/stderr during supervisor interruption. Those fields remain explicitly unknown; read-only file evidence is separate, and the action was not repeated. This is an operational deviation from uninterrupted evaluation.

Both final serving model shards are backed up off-node and independently SHA-verified. The full 66,896-file raw-evidence backup passed current source SHA checks. The complete final model and 50 GB master/Adam/RNG archive are backed up off-node with full SHA verification. Independent current state and raw-byte audits passed: every model shard and the complete master/Adam/RNG archive, plus all 66,896 files (56.3 GB of recovered original evidence). The final combined completion audit passed all 17 requirements. A later read-only live inventory confirmed the H100 is active. All original checkpoint states remain on the retained H100, which must stay running after completion, failure or deadline.

Machine-readable measurements: ../results/results.json. Frozen methodology and deviations: METHODOLOGY_ALIGNMENT.md and METHODOLOGY_COMPARISON.md. The full 68-candidate raw-output audit and report audit passed with no identity or numerical mismatches. All full-state/raw backups, current-byte and combined completion audits passed.
