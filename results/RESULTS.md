# Paper-derived Blender rerun

This is a local implementation of the ROFT paper recipe. No official ROFT trainer was found.
Qwen3.5-4B full-parameter training on one H100; continuous Blender verifier rewards, not the paper coding success rate.

| Arm | Provenance | Backward tokens | Transfer reward | Within-task reward |
|---|---|---:|---:|---:|
| base | fresh inference | 0 | 0.00678 | 0.00755 |
| roft | fresh inference | 256721 | 0.01164 | 0.01179 |

All results require preserved raw multi-turn samples, real verifier rewards and audited numerical updates.
Previous 27B one-shot results are separate historical experiments. The node remains running by user instruction.
