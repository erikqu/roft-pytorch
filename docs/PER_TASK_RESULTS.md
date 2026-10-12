# Per-task actual verifier rewards

Descriptive breakdown of the complete frozen comparison. No new sampling or task selection. Both arms are fresh inference; baseline interruptions resumed original histories.

| Split | Task | Episodes per arm | Base reward | ROFT reward | Delta |
|---|---|---:|---:|---:|---:|
| transfer | fictional-face-angular-features | 4 | 0.008160 | 0.020206 | +0.012046 |
| transfer | gazebo | 4 | 0.000000 | 0.018215 | +0.018215 |
| transfer | greenhouse | 4 | 0.005220 | 0.005973 | +0.000753 |
| transfer | left-foot-raised-heel-pose | 4 | 0.000000 | 0.000000 | +0.000000 |
| transfer | left-hand-relaxed-fingers | 4 | 0.002883 | 0.005921 | +0.003038 |
| transfer | stone-gatehouse | 4 | 0.024420 | 0.019516 | -0.004905 |
| within_task | astronaut-complete-standing-suit | 1 | 0.000000 | 0.000000 | +0.000000 |
| within_task | clock-tower | 1 | 0.000000 | 0.005747 | +0.005747 |
| within_task | cowboy-character | 1 | 0.000000 | 0.000000 | +0.000000 |
| within_task | fictional-face-soft-features | 1 | 0.000000 | 0.017306 | +0.017306 |
| within_task | pagoda | 1 | 0.000000 | 0.000000 | +0.000000 |
| within_task | right-foot-flat-resting-pose | 1 | 0.000000 | 0.015474 | +0.015474 |
| within_task | right-hand-open-palm | 1 | 0.013945 | 0.025840 | +0.011895 |
| within_task | skyscraper-exposed-platforms-and-pillars | 1 | 0.000000 | 0.000000 | +0.000000 |
| within_task | suburban-ranch-house-complete-exterior | 1 | 0.000000 | 0.035613 | +0.035613 |
| within_task | white-house-complete-exterior | 1 | 0.061571 | 0.017877 | -0.043694 |

The White House baseline includes one explicitly lost tool observation after a supervisor interruption. Its existing script/scene evidence and original raw tokens were retained; the action was not repeated.
These task rows do not add independent evidence beyond the reported 34 paired episodes. Within-task rows have only one seed each.
