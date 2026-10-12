# Setup and reproduction boundaries

This repository exports the implementation and evidence from a completed Blender run. It does not bundle the benchmark data, verifier release, weights, credentials or full raw trajectories. Numeric result verification works using only Python's standard library. Re-running Blender training requires those external assets and a compatible GPU environment.

## Runtime

The recorded run used Python 3.12, PyTorch 2.11 with CUDA 13, Transformers 5.18 and vLLM 0.26 on an 80 GB H100 PCIe. Install `requirements.txt` in a compatible environment; Docker, Blender 4.5.12 and `pigz` are additional system requirements. A Docker image matching the original runtime is named `arle-blender-pilot:4.5.12` by default; set `ROFT_BLENDER_IMAGE` for your own prepared image. CUDA wheels/drivers need to match your system; the requirements are a recorded environment, not a tested installer for every machine.

Full-parameter FP32 storage and CPU optimizer state require substantial host RAM and disk. The saved final master/Adam/RNG archive is 50.1 GB, in addition to 9.1 GB of BF16 serving shards. A full run keeps all update states and trajectories; allow ample storage. The native 131,072-token backward probe reached about 79.1 GB allocated GPU memory. Do not assume this full-parameter path fits a 27B model on one H100.

## Prepare a separate run directory

Set absolute local paths:

```bash
export ROFT_RUN="$PWD/artifacts/my-run"
export ROFT_MODEL="/absolute/path/to/pinned-Qwen3.5-4B"
export ROFT_LOCAL_DOCKER=1  # use docker directly if your user has permission
```

Before freezing a new run, populate:

```text
$ROFT_RUN/
  protocol.json                    # start from configs/experiment.json
  prompts.json                     # rows: task, text, images (relative to run)
  dataset_identity.json
  source_dataset_manifest.json
  tasks/<task>/environment/workspace/
  tasks/<task>/tests/               # exact compatible published verifier
  public_tasks/<task>/...           # original prompt image files
  source_harness/scripts/          # matching grader/host-judge transport helpers
  source_harness/vendor/refresh_blender_verifiers/
  source_harness/configs/reward.json
  work_window.json                 # started_unix and absolute deadline_unix
```

The repository includes required host-side helper sources under `code/source_harness/scripts`. Copy these into the prepared run's `source_harness/scripts`; supply the compatible published `refresh_blender_verifiers` package and exact task assets separately. The original dataset/verifier identity is recorded in `results/dataset_identity.json` and `results/frozen_manifest.json`. The original verifier release was `blender-v1.42.0`. The task package also provides its own `verifier_core` code and grading configuration.

Copy `configs/experiment.json` to `protocol.json` and choose a new run identity and a bounded work deadline before sealing. Historical `planned_updates: 4` and the descriptive 600-second episode string are stale metadata: the actual run took seven updates, and executable `max_episode_seconds: null` imposes no episode clock. Preserve the frozen original config in `configs/experiment.json`; a new protocol should correct its descriptive fields explicitly. Reflection selection uses actual accepted lengths, so the number of updates is not fixed by the descriptive planned count.

Download the exact model revision recorded in the config (`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) and point `ROFT_MODEL` at its local files. `work_window.json` must have an absolute Unix `deadline_unix`; a resume never resets an expired clock. No included tool allocates or shuts down a cloud instance.

## Host judging

Actual training uses the original Blender verifier and a configured host visual judge. The credentials stay on the judging host; task containers have no network. For the Azure bridge, set `ROFT_AZURE_ENDPOINT` and `EXPERIMENT_AZURE_CONFIG` explicitly. The latter points at a private TOML file with `[model_providers.azure.auth]`, `command` naming a token helper, and optional `timeout_ms`; the helper prints a bearer token or JSON with an `access_token` field. Do not commit this file.

The prepared `source_harness/configs/reward.json` must bind the same judge endpoint, model, source hash, limits, rubric and budget. The completed run used `gpt-6-luna` at `xhigh`, with maximum output 16,384. Changing the judge/verifier produces a new experiment and cannot reproduce the reported scores exactly. Start `code/judge_bridge.py` on the host and expose its loopback port to the GPU node through an SSH reverse tunnel if needed.

## Run and audit

After all external assets, credentials and the bounded work window are prepared:

```bash
python3 code/prepare.py
python3 code/worker.py
```

The worker performs real native-gradient, long-sequence, tool and reflection qualification before fresh training, then evaluates ROFT and base. It writes raw tokens before grading, saves each numerical update, and refuses unverified or duplicate-target recovery.

`code/controller.py` is an optional retained-Lambda-node supervisor, not a cloud provisioner. It requires a private allocation receipt with `instance_id`, `name`, `ip` and optional `ssh_user`, the Lambda API key in the environment, and already-prepared remote data/model directories. Set `ROFT_REMOTE` and `ROFT_NODE_PYTHON`; optional `ROFT_NODE_PATH` and `ROFT_NODE_LD_LIBRARY_PATH` handle machine-specific CUDA environments. The preferred entry point for a locally prepared machine is `worker.py`.

Audits under `code/audits` require full original artifact layouts. The public JSON receipts are compact exports of completed audits; they are not a replacement for running audits against actual checkpoints and raw files. Keep operational recovery scripts for diagnostics, and qualify any new repair before applying it. The full combined audit retains an original-v4 document dependency; for a new run, update that reporting requirement explicitly to your own methodology record.

## What was checked before publication

The public export's numeric verifier recomputes all four means, both paired deltas, pair counts and fixed-seed task-cluster bootstrap intervals from 68 actual episode reward records. CPU methodology tests cover native tool parsing, duplicate arguments, exact whole-target budgeting and truthful corrupt-scene feedback. All Python sources compile. Publication does not rerun the GPU training experiment or redistribute its large raw/checkpoint files.
