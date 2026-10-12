"""Pilot grader with isolated Blender processes for independent clay evidence.

The sealed verifier still computes every reward component and cap. Only its
unscored modeling evidence renders are scheduled concurrently. This file is
self-contained so the resident harness can bind-mount it as /grade.py.
"""
import concurrent.futures
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def render_job(path):
    job = json.loads(Path(path).read_text())
    sys.path.insert(0, job["core"])
    import render_compare
    session = render_compare.RenderSession(samples_used=job["samples"])
    render_compare.render_blend_views(
        Path(job["blend"]), render_compare.load_manifest(job["manifest"]),
        [job["view"]], Path(job["out_dir"]), job["prefix"], session,
        shading="clay")
    Path(job["result"]).write_text(json.dumps({
        "frames": session.frames_rendered, "seconds": session.seconds_rendered,
        "log": session.log}))


def execute_preview(command, env, stdout, stderr, job, timeout=240):
    """Only a candidate's optional clay-preview timeout may be omitted.

    Geometry scoring runs separately in the sealed verifier. A failed
    reference, non-timeout worker error, or scoring error remains fatal.
    """
    try:
        subprocess.run(command, env=env, stdout=stdout, stderr=stderr,
                       check=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if job['prefix'] != 'candidate':
            raise
        output = Path(job['out_dir']) / f"{job['prefix']}-{job['view']}.png"
        output.unlink(missing_ok=True)
        return {'frames': 0, 'seconds': 0.0, 'log': [], 'skipped': True,
                'prefix': job['prefix'], 'view': job['view'],
                'reason': 'unscored_candidate_preview_timeout', 'timeout_seconds': timeout,
                'error': str(exc), 'scoring_implementation_changed': False}
    output = Path(job['out_dir']) / f"{job['prefix']}-{job['view']}.png"
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Render worker produced no image: {output.name}")
    return json.loads(Path(job['result']).read_text())


def preview_evidence_count(count, evidence_views, skipped):
    """Account only for explicit candidate timeouts in the declared view set."""
    assert count == len(evidence_views) and 1 <= count <= 8
    assert len(set(evidence_views)) == count
    assert skipped and all(r.get('prefix') == 'candidate'
                           and r.get('reason') == 'unscored_candidate_preview_timeout'
                           and r.get('skipped') is True for r in skipped)
    views = [r['view'] for r in skipped]
    assert len(set(views)) == len(views)
    return count - len(set(evidence_views).intersection(views))


def run_evidence_runner(runner):
    """Adapt unscored evidence presence checks without changing sealed scoring."""
    import yaml
    sys.path.insert(0, str(Path(runner).parent))
    spec = importlib.util.spec_from_file_location('pilot_original_runner', runner)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = module.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    scene = config['grading']['checks']['scene']
    original = module.verify_candidate_evidence
    skips_path = Path(args.logs_root) / 'preview-skips.json'
    verified_omission = False

    def verify(count, artifacts, *, require_candidates):
        nonlocal verified_omission
        if not skips_path.exists() or not require_candidates:
            return original(count, artifacts, require_candidates=require_candidates)
        assert scene['verifier'] == 'standard_prop'
        assert config['archetype'] == 'modeling'
        skipped = json.loads(skips_path.read_text())['skipped']
        remaining = preview_evidence_count(count, scene['parameters']['evidence_views'], skipped)
        # The engine writes surviving paired evidence with consecutive indices.
        original(remaining or count, artifacts, require_candidates=bool(remaining))
        expected_refs = [artifacts / f'reference-{i:02d}.png' for i in range(1, remaining + 1)]
        assert sorted(artifacts.glob('reference-*.png')) == expected_refs
        assert all(p.stat().st_size > 0 for p in expected_refs)
        verified_omission = True

    original_request = module.build_caliper_judge_request

    def build_request(config, config_path, logs, artifacts):
        if not verified_omission:
            return original_request(config, config_path, logs, artifacts)
        # This pilot never calls the host judge or consumes a visual reward.
        # Do not fabricate a judge-ready request with incomplete previews.
        (logs / 'pilot-judge-request-omitted.json').write_text(json.dumps({
            'reason': 'Optional previews timed out; deterministic-only pilot reward.',
            'judge_called': False, 'judge_attestation_created': False}))
        return None

    module.verify_candidate_evidence = verify
    module.build_caliper_judge_request = build_request
    return module.main()


def run_scene(engine):
    spec = importlib.util.spec_from_file_location("pilot_original_scene", engine)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = module.render_both_sides
    root = Path(os.environ["ARLE_PARALLEL_LOGS"])

    def parallel_both(manifest, reference, candidate, views, out_dir, session, shading):
        # Restrict the optimization to unscored Workbench modeling evidence.
        if shading != "clay" or manifest.engine != "WORKBENCH":
            return original(manifest, reference, candidate, views, out_dir, session, shading)
        started = time.monotonic()
        runtime = root / "parallel-runtime" / "jobs"
        runtime.mkdir(parents=True, exist_ok=True)
        samples = session.samples_used if session.samples_used is not None else manifest.samples
        jobs = []
        for prefix, blend in (("reference", reference), ("candidate", candidate)):
            for view in views:
                stem = runtime / f"{prefix}-{view}"
                request = stem.with_suffix(".request.json")
                job = {"core": str(Path(engine).parent.parent), "blend": str(blend),
                       "manifest": str(manifest.path), "view": view, "prefix": prefix,
                       "out_dir": str(out_dir), "samples": samples,
                       "result": str(stem.with_suffix(".result.json"))}
                request.write_text(json.dumps(job))
                jobs.append((request, job))

        def execute(item):
            request, job = item
            env = dict(os.environ, LP_NUM_THREADS="2", OMP_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="1")
            command = [os.environ["ARLE_REAL_BLENDER"], "--background", "--factory-startup",
                       "--disable-autoexec", "--threads", "2", "--python", str(Path(__file__).resolve()),
                       "--", "--render-job", str(request)]
            with request.with_suffix(".stdout").open("w") as stdout, \
                 request.with_suffix(".stderr").open("w") as stderr:
                return execute_preview(command, env, stdout, stderr, job)

        try:
            workers = min(8, len(jobs))
            if workers == 0:
                return {}, {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                # Each job runs a fresh Blender process; no bpy state crosses threads.
                results = list(pool.map(execute, jobs))
            if session.samples_used is None:
                session.samples_used = samples
                session.first_frame_seconds = results[0]["seconds"]
            session.frames_rendered += sum(r["frames"] for r in results)
            session.seconds_rendered += sum(r["seconds"] for r in results)
            session.log.extend(item for result in results for item in result["log"])
            skipped = [result for result in results if result.get('skipped')]
            if skipped:
                (root / 'preview-skips.json').write_text(json.dumps({
                    'scope': 'Optional candidate clay previews only; deterministic scoring and caps remain unchanged.',
                    'skipped': skipped}, indent=2))
            (root / "parallel-render-report.json").write_text(json.dumps({
                "workers": workers, "threads_per_worker": 2, "frames": sum(r['frames'] for r in results),
                "skipped_preview_count": len(skipped),
                "wall_seconds": time.monotonic() - started,
                "worker_render_seconds_sum": sum(r["seconds"] for r in results),
                "scoring_implementation_changed": False}, indent=2))
            skipped_views = {(r['prefix'], r['view']) for r in skipped}
            return tuple({view: Path(out_dir) / f"{prefix}-{view}.png" for view in views
                          if (prefix, view) not in skipped_views}
                         for prefix in ("reference", "candidate"))
        except Exception as exc:
            # The sealed scene driver catches exceptions; distinguish infrastructure
            # failures explicitly so a failed worker can never become reward zero.
            (root / "parallel-failure.json").write_text(json.dumps({"error": str(exc)}))
            raise

    if module.parse_args().task != "model":
        raise RuntimeError("Parallel pilot grader only supports modeling tasks")
    module.render_both_sides = parallel_both
    module.main()


def grade(tests, app, logs):
    import yaml
    config = yaml.safe_load((tests / "verifier.yaml").read_text())
    if config["grading"]["checks"].get("llm_judge", {}).get("provider") == "openai":
        return grade_shared(tests, app, logs, phase="evidence")
    runtime = logs / "parallel-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    real_blender = shutil.which("blender")
    if not real_blender:
        raise RuntimeError("Blender unavailable")
    launcher = runtime / "blender"
    launcher.write_text("#!/usr/bin/python3\nimport os, sys\n"
        "args = sys.argv[1:]\n"
        "if '--python' in args:\n"
        "    index = args.index('--python') + 1\n"
        "    if args[index].endswith('/engines/standard_scene.py'):\n"
        "        engine = args[index]\n"
        f"        args[index] = {str(Path(__file__).resolve())!r}\n"
        "        split = args.index('--') + 1\n"
        "        args[split:split] = ['--parallel-scene', engine]\n"
        "        args[:0] = ['--threads', '4']\n"
        "os.execv(os.environ['ARLE_REAL_BLENDER'], [os.environ['ARLE_REAL_BLENDER']] + args)\n")
    launcher.chmod(0o755)
    env = dict(os.environ, ARLE_PARALLEL_LOGS=str(logs), ARLE_REAL_BLENDER=real_blender,
               PATH=str(runtime) + os.pathsep + os.environ["PATH"],
               LP_NUM_THREADS="4", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    started = time.monotonic()
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
        "--evidence-runner", str(tests / "verifier_core/runner.py"),
        "--config", str(tests / "verifier.yaml"), "--tests-root", str(tests),
        "--app-root", str(app), "--logs-root", str(logs), "--phase", "evidence"],
        capture_output=True, text=True, env=env)
    (logs / "grader.stdout").write_text(result.stdout)
    (logs / "grader.stderr").write_text(result.stderr)
    if result.returncode or (logs / "parallel-failure.json").exists():
        raise RuntimeError("Verifier infrastructure failure; inspect grader and worker logs")
    technical = json.loads((logs / "verifier/technical.json").read_text())
    config = yaml.safe_load((tests / "verifier.yaml").read_text())
    sys.path.insert(0, str(tests / "verifier_core"))
    from runner import apply_score_caps
    score = float(technical["deterministic"])
    if not math.isfinite(score):
        raise RuntimeError("Nonfinite deterministic score")
    reward = apply_score_caps(config, technical, {"reward": score}, logs / "verifier")
    reward["reward_contract"] = "modeling-v3-deterministic-pilot"
    (logs / "pilot-reward.json").write_text(json.dumps(reward, indent=2))
    (logs / "parallel-grader-time.json").write_text(json.dumps({"seconds": time.monotonic() - started}))
    print(json.dumps(reward))


def grade_shared(tests, app, logs, *, phase):
    """Run sealed evidence/finalization without preview or reward overrides."""
    # Blender otherwise creates render workers for every host core despite the
    # container CPU quota. Limit worker count without changing sealed scoring.
    real_blender = shutil.which("blender")
    if not real_blender:
        raise RuntimeError("Blender unavailable")
    runtime = logs / "cpu-runtime"
    runtime.mkdir(exist_ok=True)
    launcher = runtime / "blender"
    launcher.write_text("#!/usr/bin/python3\nimport os, sys\n"
                        f"os.execv({real_blender!r}, [{real_blender!r}, '--threads', '4', *sys.argv[1:]])\n")
    launcher.chmod(0o755)
    result = subprocess.run([
        sys.executable, str(tests / "verifier_core/runner.py"),
        "--config", str(tests / "verifier.yaml"), "--tests-root", str(tests),
        "--app-root", str(app), "--logs-root", str(logs), "--phase", phase,
    ], capture_output=True, text=True,
        env=dict(os.environ, PATH=str(runtime) + os.pathsep + os.environ["PATH"],
                 LP_NUM_THREADS="4", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1"))
    (logs / f"{phase}.stdout").write_text(result.stdout)
    (logs / f"{phase}.stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"Shared verifier {phase} failed; no training reward produced")
    if phase == "finalize":
        reward = json.loads((logs / "verifier/reward.json").read_text())
        evaluation = json.loads((logs / "verifier/evaluation.json").read_text())
        value = reward["reward"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise RuntimeError("Shared verifier returned invalid reward")
        if value != evaluation["reward"]:
            raise RuntimeError("Shared reward and evaluation disagree")
        reward["applied_caps"] = evaluation.get("applied_caps", [])
        import yaml
        config = yaml.safe_load((tests / "verifier.yaml").read_text())
        reward["reward_contract"] = config["verifier"]["release"]
        reward["geometry_profile_revision"] = config["grading"]["checks"]["scene"]["parameters"]["geometry_profile_revision"]
        (logs / "pilot-reward.json").write_text(json.dumps(reward, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--finalize-shared':
        grade_shared(*map(Path, sys.argv[2:]), phase="finalize")
    elif len(sys.argv) > 1 and sys.argv[1] == '--evidence-runner':
        runner = sys.argv[2]
        del sys.argv[1:3]
        sys.exit(run_evidence_runner(runner))
    elif "--" in sys.argv:
        split = sys.argv.index("--")
        mode = sys.argv[split + 1]
        if mode == "--render-job":
            render_job(sys.argv[split + 2])
        elif mode == "--parallel-scene":
            engine = sys.argv[split + 2]
            del sys.argv[split + 1:split + 3]
            run_scene(engine)
        else:
            raise RuntimeError(f"Unsupported Blender entry mode: {mode}")
    else:
        grade(*map(Path, sys.argv[1:]))
