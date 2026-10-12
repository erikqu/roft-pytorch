"""Recompute published scores from compact actual-verifier episode records."""
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def same(actual, expected):
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
        raise AssertionError('Expected a finite numerical reward')
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f'Numerical mismatch: {actual} != {expected}')

def verify():
    result = json.loads((ROOT/'results/results.json').read_text())
    rows = json.loads((ROOT/'results/episodes.json').read_text())['episodes']
    cfg = json.loads((ROOT/'configs/experiment.json').read_text())
    expected = {(split, task, seed) for split, tasks, seeds in [
        ('transfer', cfg['transfer'], cfg['transfer_seeds']),
        ('within_task', cfg['within_task_tasks'], cfg['within_task_seeds'])]
        for task in tasks for seed in seeds}
    assert len(rows) == 68 and len({r['id'] for r in rows}) == 68
    values = {}
    for arm in ('base', 'roft'):
        subset = [r for r in rows if r['arm'] == arm]
        assert len(subset) == 34
        assert {(r['split'],r['task'],r['seed']) for r in subset} == expected
        assert {r['id'] for r in subset} == set(result['arms'][arm]['episodes'])
        assert result['arms'][arm]['provenance'] == 'fresh inference'
        values[arm] = {}
        for r in subset:
            same(r['reward'],r['reward'])
            assert 0 <= r['reward'] <= 1 and r['provenance'] == 'fresh inference'
            assert all(isinstance(r[k],str) and len(r[k]) == 64 for k in ['prompt_sha256','response_sha256','reward_file_sha256'])
            assert (r['checkpoint_sha256'] is None) == (arm == 'base')
            values[arm][(r['split'],r['task'],r['seed'])] = r['reward']
        for split in ('transfer','within_task'):
            scores = [r['reward'] for r in subset if r['split'] == split]
            summary = result['arms'][arm]['scores'][split]
            assert summary['n'] == len(scores) and summary['actual_verifier_results']
            same(summary['mean_reward'],sum(scores)/len(scores))
    for split in ('transfer','within_task'):
        keys = sorted(k for k in values['roft'] if k[0] == split)
        differences = [values['roft'][k]-values['base'][k] for k in keys]
        summary = result['deltas']['roft_minus_base_'+split]
        same(summary['mean_reward_delta'],sum(differences)/len(differences))
        assert summary['paired_fixes'] == sum(x > 0 for x in differences)
        assert summary['paired_regressions'] == sum(x < 0 for x in differences)
        tasks = sorted({k[1] for k in keys});rng = random.Random(20261008);bootstrap = []
        for _ in range(10000):
            sample = [rng.choice(tasks) for _ in tasks]
            deltas = [values['roft'][k]-values['base'][k] for t in sample for k in keys if k[1] == t]
            bootstrap.append(sum(deltas)/len(deltas))
        bootstrap.sort()
        for a,b in zip(summary['ci95_task_bootstrap'],[bootstrap[249],bootstrap[9749]]):same(a,b)
        assert summary['ci95_task_bootstrap'][0] <= 0 <= summary['ci95_task_bootstrap'][1]
    checkpoint_audits = [json.loads(p.read_text()) for p in sorted((ROOT/'results/checkpoints').glob('checkpoint-*/audit.json'))]
    assert len(checkpoint_audits) == 7
    assert sum(x['supervised_tokens'] for x in checkpoint_audits) == 256721
    assert all(x['numerical_reload_verified'] and x['actual_changed_tensors'] == 723 for x in checkpoint_audits)
    assert result['arms']['roft']['backward_tokens'] == 256721 and result['arms']['base']['backward_tokens'] == 0
    combined = json.loads((ROOT/'results/audits/current_goal_completion_audit.json').read_text())
    assert combined['all_requirements_verified'] and len(combined['requirements']) == 17
    assert all(x['verified'] for x in combined['requirements'])
    print('PASS: 68 actual rewards; all means, paired deltas, bootstrap intervals, seven updates and 256,721 backward tokens reconcile.')
    print('Both confidence intervals include zero; no reliable improvement is established.')

if __name__ == '__main__':
    verify()
