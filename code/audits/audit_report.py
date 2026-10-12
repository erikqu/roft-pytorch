"""Reconcile the final reported numbers with complete fresh verifier evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from audit_evaluation import audit as evaluation_audit
from raw_evidence_io import load_json


def load(path):
    return load_json(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_number(left, right):
    return isinstance(left, (int, float)) and not isinstance(left, bool) and \
        math.isfinite(left) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def expected_delta(candidate, control, split):
    keys = sorted(k for k in candidate if k[0] == split)
    if set(keys) != {k for k in control if k[0] == split} or not keys:
        raise RuntimeError('Actual verifier rows are not paired')
    differences = [candidate[k] - control[k] for k in keys]
    tasks = sorted({k[1] for k in keys})
    rng = random.Random(20261008)
    boot = []
    for _ in range(10000):
        sample = [rng.choice(tasks) for _ in tasks]
        values = [candidate[k] - control[k] for task in sample for k in keys if k[1] == task]
        boot.append(sum(values) / len(values))
    boot.sort()
    return {'mean_reward_delta': sum(differences) / len(differences),
            'ci95_task_bootstrap': [boot[249], boot[9749]],
            'paired_fixes': sum(x > 0 for x in differences),
            'paired_regressions': sum(x < 0 for x in differences)}


def audit(root):
    paired = evaluation_audit(root)
    result_path = root / 'results.json'
    if not result_path.exists():
        return {'at': time.time(), 'verified': False, 'read_only': True,
                'report_present': False, 'fresh_full_set_verified': paired['paired_full_set_verified'],
                'problems': ['Final report has not been generated']}
    result = load(result_path)
    cfg = load(root / 'protocol.json')
    problems = []
    if not paired['paired_full_set_verified']:
        problems.append('Both complete fresh paired arms are required')
    if result.get('model') != cfg['model'] or result.get('protocol_sha256') != sha(root / 'protocol.json'):
        problems.append('Report model/protocol differs from frozen run')
    if result.get('implementation_provenance') != cfg['implementation_provenance']:
        problems.append('Report implementation provenance differs')
    if set(result.get('arms', {})) != {'base', 'roft'}:
        problems.append('Report arms differ from the requested measured comparison')
    values = {}
    for arm in ('base', 'roft'):
        rows = []
        for candidate in paired['arms'][arm]['actual_rewards']:
            rows.append(load(root / 'episodes' / candidate['id'] / 'result.json'))
        record = result.get('arms', {}).get(arm, {})
        if record.get('provenance') != 'fresh inference':
            problems.append(arm + ': fresh provenance missing')
        ids = record.get('episodes', [])
        if len(ids) != len(set(ids)) or set(ids) != {r['id'] for r in rows}:
            problems.append(arm + ': reported episodes differ from actual verifier rows')
        expected_tokens = 0 if arm == 'base' else 256721
        if record.get('backward_tokens') != expected_tokens:
            problems.append(arm + ': reported backward token budget differs')
        values[arm] = {(r['split'], r['task'], r['seed']): r['reward'] for r in rows}
        for split in ('transfer', 'within_task'):
            scores = [r['reward'] for r in rows if r['split'] == split]
            displayed = record.get('scores', {}).get(split, {})
            if not scores:
                problems.append(arm + ': missing verifier rows for ' + split)
                continue
            if displayed.get('n') != len(scores) or not same_number(displayed.get('mean_reward'), sum(scores) / len(scores)):
                problems.append(arm + ': reported count/mean differs from actual ' + split + ' rewards')
            if displayed.get('actual_verifier_results') is not True:
                problems.append(arm + ': actual-verifier provenance missing')
    for split in ('transfer', 'within_task'):
        if not values['base'] or not values['roft']:
            continue
        try:
            expected = expected_delta(values['roft'], values['base'], split)
        except RuntimeError as error:
            problems.append(str(error))
            continue
        row = result.get('deltas', {}).get('roft_minus_base_' + split, {})
        if not same_number(row.get('mean_reward_delta'), expected['mean_reward_delta']):
            problems.append(split + ': reported paired delta differs')
        interval = row.get('ci95_task_bootstrap')
        if not isinstance(interval, list) or len(interval) != 2 or not all(
                same_number(observed, actual) for observed, actual in
                zip(interval, expected['ci95_task_bootstrap'])):
            problems.append(split + ': reported task-cluster interval differs')
        for key in ('paired_fixes', 'paired_regressions'):
            if row.get(key) != expected[key]:
                problems.append(split + ': reported ' + key + ' differs')
    markdown = root / 'RESULTS.md'
    if not markdown.exists():
        problems.append('Readable report missing')
    else:
        text = markdown.read_text()
        for arm, record in result.get('arms', {}).items():
            scores = record.get('scores', {})
            if set(scores) >= {'transfer', 'within_task'}:
                expected = (f"| {arm} | fresh inference | {record.get('backward_tokens')} | "
                            f"{scores['transfer']['mean_reward']:.5f} | {scores['within_task']['mean_reward']:.5f} |")
                if expected not in text:
                    problems.append(arm + ': readable table differs from verified JSON')
    return {'at': time.time(), 'verified': not problems, 'report_present': True,
            'fresh_full_set_verified': paired['paired_full_set_verified'],
            'actual_verifier_rows': {arm: len(rows) for arm, rows in values.items()},
            'report_sha256': sha(result_path), 'problems': problems, 'read_only': True,
            'historical_or_baseline_output_substitution': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2), flush=True)
