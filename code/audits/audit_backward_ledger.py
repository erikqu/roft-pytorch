"""Read-only reconciliation of completed native backwards and saved updates."""
import argparse
import json
import math
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def reconcile(root, ledger):
    problems = []
    steps = []
    saved_total = 0
    ordinals = sorted({row['step'] for row in ledger})
    if ordinals != list(range(1, len(ordinals) + 1)):
        problems.append('Backward ledger steps are not contiguous')
    consumed = set()
    for step in ordinals:
        rows = [row for row in ledger if row['step'] == step]
        targets = load(root / 'jobs' / f'roft-{step:03d}-targets.json')
        expected = [(t['id'], len(t.get('supervised_positions', t['token_ids']))) for t in targets]
        total = sum(n for _, n in expected)
        completed = 0
        loss = 0.0
        if len(rows) > len(expected):
            problems.append({'step': step, 'reason': 'More backwards than selected targets'})
        for index, row in enumerate(rows):
            if row['id'] in consumed:
                problems.append({'step': step, 'id': row['id'], 'reason': 'Reflection backward replay'})
            consumed.add(row['id'])
            if index >= len(expected) or (row['id'], row['backward_tokens']) != expected[index]:
                problems.append({'step': step, 'id': row['id'], 'reason': 'Backward target/order/token mismatch'})
            completed += row['backward_tokens']
            if row['batch_completed_tokens'] != completed or row['batch_target_tokens'] != total:
                problems.append({'step': step, 'id': row['id'], 'reason': 'Backward cumulative accounting mismatch'})
            if not math.isfinite(row['loss_contribution']):
                problems.append({'step': step, 'id': row['id'], 'reason': 'Nonfinite actual backward loss'})
            loss += row['loss_contribution']
        checkpoint = root / 'train/roft' / f'checkpoint-{step:03d}' / 'audit.json'
        saved = checkpoint.exists()
        if saved:
            audit = load(checkpoint)
            if len(rows) != len(expected) or completed != total or completed != audit['supervised_tokens']:
                problems.append({'step': step, 'reason': 'Saved update lacks every completed selected backward'})
            if [row['id'] for row in rows] != audit['target_ids']:
                problems.append({'step': step, 'reason': 'Saved update target identities differ from actual backwards'})
            if not math.isclose(loss, audit['loss'], rel_tol=1e-7, abs_tol=1e-9):
                problems.append({'step': step, 'reason': 'Saved loss differs from actual backward contributions'})
            saved_total += completed
        steps.append({'step': step, 'completed_backwards': len(rows), 'selected_targets': len(expected),
            'actual_backward_tokens': completed, 'selected_backward_tokens': total,
            'saved_numerical_update': saved, 'actual_loss_contribution_sum': loss})
    return {'scope': 'all persisted native backward receipts reconciled with actual target IDs and saved update audits',
        'steps': steps, 'saved_backward_tokens': saved_total,
        'pending_backward_tokens': sum(s['actual_backward_tokens'] for s in steps if not s['saved_numerical_update']),
        'problems': problems, 'verified': not problems, 'read_only': True,
        'note': 'An unfinished update remains pending; this audit alone does not prove gradients, optimizer states or final quality.'}


def audit(root):
    path = root / 'train/roft/microbatches.jsonl'
    return reconcile(root, [json.loads(line) for line in path.read_text().splitlines() if line])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))
