"""Read-only audit of fresh reflection evidence; never changes target admission."""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import reflection_feedback, reflection_instruction, sha
from reflection_targets import bounded, trajectory


def load(path):
    return json.loads(path.read_text())


def audit(root, step):
    cfg = load(root / 'protocol.json')
    sources = {p.stem: load(p) for p in (root / 'generations').glob(
        f'roft-{step:03d}-attempt-*.json') if 'reflection' not in p.name}
    reflections = [load(p) for p in sorted((root / 'generations').glob(
        f'roft-{step:03d}-attempt-*-reflection-*.json'))]
    problems = []
    per_source = collections.Counter()
    finishes = collections.Counter()
    source_evidence = {}
    protocol_sha256 = sha(root / 'protocol.json')
    instruction = reflection_instruction()
    for row in reflections:
        rid = row['id']
        source = sources.get(row['attempt_id'])
        if source is None:
            problems.append({'id': rid, 'reason': 'original source missing'})
            continue
        per_source[source['id']] += 1
        finishes[row['finish_reason']] += 1
        grade_path = root / 'episodes' / source['id'] / 'result.json'
        if not grade_path.exists():
            problems.append({'id': rid, 'reason': 'actual verifier feedback missing'})
            continue
        if source['id'] not in source_evidence:
            grade = load(grade_path)
            expected_evidence = ('Recorded interaction:\n' + trajectory(source) +
                '\nFinal files/patch summary:\n' + bounded(source.get('file_summary', ''), 2000) +
                '\n' + bounded(reflection_feedback(grade), 3000))
            source_evidence[source['id']] = (expected_evidence,
                grade['response_sha256'], grade['raw_verdict'])
        expected_evidence, response_sha256, raw_verdict = source_evidence[source['id']]
        msg = row['sampling_messages']
        tests = {
            'checkpoint identity': row['checkpoint_sha256'] == source['checkpoint_sha256'],
            'policy version': row.get('policy_version') == step - 1,
            'unchanged protocol': row['protocol_sha256'] == protocol_sha256,
            'original task': msg[1]['content'][0]['text'] == bounded(source['task_description'], 4000),
            'paper reflection instruction': msg[0]['content'] == instruction,
            'actual bounded trajectory/patch/verifier': msg[2]['content'][0]['text'] == expected_evidence,
            'raw candidate identity': response_sha256 == source['response_sha256'],
            'unknown continuous binary verdict': raw_verdict == 'unknown',
            'single reflection turn': len(row['turns']) == 1,
            'raw reflection token identity': row['token_ids'] == row['turns'][0]['token_ids'],
            'fresh inference provenance': row['provenance'] == 'fresh inference',
        }
        for check, passed in tests.items():
            if not passed:
                problems.append({'id': rid, 'reason': check})
    missing_feedback = [sid for sid in sources if not (root / 'episodes' / sid / 'result.json').exists()]
    missing = {sid: cfg['reflections_per_attempt'] - per_source[sid]
               for sid in sources if per_source[sid] != cfg['reflections_per_attempt']}
    selection_path = root / 'train' / 'roft' / f'selection-{step:03d}.json'
    selection = load(selection_path) if selection_path.exists() else None
    return {
        'scope': 'all currently saved reflections for source cycle ' + str(step),
        'source_count': len(sources), 'reflection_count': len(reflections),
        'expected_reflections_per_source': cfg['reflections_per_attempt'],
        'reflection_counts': dict(per_source), 'missing_or_extra_reflections': missing,
        'sources_without_actual_verifier_feedback': missing_feedback,
        'finish_reasons': dict(finishes), 'problems': problems,
        'all_sources_and_four_reflections_present':
            len(sources) == cfg['source_attempts_per_update'] and not missing and not missing_feedback,
        'selected_target_audit_available': selection is not None,
        'selection': selection,
        'read_only': True, 'routing_or_training_changes': False,
        'note': 'Correct feedback and fresh generation do not establish a task-performance gain.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--step', type=int, default=1)
    args = p.parse_args()
    print(json.dumps(audit(args.root, args.step), indent=2))
