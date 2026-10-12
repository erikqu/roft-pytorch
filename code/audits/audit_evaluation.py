"""Check frozen evaluation membership and actual candidate/reward provenance."""
import argparse
import hashlib
import json
from pathlib import Path
from raw_evidence_io import load_json, has_json, json_paths


def load(path):
    return load_json(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha(path):
    # Transport preserves the node's public-task symlink, which is not usable
    # locally. The admitted host snapshot contains the same frozen image bytes.
    expected = None
    if not path.is_file():
        root = next((p for p in path.parents if p.name == 'remote'), None)
        if root is not None:
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0] == 'public_tasks':
                manifest = load(root.parent / 'manifest.json')
                expected = manifest['files'].get(str(relative))
                if expected is None:
                    raise RuntimeError('Prompt image is absent from frozen manifest')
                path = root.parent / relative
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    value = h.hexdigest()
    if expected is not None and value != expected:
        raise RuntimeError('Original frozen host prompt-image bytes differ')
    return value


def audit_raw_turns(root, row, cfg, prompt):
    """Compare episode history to tokens persisted before tool execution/grading."""
    problems = []
    image_hashes = [sha(root / name) for name in prompt['images']]
    if row['task_description'] != prompt['text']:
        problems.append({'id': row['id'], 'reason': 'Original frozen task text changed'})
    if row['response_sha256'] != digest({'turns': row['turns']}):
        problems.append({'id': row['id'], 'reason': 'Actual episode response digest differs'})
    for index, turn in enumerate(row['turns']):
        path = root / 'raw_turns' / (row['id'] + f'-{index:03d}.json')
        if not has_json(path):
            problems.append({'id': turn['id'], 'reason': 'Pre-grading raw turn missing'})
            continue
        raw = load(path)
        if any(turn.get(key) != value for key, value in raw.items()):
            problems.append({'id': turn['id'], 'reason': 'Episode differs from pre-grading raw turn'})
        if raw['id'] != row['id'] + f'/turn-{index:03d}':
            problems.append({'id': turn['id'], 'reason': 'Raw turn identity/index differs'})
        if raw['seed'] != row['seed'] + index * 1000003:
            problems.append({'id': turn['id'], 'reason': 'Raw sampling seed differs from frozen seed'})
        if len(raw['token_ids']) != len(raw['sampled_raw_logprobs']):
            problems.append({'id': turn['id'], 'reason': 'Raw sampled log-probability coverage differs'})
        expected_prompt = digest({'ids': raw['prompt_token_ids'], 'images': image_hashes})
        if raw['prompt_sha256'] != expected_prompt:
            problems.append({'id': turn['id'], 'reason': 'Actual raw prompt/image digest differs'})
        if len(raw['prompt_token_ids']) + len(raw['token_ids']) > cfg['max_context_tokens']:
            problems.append({'id': turn['id'], 'reason': 'Actual raw context exceeds frozen budget'})
    return problems


def audit(root):
    cfg = load(root / 'protocol.json')
    prompts = {p['task']: p for p in load(root / 'prompts.json')}
    expected = {(split, task, seed)
        for split, tasks, seeds in [
            ('transfer', cfg['transfer'], cfg['transfer_seeds']),
            ('within_task', cfg['within_task_tasks'], cfg['within_task_seeds'])]
        for task in tasks for seed in seeds}
    problems = []
    arms = {}
    manifests = {}
    first_prompt_hashes = {}
    for arm in ['roft', 'base']:
        receipt_path = root / 'evaluation' / (arm + '-complete.json')
        rows = [load(p) for p in json_paths(root / 'generations', 'evaluation-' + arm + '-*.json')]
        membership = [(r['split'], r['task'], r['seed']) for r in rows]
        if len(set(membership)) != len(membership):
            problems.append({'arm': arm, 'reason': 'Repeated evaluation member'})
        if not set(membership) <= expected:
            problems.append({'arm': arm, 'reason': 'Unfrozen evaluation membership'})
        identity = None
        if arm == 'roft' and (root / 'train/roft/complete.json').exists():
            complete = load(root / 'train/roft/complete.json')
            # Local backups have a different root; final folder name is pinned.
            folder = root / 'train/roft' / Path(complete['checkpoint']).name
            identity = digest(load(folder / 'weight_manifest.json'))
        scored = []
        missing_scores = []
        first_prompt_hashes[arm] = {}
        for row in rows:
            rid = row['id']
            member = (row['split'], row['task'], row['seed'])
            if row['task'] in prompts:
                problems.extend(audit_raw_turns(root, row, cfg, prompts[row['task']]))
            else:
                problems.append({'id': rid, 'reason': 'Task absent from frozen prompt snapshot'})
            first_prompt_hashes[arm][member] = row['prompt_sha256']
            if row['provenance'] != 'fresh inference' or row['phase'] != 'evaluation':
                problems.append({'id': rid, 'reason': 'Evaluation is not fresh inference'})
            if row['protocol_sha256'] != sha(root / 'protocol.json'):
                problems.append({'id': rid, 'reason': 'Protocol changed'})
            if row['checkpoint_sha256'] != identity:
                problems.append({'id': rid, 'reason': 'Wrong final/base checkpoint'})
            if len(row['turns']) > cfg['max_solver_turns'] or row['output_tokens'] > cfg['max_solver_total_tokens']:
                problems.append({'id': rid, 'reason': 'Solver work exceeds frozen evaluation budget'})
            if row['output_tokens'] != sum(len(t['token_ids']) for t in row['turns']):
                problems.append({'id': rid, 'reason': 'Actual episode token accounting differs'})
            initial = row['turns'][0]['sampling_messages'] if row['turns'] else []
            if len(initial) != 2 or [m['role'] for m in initial] != ['system', 'user']:
                problems.append({'id': rid, 'reason': 'Evaluation initial context includes extra reflection/history'})
            if row.get('reflection') or row.get('attempt_id'):
                problems.append({'id': rid, 'reason': 'Evaluation conditioned on a stored training attempt/reflection'})
            for turn in row['turns']:
                if len(turn['token_ids']) > cfg['max_response_tokens']:
                    problems.append({'id': turn['id'], 'reason': 'Per-turn output exceeds frozen budget'})
                if turn['checkpoint_sha256'] != identity:
                    problems.append({'id': turn['id'], 'reason': 'Mid-episode checkpoint substitution'})
                if turn['sampling_messages'][0]['role'] != 'system':
                    problems.append({'id': turn['id'], 'reason': 'System prompt missing'})
            grade_path = root / 'episodes' / rid / 'result.json'
            if not has_json(grade_path):
                missing_scores.append(rid)
                continue
            grade = load(grade_path)
            if grade['response_sha256'] != row['response_sha256']:
                problems.append({'id': rid, 'reason': 'Graded candidate changed'})
            reward_path = root / 'episodes' / rid / 'grader/verifier/reward.json'
            if not reward_path.exists() or sha(reward_path) != grade['reward_file_sha256']:
                problems.append({'id': rid, 'reason': 'Actual verifier reward file missing or altered'})
            elif load(reward_path).get('reward') != grade['reward']:
                problems.append({'id': rid, 'reason': 'Recorded reward differs from actual verifier reward file'})
            scored.append({'id': rid, 'reward': grade['reward'],
                'termination': row['termination_reason'], 'checkpoint_identity': identity})
        receipt = load(receipt_path) if receipt_path.exists() else None
        if receipt:
            if set(receipt['jobs']) != {r['id'] for r in rows}:
                problems.append({'arm': arm, 'reason': 'Receipt/output ID mismatch'})
            if set(membership) != expected or missing_scores:
                problems.append({'arm': arm, 'reason': 'Completed receipt lacks full frozen set and actual scores'})
        manifests[arm] = {m for m in membership}
        arms[arm] = {'provenance': 'fresh inference' if rows else 'not measured',
            'expected_episodes': len(expected), 'actual_candidate_episodes': len(rows),
            'actual_verifier_results': len(scored), 'completion_receipt': receipt is not None,
            'missing_verifier_results': missing_scores,
            'full_set_present': set(membership) == expected,
            'actual_rewards': scored}
    full = all(a['completion_receipt'] and a['full_set_present'] and
        a['actual_verifier_results'] == len(expected) for a in arms.values())
    matched_prompts = 0
    for member in first_prompt_hashes['roft'].keys() & first_prompt_hashes['base'].keys():
        if first_prompt_hashes['roft'][member] != first_prompt_hashes['base'][member]:
            problems.append({'member': member, 'reason': 'ROFT/base first multimodal prompt hashes differ'})
        else:
            matched_prompts += 1
    return {'arms': arms, 'problems': problems, 'paired_full_set_verified': full and not problems,
        'frozen_membership_size_per_arm': len(expected), 'paired_initial_prompt_hashes_matched': matched_prompts,
        'pregrading_raw_turn_seed_context_prompt_image_and_response_digests_checked': True,
        'read_only': True,
        'historical_scores_substituted': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    args = p.parse_args()
    print(json.dumps(audit(args.root), indent=2))
