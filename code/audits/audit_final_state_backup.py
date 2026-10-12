"""Hash current off-node final weights/state against the actually audited run."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(artifacts):
    root = artifacts / 'remote'
    receipt_path = artifacts / 'roft-final-state-backup.json'
    complete_path = root / 'train/roft/complete.json'
    if not receipt_path.exists() or not complete_path.exists():
        return {'at': time.time(), 'verified': False, 'read_only': True,
                'problems': ['Full final training/backup receipt is not present']}
    complete = load(complete_path)
    receipt = load(receipt_path)
    folder = root / 'train/roft' / Path(complete['checkpoint']).name
    problems = []
    if complete.get('backward_tokens') != 256721:
        problems.append('Final training token count is incomplete')
    if not receipt.get('verified') or not receipt.get('full_weights_optimizer_rng'):
        problems.append('Full state receipt is unverified')
    if Path(receipt.get('checkpoint', '')).resolve() != folder.resolve():
        problems.append('Backup receipt does not identify this final checkpoint')
    required = ['audit.json', 'weight_manifest.json', 'resume.pt.gz']
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        return {'at': time.time(), 'verified': False, 'read_only': True,
                'problems': problems + ['Required final-state files missing: ' + ', '.join(missing)]}
    update = load(folder / 'audit.json')
    weights = load(folder / 'weight_manifest.json')
    step = update['step']
    state_path = root / 'audits/optimizer_states' / f'checkpoint-{step:03d}.json'
    state = load(state_path) if state_path.exists() else None
    if not state or not state.get('verified') or state.get('problems') or \
            state.get('resume_sha256') != update['resume_sha256'] or \
            state.get('saved_master_tensors') != 723 or state.get('adam_moment_tensors') != 723 or \
            state.get('adam_step_counters') != [step] or not state.get('cpu_cuda_rng_valid'):
        problems.append('Copied state identity lacks its full masters/Adam/RNG numerical audit')
    if not update.get('numerical_reload_verified') or update.get('actual_changed_tensors') != 723 or \
            update.get('cumulative_backward_tokens') != 256721:
        problems.append('Final update numerical/token evidence is incomplete')
    manifest_sha = sha(folder / 'weight_manifest.json')
    if manifest_sha != update.get('weight_manifest_sha256') or manifest_sha != receipt.get('weight_manifest_sha256'):
        problems.append('Final weight manifest identity differs')
    hashed = {}
    for name, expected in weights.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            problems.append('Unsafe final model path')
            continue
        path = folder / name
        if not path.is_file():
            problems.append('Final model shard missing: ' + name)
            continue
        hashed[name] = sha(path)
        if hashed[name] != expected:
            problems.append('Current final model shard bytes differ: ' + name)
    hashed['resume.pt.gz'] = sha(folder / 'resume.pt.gz')
    if hashed['resume.pt.gz'] != update.get('resume_sha256') or \
            hashed['resume.pt.gz'] != receipt.get('resume_sha256'):
        problems.append('Current full optimizer/master/RNG bytes differ')
    return {'at': time.time(), 'verified': not problems, 'read_only': True,
            'checkpoint': str(folder), 'step': step, 'current_file_hashes': hashed,
            'full_weights_optimizer_rng': not problems, 'problems': problems,
            'all_model_shards_hashed': len(weights) == len(hashed) - 1,
            'complete_state_bytes_hashed': True,
            'master_adam_rng_parse_proof': str(state_path),
            'state_parse_proof_matched_by_full_bytes_sha256': bool(state and state.get('resume_sha256') == hashed['resume.pt.gz'])}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('artifacts', type=Path)
    args = p.parse_args()
    result = audit(args.artifacts)
    (args.artifacts / 'final_state_current_hash_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
