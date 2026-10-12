"""Complete final serving/master/Adam/RNG backup with admitted range reads."""
from pathlib import Path
import json
import shlex
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RUN, load, save, sha
from cloud import ssh, REMOTE
from parallel_file_backup import transfer


def main():
    if not load(RUN / 'parallel_file_backup_qualification.json')['verified']:
        raise RuntimeError('Actual byte-range qualification missing')
    remote_proof = load(RUN / 'actual_remote_parallel_backup_qualification.json')
    if not remote_proof['verified'] or not remote_proof['exact_original_full_hash_verified']:
        raise RuntimeError('Actual remote multi-stream range qualification missing')
    deadline = load(RUN / 'work_window.json')['deadline_unix']
    complete = load(RUN / 'remote/train/roft/complete.json')
    if complete['backward_tokens'] != 256721:
        raise RuntimeError('Exact saved training is incomplete')
    folder = Path(complete['checkpoint']).name
    source = REMOTE + '/run/train/roft/' + folder
    target = RUN / 'remote/train/roft' / folder
    snippet = "from pathlib import Path;import json;p=Path(" + repr(source) + ");print(json.dumps({x.name:x.stat().st_size for x in p.iterdir() if x.is_file()}))"
    result = subprocess.run(ssh() + ['python3 -c ' + shlex.quote(snippet)],
                            capture_output=True, text=True, check=True, timeout=30)
    sizes = json.loads(result.stdout)
    required = sum(size for name, size in sizes.items() if not (target / name).is_file())
    # At this stage all candidate/raw transport and lossless compaction ended.
    # Range assembly uses one sparse temporary file, not duplicate chunk files.
    # Retain 1 GiB for small audit/metadata writes after the final immutable copy.
    if not load(RUN / 'raw_backup_audit.json')['verified']:
        raise RuntimeError('Final-state capacity decision requires finished verified raw backup')
    reserve = 1024**3
    save(RUN / 'final_state_capacity_admission.json', {
        'at': time.time(), 'required_new_bytes': required,
        'host_free_bytes': shutil.disk_usage(RUN).free, 'reserved_audit_bytes': reserve,
        'raw_transport_complete': True, 'parallel_chunks_duplicate_storage': False})
    if shutil.disk_usage(RUN).free < required + reserve:
        raise RuntimeError('Full state capacity unavailable; keep all evidence')
    audit = load(target / 'audit.json')
    hashes = load(target / 'weight_manifest.json')
    hashes['resume.pt.gz'] = audit['resume_sha256']
    results = {}
    for name, expected in hashes.items():
        if time.time() >= deadline:
            raise TimeoutError('Approved work deadline ended')
        save(RUN / 'parallel_final_state_backup_status.json', {
            'at': time.time(), 'copying_file': name, 'completed_files': list(results),
            'bytes': sizes[name], 'deadline_unix': deadline, 'automatic_shutdown': False})
        results[name] = transfer(source + '/' + name, target / name,
                                 sizes[name], expected, deadline, workers=8)
    save(RUN / 'roft-final-state-backup.json', {'at': time.time(), 'verified': True,
        'checkpoint': str(target), 'weight_manifest_sha256': sha(target / 'weight_manifest.json'),
        'resume_sha256': audit['resume_sha256'], 'full_weights_optimizer_rng': True,
        'actual_parallel_range_transfers': results, 'automatic_shutdown': False})
    print(json.dumps({'verified': True, 'full_weights_optimizer_rng': True,
                      'files': len(results)}), flush=True)


if __name__ == '__main__':
    main()
