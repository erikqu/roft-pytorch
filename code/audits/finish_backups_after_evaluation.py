"""Finish verified backups after the completed run releases its controller lock.

No sampling, training, deadline edits, candidate replacement, or node control.
Existing local evidence may be stored losslessly with its original hash mapping.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from common import RUN, load, save
from cloud import owned
from controller import backup_final_states


def deadline():
    return load(RUN / 'work_window.json')['deadline_unix']


def command(script, *arguments):
    remaining = deadline() - time.time()
    if remaining <= 0:
        raise TimeoutError('Original admitted work window ended; backups remain incomplete')
    subprocess.run([sys.executable, '-u', str(CODE / 'audits' / script), *map(str, arguments)],
                   check=True, timeout=remaining)


def finish():
    root = RUN / 'remote'
    complete = load(root / 'worker_complete.json')
    training = load(root / 'train/roft/complete.json')
    if not complete['token_equal_training'] or not complete['fresh_evaluation_all_arms'] or \
            training['backward_tokens'] != 256721:
        raise RuntimeError('Exact training and full fresh evaluation are prerequisites')
    for arm in ('roft', 'base'):
        receipt = load(root / 'evaluation' / (arm + '-complete.json'))
        if len(receipt['jobs']) != 34 or len(set(receipt['jobs'])) != 34:
            raise RuntimeError('Full frozen evaluation receipt missing')
    for step in range(1, training['updates'] + 1):
        receipt = root / 'audits/optimizer_states' / f'checkpoint-{step:03d}.json'
        if step == 1 and not receipt.exists():
            receipt = RUN / 'recovered_optimizer_state_audit.json'
        state = load(receipt)
        if not state['verified'] or state['problems']:
            raise RuntimeError('Required saved-state numerical audit missing')
    save(RUN / 'postevaluation_backup_status.json', {
        'at': time.time(), 'state': 'compacting previously verified completed-cycle backups',
        'sampling_training_unchanged': True, 'node_kept_running': True})
    command('compact_finished_local_evidence.py', RUN, '--apply')
    # Fresh remote manifest verifies compressed originals before omitting them
    # from transport. New evaluation/app/failed/qualification artifacts are copied.
    command('backup_raw_preserving_storage.py')
    command('compact_finished_local_evidence.py', RUN, '--apply')
    if not (RUN / 'roft-final-state-backup.json').exists():
        command('backup_final_state_parallel.py')
    command('audit_final_state_backup.py', RUN)
    command('audit_local_raw_storage.py', RUN)
    for name in ('final_state_current_hash_audit.json', 'local_raw_storage_audit.json'):
        if not load(RUN / name)['verified']:
            raise RuntimeError('Current backup byte audit did not pass: ' + name)
    node = owned()
    save(RUN / 'node_kept_running.json', {
        'at': time.time(), 'state': node['status'] if node else 'missing',
        'instance_id': node['id'] if node else None, 'automatic_shutdown': False,
        'reason': 'Explicit user instruction; read-only post-backup inventory',
        'still_billable': bool(node)})
    save(RUN / 'postevaluation_backup_complete.json', {
        'at': time.time(), 'verified': True, 'full_final_state_and_raw_bytes': True,
        'node_running': bool(node and node['status'] == 'active'),
        'goal_completion_not_claimed': True})


def main():
    own_lock = (RUN / 'postevaluation_backup.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(RUN / 'postevaluation_backup_watcher.json', {
        'at': time.time(), 'pid': os.getpid(), 'waiting_for_complete_evaluation': True,
        'deadline_unix': deadline(), 'automatic_shutdown': False})
    controller_lock = (RUN / 'controller.lock').open('a')
    try:
        while time.time() < deadline():
            if (RUN / 'remote/worker_complete.json').is_file():
                try:
                    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(10)
                    continue
                finish()
                return
            time.sleep(15)
        raise TimeoutError('Admitted work window ended before full evaluation completion; node retained')
    except Exception as error:
        save(RUN / 'postevaluation_backup_failure.json', {
            'at': time.time(), 'error': str(error), 'node_kept_running': True,
            'all_existing_evidence_retained': True, 'goal_completion_not_claimed': True})
        raise


if __name__ == '__main__':
    main()
