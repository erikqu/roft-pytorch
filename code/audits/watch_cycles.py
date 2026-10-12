"""Bounded off-node cycle audit/backup observer; never changes training admission."""
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RUN, load, save


def command(arguments, output=None):
    deadline = load(RUN / 'work_window.json')['deadline_unix']
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError('Original observer work window ended')
    result = subprocess.run([sys.executable, *map(str, arguments)],
        capture_output=True, text=True, timeout=remaining)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    if output:
        save(output, json.loads(result.stdout))


def main():
    root = RUN / 'remote'
    scripts = Path(__file__).resolve().parent
    output = RUN / 'cycle_audits'
    output.mkdir(exist_ok=True)
    lock = (output / 'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = load(RUN / 'work_window.json')['deadline_unix']
    save(output / 'watcher.json', {'pid': __import__('os').getpid(), 'at': time.time(),
        'deadline_unix': deadline, 'read_only_training': True, 'automatic_shutdown': False})
    while time.time() < deadline:
        cfg = load(RUN / 'protocol.json')
        for selection in sorted((root / 'train/roft').glob('selection-*.json')):
            step = int(selection.stem.split('-')[1])
            receipt = output / f'cycle-{step:03d}-complete.json'
            failure = output / f'cycle-{step:03d}-failure.json'
            if receipt.exists() or failure.exists():
                continue
            # Metadata can arrive in several sync waves. Wait for the complete set.
            sources = list((root / 'generations').glob(f'roft-{step:03d}-attempt-*.json'))
            original = [p for p in sources if 'reflection' not in p.name]
            reflected = [p for p in sources if 'reflection' in p.name]
            if len(original) != cfg['source_attempts_per_update'] or len(reflected) != len(original) * cfg['reflections_per_attempt']:
                continue
            if not (root / 'jobs' / f'roft-{step:03d}-targets.json').exists():
                continue
            if any(not (root / 'episodes' / p.stem / 'result.json').exists() for p in original):
                continue
            try:
                backup = RUN / 'cycle_backup_audits' / f'cycle-{step:03d}.json'
                if not backup.exists() or not load(backup)['verified']:
                    command([scripts / 'backup_cycle.py', step])
                command([scripts / 'audit_sources.py', root, '--step', step], output / f'cycle-{step:03d}-sources.json')
                command([scripts / 'audit_reflections.py', root, '--step', step], output / f'cycle-{step:03d}-reflections.json')
                tokenizer = root / 'train/roft/checkpoint-001'
                command([scripts / 'audit_target_tokens.py', root, '--step', step, '--model', tokenizer], output / f'cycle-{step:03d}-tokens.json')
                source = load(output / f'cycle-{step:03d}-sources.json')
                reflections = load(output / f'cycle-{step:03d}-reflections.json')
                tokens = load(output / f'cycle-{step:03d}-tokens.json')
                verified = (not source['identity_mismatches'] and not source['missing_rewards'] and
                    not source.get('recorded_reward_file_mismatches', []) and
                    not reflections['problems'] and reflections['all_sources_and_four_reflections_present'] and tokens['verified'])
                save(receipt, {'at': time.time(), 'step': step, 'verified': verified,
                    'original_sources': len(original), 'fresh_reflections': len(reflected),
                    'supervised_tokens_selected': tokens['actual_supervised_tokens'],
                    'complete_raw_backup_verified': load(backup)['verified'],
                    'optimizer_update_verified_by_this_receipt': False,
                    'read_only_training': True, 'automatic_shutdown': False})
                print('CYCLE_AUDITED_AND_BACKED_UP', step, verified, flush=True)
            except Exception as error:
                save(failure, {'at': time.time(), 'step': step, 'error': str(error),
                    'training_stopped': False, 'automatic_shutdown': False})
                print('CYCLE_AUDIT_FAILURE', step, str(error), flush=True)
        if (root / 'worker_complete.json').exists() or (root / 'failure.json').exists():
            break
        time.sleep(min(20, max(0, deadline - time.time())))
    save(output / 'watcher_complete.json', {'at': time.time(), 'automatic_shutdown': False})


if __name__ == '__main__':
    main()
