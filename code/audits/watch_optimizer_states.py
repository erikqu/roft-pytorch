"""Bounded CPU-only checkpoint auditor. No generation, training, or node control."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--base-model', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    output = root / 'audits' / 'optimizer_states'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    window = json.loads((root / 'work_window.json').read_text())
    deadline = window['deadline_unix']
    write(output / 'watcher.json', {'pid': os.getpid(), 'started': time.time(),
        'deadline_unix': deadline, 'read_only': True, 'automatic_shutdown': False})
    while time.time() < deadline:
        found = False
        for folder in sorted((root / 'train' / 'roft').glob('checkpoint-*')):
            if not (folder / 'audit.json').exists():
                continue
            receipt = output / (folder.name + '.json')
            failure = output / (folder.name + '-failure.json')
            if receipt.exists() or failure.exists():
                continue
            found = True
            write(output / 'status.json', {'at': time.time(), 'checkpoint': folder.name,
                'state': 'inspecting saved weights/optimizer/RNG', 'read_only': True})
            cmd = [sys.executable, '-u', str(Path(__file__).with_name('audit_optimizer_state.py')),
                str(folder), '--base-model', str(args.base_model)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                    timeout=max(1, deadline - time.time()))
                if r.returncode:
                    raise RuntimeError(r.stderr[-4000:])
                audit = json.loads(r.stdout)
                audit.update(audited_at=time.time())
                write(receipt, audit)
                print('SAVED_STATE_AUDITED', folder.name, audit['verified'], flush=True)
            except Exception as error:
                write(failure, {'at': time.time(), 'checkpoint': folder.name,
                    'error': str(error), 'read_only': True, 'automatic_shutdown': False})
                print('SAVED_STATE_AUDIT_ERROR', folder.name, str(error), flush=True)
        if (root / 'worker_complete.json').exists() and not found:
            break
        if (root / 'failure.json').exists() and not found:
            break
        time.sleep(min(10, max(0, deadline - time.time())))
    write(output / 'watcher_complete.json', {'at': time.time(), 'read_only': True,
        'automatic_shutdown': False, 'node_termination_requested': False})


if __name__ == '__main__':
    main()
