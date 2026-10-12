"""Share identical immutable local backup bytes; preserve every path and hash."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('--through-step', type=int, required=True)
    args = parser.parse_args()
    root = args.artifacts / 'remote'
    free_before = shutil.disk_usage(root).free
    entries = []
    for step in range(1, args.through_step + 1):
        receipt = json.loads((args.artifacts / 'cycle_backup_audits' / f'cycle-{step:03d}.json').read_text())
        if not receipt['verified']:
            raise RuntimeError('Only previously hash-verified finished backups qualify')
        for name, expected in receipt['source_manifest'].items():
            # These records are immutable after source completion; exclude live
            # episodes, model weights, arbitrary task files and all executable data.
            path = root / name
            eligible = (name.startswith('generations/') or
                name.startswith('episodes/') and name.endswith('/response.json'))
            if eligible and path.stat().st_size >= 1024 * 1024:
                entries.append((path, name, expected))
    canonical = {}
    changed = []
    checked = []
    for path, name, expected in entries:
        if path.is_symlink() or sha(path) != expected:
            raise RuntimeError('Original backup hash mismatch: ' + name)
        checked.append(name)
        if expected not in canonical:
            canonical[expected] = (path, name)
            continue
        source, source_name = canonical[expected]
        if os.path.samefile(source, path):
            continue
        original = path.stat()
        original_source = source.stat()
        if (original.st_dev, original.st_mode) != (
                original_source.st_dev, original_source.st_mode):
            continue
        temporary = path.with_name(path.name + '.verified-identical-link.tmp')
        if temporary.exists():
            raise RuntimeError('Existing temporary backup link requires inspection')
        try:
            os.link(source, temporary)
            if sha(temporary) != expected or sha(path) != expected:
                raise RuntimeError('Backup changed during identical-byte verification')
            os.replace(temporary, path)
            if sha(path) != expected:
                raise RuntimeError('Post-replacement hash mismatch')
        finally:
            temporary.unlink(missing_ok=True)
        changed.append({'path': name, 'identical_to': source_name,
            'sha256': expected, 'bytes': original.st_size,
            'original_local_mtime_ns': original.st_mtime_ns,
            'shared_local_mtime_ns': original_source.st_mtime_ns,
            'remote_original_timestamp_unchanged': True})
    result = {'at': time.time(), 'scope': 'finished hash-verified local backup copies only',
        'through_step': args.through_step, 'checked_records': len(checked),
        'shared_storage_records': changed, 'free_bytes_before': free_before,
        'free_bytes_after': shutil.disk_usage(root).free,
        'all_original_paths_and_bytes_preserved': True, 'remote_originals_changed': False,
        'training_or_candidate_outputs_changed': False, 'verified': True}
    target = args.artifacts / f'backup_identical_storage_through_{args.through_step:03d}.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'shared_storage_records'}))
    print('Identical records sharing storage:', len(changed))


if __name__ == '__main__':
    main()
