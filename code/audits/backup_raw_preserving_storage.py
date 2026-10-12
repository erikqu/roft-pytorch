"""Post-completion raw backup that preserves verified lossless local storage.

This is a manual backup tool, not part of the frozen sampling/training runtime.
Remote originals are read only. A compressed copy is skipped by transport only
after its decompressed bytes match the current remote source manifest.
"""
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RUN, save, sha
from cloud import REMOTE, ssh


def stream_sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def verify_compressed(artifacts, manifest):
    receipt = artifacts / 'compressed_raw_storage.json'
    records = json.loads(receipt.read_text()).get('records', []) if receipt.exists() else []
    verified = {}
    for row in records:
        name = row['path']
        if name not in manifest:
            continue
        if name in verified:
            raise RuntimeError('Duplicate compressed evidence record: ' + name)
        path = Path(row['compressed_path']).resolve()
        if not path.is_relative_to((artifacts / 'compressed_raw').resolve()):
            raise RuntimeError('Compressed evidence outside backup directory: ' + name)
        if row['sha256'] != manifest[name] or not path.is_file():
            raise RuntimeError('Compressed record differs from remote identity: ' + name)
        with gzip.open(path, 'rb') as stream:
            if stream_sha(stream) != manifest[name]:
                raise RuntimeError('Compressed original bytes differ: ' + name)
        verified[name] = row
    return verified


def safe_names(manifest):
    for name in manifest:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or '\n' in name or '\r' in name:
            raise RuntimeError('Invalid remote evidence path')


def main():
    code = """from pathlib import Path
import hashlib,json
root=Path(%r)
if not (root/'worker_complete.json').is_file():
 raise RuntimeError('Raw backup requires completed training and evaluation')
folders=['generations','raw_turns','episodes','evaluation','jobs','train','audits','qualification',
 'deadline_failure_archive','sync_interruption_archive']
folders += [p.name for p in root.glob('implementation_repair_*') if p.is_dir() and not p.is_symlink()]
files=[p for folder in folders
 for p in (root/folder).rglob('*')]
files += [p for p in root.iterdir() if p.is_file() and not p.is_symlink()]
result={}
for p in files:
 if p.is_file() and p.suffix!='.safetensors' and p.name!='resume.pt.gz' and not any(part.startswith(('.verified-resume-','.audit-state-reader-','.recovery-reader-')) for part in p.parts):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  result[str(p.relative_to(root))]=h.hexdigest()
print(json.dumps(result))
""" % (REMOTE + '/run')
    result = subprocess.run(ssh() + ['python3 -c ' + shlex.quote(code)],
                            capture_output=True, text=True, check=True, timeout=1200)
    manifest = json.loads(result.stdout)
    safe_names(manifest)
    compressed = verify_compressed(RUN, manifest)
    target = RUN / 'remote'
    target.mkdir(exist_ok=True)
    names = sorted(set(manifest) - set(compressed))
    listfile = RUN / 'raw-backup-preserving-storage-file-list.txt'
    listfile.write_text('\n'.join(names) + '\n')
    save(RUN / 'raw_backup_storage_transport_plan.json', {
        'at': time.time(), 'files': len(manifest), 'direct_transport_files': len(names),
        'verified_compressed_files': len(compressed), 'source_manifest': manifest,
        'sampling_or_training_changed': False, 'remote_originals_removed': False})
    subprocess.run(['rsync', '-a', '--files-from=' + str(listfile), '--timeout=60',
                    '-e', shlex.join(ssh()[:-1]), ssh()[-1] + ':' + REMOTE + '/run/',
                    str(target) + '/'], check=True, timeout=3600)
    failed = [name for name in names
              if not (target / name).is_file() or sha(target / name) != manifest[name]]
    # Re-read compressed bytes after transport; do not rely on a stale receipt.
    verify_compressed(RUN, manifest)
    save(RUN / 'raw_backup_audit.json', {
        'at': time.time(), 'verified': not failed, 'files': len(manifest),
        'direct_files': len(names), 'compressed_files': len(compressed),
        'missing_or_mismatched': failed, 'source_manifest': manifest,
        'compressed_original_bytes_verified': True, 'remote_originals_unchanged': True,
        'additional_scope': 'root regular files, qualification evidence, saved read-only audits and original implementation repair archives'})
    if failed:
        raise RuntimeError('Raw evidence backup failed')
    print(json.dumps({'verified': True, 'files': len(manifest),
                      'compressed_files': len(compressed)}), flush=True)


if __name__ == '__main__':
    main()
