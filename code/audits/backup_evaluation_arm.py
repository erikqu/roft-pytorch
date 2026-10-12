"""Back up one fully completed evaluation arm while the other keeps running."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RUN, load, save, sha
from cloud import REMOTE, ssh


def remaining():
    seconds = load(RUN / 'work_window.json')['deadline_unix'] - time.time()
    if seconds <= 0:
        raise TimeoutError('Original work deadline ended; preserve existing evidence')
    return seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=['roft', 'base'])
    parser.add_argument('--snapshot-completed', action='store_true',
                        help='Preserve all currently graded episodes; never claim full-arm completion')
    args = parser.parse_args()
    arm = args.arm
    snippet = """from pathlib import Path
import hashlib,json
root=Path(%r);arm=%r;snapshot=%r
receipt=root/'evaluation'/(arm+'-complete.json')
jobfiles=sorted((root/'jobs').glob('evaluation-'+arm+'-*.json'))
assert len(jobfiles)==1
jobs=json.loads(jobfiles[0].read_text())
data={'jobs':[job['id'] for job in jobs]}
assert len(data['jobs'])==34 and len(set(data['jobs']))==34
if snapshot:
 data['jobs']=[identity for identity in data['jobs']
  if (root/'generations'/(identity+'.json')).is_file()
  and (root/'episodes'/identity/'result.json').is_file()]
 assert data['jobs'], 'No genuinely completed/graded episodes to preserve'
 paths=[]
else:
 published=json.loads(receipt.read_text())
 assert set(published['jobs'])==set(data['jobs'])
 assert published['provenance']=='fresh inference'
 paths=[receipt]
for identity in data['jobs']:
 assert identity.startswith('evaluation-'+arm+'-')
 generation=root/'generations'/(identity+'.json')
 assert generation.is_file() and (root/'episodes'/identity/'result.json').is_file()
 paths.append(generation)
 paths.extend((root/'raw_turns').glob(identity+'-*.json'))
 paths.extend(p for p in (root/'episodes'/identity).rglob('*') if p.is_file())
paths.extend((root/'jobs').glob('evaluation-'+arm+'-*.json'))
result={};sizes={}
for path in paths:
 name=str(path.relative_to(root));h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 result[name]=h.hexdigest();sizes[name]=path.stat().st_size
print(json.dumps({'source_manifest':result,'sizes':sizes,'jobs':data['jobs']}))
""" % (REMOTE + '/run', arm, args.snapshot_completed)
    output = subprocess.run(ssh() + ['python3 -c ' + shlex.quote(snippet)],
                            capture_output=True, text=True, check=True,
                            timeout=min(180, remaining()))
    source = json.loads(output.stdout)
    manifest = source['source_manifest']
    for name in manifest:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or '\n' in name or '\r' in name:
            raise RuntimeError('Invalid remote evidence path')
    target = RUN / 'remote'
    missing_bytes = sum(source['sizes'][name] for name in manifest
                        if not (target / name).is_file())
    if __import__('shutil').disk_usage(RUN).free < missing_bytes + 2 * 1024**3:
        raise RuntimeError('Insufficient host space; existing artifacts retained')
    listing = RUN / (arm + '-evaluation-backup-files.txt')
    listing.write_text('\n'.join(sorted(manifest)) + '\n')
    subprocess.run(['rsync', '-a', '--timeout=60', '--files-from=' + str(listing),
                    '-e', shlex.join(ssh()[:-1]), ssh()[-1] + ':' + REMOTE + '/run/',
                    str(target) + '/'], check=True, timeout=remaining())
    mismatches = []
    for name, expected in manifest.items():
        remaining()
        if not (target / name).is_file() or sha(target / name) != expected:
            mismatches.append(name)
    receipt = {'at': time.time(), 'arm': arm, 'verified': not mismatches,
               'completed_frozen_episodes': len(source['jobs']),
               'files': len(manifest), 'bytes': sum(source['sizes'].values()),
               'source_manifest': manifest, 'missing_or_mismatched': mismatches,
               'full_candidates_raw_turns_verifiers_and_app_artifacts': True,
               'read_only_remote_source': True, 'full_run_backup_claimed': False,
               'full_arm_complete_claimed': not args.snapshot_completed,
               'all_currently_graded_episodes_snapshot': args.snapshot_completed,
               'automatic_shutdown': False}
    suffix = '-evaluation-snapshot-backup-audit.json' if args.snapshot_completed else '-evaluation-backup-audit.json'
    save(RUN / (arm + suffix), receipt)
    if mismatches:
        raise RuntimeError('Completed evaluation backup differs from original bytes')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'source_manifest'}), flush=True)


if __name__ == '__main__':
    main()
