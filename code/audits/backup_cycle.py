"""Back up a fully generated source/reflection cycle without touching live training."""
import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RUN, save, sha
from cloud import REMOTE, ssh


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', type=int)
    args = p.parse_args()
    prefix = f'roft-{args.step:03d}-attempt-'
    code = '''from pathlib import Path
import hashlib,json
root=Path(%s);prefix=%s
sources=[p for p in (root/'generations').glob(prefix+'*.json') if 'reflection' not in p.name]
cfg=json.loads((root/'protocol.json').read_text())
assert len(sources)==cfg['source_attempts_per_update'], 'Source batch incomplete'
for p in sources:
 assert (root/'episodes'/p.stem/'result.json').exists(), 'Actual verifier missing'
 assert len(list((root/'generations').glob(p.stem+'-reflection-*.json')))==cfg['reflections_per_attempt'], 'Reflection set incomplete'
paths=list((root/'generations').glob(prefix+'*.json'))+list((root/'raw_turns').glob(prefix+'*.json'))
for folder in (root/'episodes').glob(prefix+'*'):
 paths.extend(p for p in folder.rglob('*') if p.is_file())
result={}
for path in paths:
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 result[str(path.relative_to(root))]=h.hexdigest()
print(json.dumps(result))
''' % (repr(REMOTE + '/run'), repr(prefix))
    result = subprocess.run(ssh() + ['python3 -c ' + shlex.quote(code)],
        capture_output=True, text=True, check=True, timeout=180)
    manifest = json.loads(result.stdout)
    folder = RUN / 'cycle_backup_audits'
    folder.mkdir(exist_ok=True)
    listing = folder / f'cycle-{args.step:03d}-files.txt'
    listing.write_text('\n'.join(sorted(manifest)) + '\n')
    target = RUN / 'remote'
    target.mkdir(exist_ok=True)
    subprocess.run(['rsync', '-a', '--timeout=60', '--files-from=' + str(listing),
        '-e', shlex.join(ssh()[:-1]), ssh()[-1] + ':' + REMOTE + '/run/', str(target) + '/'],
        check=True, timeout=1800)
    failed = [name for name, expected in manifest.items()
        if not (target / name).exists() or sha(target / name) != expected]
    save(folder / f'cycle-{args.step:03d}.json', {'at': time.time(),
        'step': args.step, 'verified': not failed, 'files': len(manifest),
        'missing_or_mismatched': failed, 'source_manifest': manifest,
        'full_candidates_raw_turns_reflections_verifiers': True,
        'optimizer_state_included': False, 'read_only_remote_source': True})
    if failed:
        raise RuntimeError('Source/reflection cycle evidence backup mismatch')
    print('FULL_CYCLE_EVIDENCE_BACKUP_VERIFIED', args.step, len(manifest), flush=True)


if __name__ == '__main__':
    main()
