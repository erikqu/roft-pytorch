"""Freeze v5 input snapshot/code with explicit continuing user constraints."""
from common import *

def seal():
    if (RUN/'manifest.json').exists():
        # Resuming is not a new experiment. Explicit repair receipts amend the
        # existing manifest before launch; do not silently reseal changed files.
        validate_prepared()
        return
    files={name:sha(RUN/name) for name in ('protocol.json','prompts.json','dataset_identity.json','source_dataset_manifest.json')}
    for folder in ('tasks','public_tasks','source_harness'):
        for p in (RUN/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(RUN))]=sha(p)
    save(RUN/'manifest.json',{'files':files,'code':{p.name:sha(p) for p in CODE.glob('*.py')},
        'official':{p.name:sha(p) for p in (CODE/'official').iterdir() if p.is_file()},
        'source':'same v4 frozen Blender snapshot; explicit v5 tool/schedule fixes'})
if __name__=='__main__':seal();print('V5_SEALED')
